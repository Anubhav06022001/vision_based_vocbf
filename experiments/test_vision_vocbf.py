#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
import mujoco as mj
import glfw
from pathlib import Path

# ===================== CONFIG =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene_cylinder.xml"

# Adjust paths to match your folder structure
from src.control.cbf_qp_multi import cbf_qp_osqp
from src.vocbf.vision_dynamics_net import VisionDynamicsBarrierNet
from src.utils.mujoco_ids import initialize_ids
from src.config.franka import q_min, q_max, simend, q_offsets
from src.utils.mouse_keyboard import MouseKeyboard
from src.safety.safety_function import ell_from_distance
from src.safety.distance_utils import compute_min_dist_from_q, get_robot_collision_geom_ids, get_obstacle_geom_ids
from src.utils.video_recorder import VideoRecorder

# Global state
reached_init = False
q_init_target = np.array([-2.0, 0.8, -1.8, -1.0, 2.5, 1.0, -2.1])
# q_init_target = np.array([-2.0, 0.8, -1.8, -1.0, 2.5, 1.0, 0])

# ================= MUJOCO & MODELS =================
model = mj.MjModel.from_xml_path(str(xml_path))
data = mj.MjData(model)

x_mean = torch.tensor(np.load(PROJECT_ROOT / "models" / "dyn_x_mean.npy"), device=DEVICE).float()
x_std  = torch.tensor(np.load(PROJECT_ROOT / "models" / "dyn_x_std.npy"), device=DEVICE).float()

B_net = VisionDynamicsBarrierNet(latent_dim=64, state_dim=14).to(DEVICE)
B_net.load_state_dict(torch.load(PROJECT_ROOT / "models" / "vision_vocbf_epoch_10.pt", map_location=DEVICE))
B_net.eval()

joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)
joint_idx = [joint_ids[f"joint{i}"] for i in range(1, 8)]
act_idx = [actuator_ids[f"actuator{i}"] for i in range(1, 8)]
cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')

# ================= PYTORCH WARMUP =================
print("Warming up PyTorch CUDA Context...")
dummy_img = torch.zeros(1, 3, 128, 128, device=DEVICE)
dummy_x = torch.zeros(1, 14, device=DEVICE)
dummy_u = torch.zeros(1, 7, device=DEVICE, requires_grad=True)
_ = torch.autograd.grad(B_net(dummy_img, dummy_x, dummy_u), dummy_u)[0]

# ================= GLFW INIT =================
glfw.init()
window = glfw.create_window(1200, 900, "Vision V-OCBF", None, None)
glfw.make_context_current(window)
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, int(mj.mjtFontScale.mjFONTSCALE_150.value))
mousekbd = MouseKeyboard(model, data, scene, cam)
cam.azimuth =180.86
cam.elevation = -15.95
cam.distance = 3.22
cam.lookat = np.array([0.0, 0.0, 0.0])

glfw.set_key_callback(window, mousekbd.keyboard)
glfw.set_cursor_pos_callback(window, mousekbd.mouse_move)
glfw.set_mouse_button_callback(window, mousekbd.mouse_button)
glfw.set_scroll_callback(window, mousekbd.scroll)

q_home = data.qpos[joint_idx].copy()

# Offscreen camera setup for reading pixels
offscreen_cam = mj.MjvCamera()
offscreen_cam.type = mj.mjtCamera.mjCAMERA_FIXED
offscreen_cam.fixedcamid = cam_id


viewport_width, viewport_height = glfw.get_framebuffer_size(window)
rec = VideoRecorder(viewport_width, viewport_height, fps=10)

while not glfw.window_should_close(window):
    time_prev = data.time
    
    # ================= VISION CONTROL (60Hz) =================
    q = data.qpos[joint_idx].copy()
    qd = data.qvel[joint_idx].copy()

    if not reached_init:
        data.ctrl[act_idx] = q_init_target
        if np.linalg.norm(q - q_init_target) < 0.1:
            reached_init = True
            print("Reached initial pose. Activating Vision Safety.")
    else:
        q_des = q_home + q_offsets
        
        # 1. Render wrist camera to a hidden 128x128 buffer
        mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        read_vport = mj.MjrRect(0, 0, 128, 128)
        mj.mjr_render(read_vport, scene, context)
        
        # 2. Read pixels directly from GLFW context
        rgb = np.zeros((128, 128, 3), dtype=np.uint8)
        mj.mjr_readPixels(rgb, None, read_vport, context)
        img = np.flipud(rgb).copy()

        # 3. Vision Safety Pass
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        x = torch.tensor(np.concatenate([q, qd]), device=DEVICE).float()
        x_norm = ((x - x_mean) / (x_std + 1e-6)).unsqueeze(0).requires_grad_(True)

        u_tensor = torch.tensor(q_des, dtype=torch.float32, device=DEVICE).unsqueeze(0).requires_grad_(True)
        B_val = B_net(img_t, x_norm, u_tensor)
        grad_B_u = torch.autograd.grad(B_val, u_tensor)[0].squeeze().detach().cpu().numpy()

        robot_geom_ids = get_robot_collision_geom_ids(model)
        obstacle_geom_ids = get_obstacle_geom_ids(model)
        min_dist = compute_min_dist_from_q(model, data, q, robot_geom_ids, obstacle_geom_ids)
        ell_val = ell_from_distance(min_dist, 0)
        print(" l:", float(ell_val),
            "B(u):", B_val.item(),
            "||grad_B_u||:", np.linalg.norm(grad_B_u),
            "ncon =", data.ncon)

        alpha = 100
        b_bound = np.dot(grad_B_u, q_des) - alpha * B_val.item()
        tau_safe = cbf_qp_osqp(u_ref=q_des, A_list=[grad_B_u], b_list=[b_bound])
        data.ctrl[act_idx] = tau_safe

    # ================= PHYSICS ENGINE =================
    while data.time - time_prev < 1.0 / 60.0:
        if not reached_init:
            data.ctrl[act_idx] = q_init_target
        mj.mj_step(model, data)
    
    if data.time >= simend: break

    # ================= MAIN RENDERING =================
    vw, vh = glfw.get_framebuffer_size(window)
    vport = mj.MjrRect(0, 0, vw, vh)
    
    # Render Main View (overwrites the 128x128 read buffer so you don't see it)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(vport, scene, context)

    # Render Inset Wrist View (Visual only, 320x240)
    inset_w, inset_h = 320, 240
    inset_vport = mj.MjrRect(vw - inset_w, vh - inset_h, inset_w, inset_h)
    mj.mjv_updateScene(model, data, opt, None, offscreen_cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(inset_vport, scene, context)
    rec.capture(vport, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
# rec.save(PROJECT_ROOT / "results" / "videos" /"ground_unsafe.mp4")