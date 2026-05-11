# # FOR RUNNING ON GPU-->export MUJOCO_GL=egl
#!/usr/bin/env python
import os
import sys
import mujoco as mj
import numpy as np
import h5py           # for image data in hdf5 files
from pathlib import Path

# --- Setup Paths & MuJoCo ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
xml_path = PROJECT_ROOT / "assets" / "franka_emika_panda" / "scene_cylinder.xml"

from src.utils.mujoco_ids import initialize_ids
from src.safety.distance_utils import compute_min_dist_from_q, get_robot_collision_geom_ids, get_obstacle_geom_ids
from src.kinematics.ik_utils import get_ik_goal
 

# ===================== CONFIG =====================
NUM_EPISODES = 200     
STEPS_PER_EPISODE = 150
SAVE_PATH = PROJECT_ROOT / "data" / "vision_vocbf" / "transition_dataset.h5"

# ================== MAIN SCRIPT ==================
def collect_data():
    model = mj.MjModel.from_xml_path(str(xml_path))
    data = mj.MjData(model)
    
    joint_ids, actuator_ids, site_ids, body_ids = initialize_ids(model)
    act_idx = [actuator_ids[f"actuator{i}"] for i in range(1, 8)]
    
    robot_geom_ids = get_robot_collision_geom_ids(model)
    obstacle_geom_ids = get_obstacle_geom_ids(model)
    
    # Setup Wrist Camera
    cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, 'wrist_camera')
    vision_renderer = mj.Renderer(model, height=128, width=128)
    
    # Data Buffers
    b_img, b_q, b_qd, b_u, b_dist, b_ncon = [], [], [], [], [], []
    
    print(f"Starting Data Collection: {NUM_EPISODES} episodes...")
    
    for ep in range(NUM_EPISODES):
        mj.mj_resetData(model, data)
        u_t = data.qpos[:7].copy() # Initial command is to stay still
        
        # Optional: Slightly randomize obstacle position here if you want robustness
        
        for step in range(STEPS_PER_EPISODE):
            # 1. Capture State
            q_t = data.qpos[:7].copy()
            qd_t = data.qvel[:7].copy()
            
            # 2. Capture Image
            vision_renderer.update_scene(data, camera=cam_id)
            img_t = vision_renderer.render()
            
            # 3. Capture Safety Metrics
            min_dist = compute_min_dist_from_q(model, data, q_t, robot_geom_ids, obstacle_geom_ids)
            # ncon_t = data.ncon # 0 means safe, >0 means collision

            # Inside your data collection loop:
            
            # Find if the obstacle is involved in ANY active collision
            obstacle_collision = False
            for i in range(data.ncon):
                contact = data.contact[i]
                geom1 = contact.geom1
                geom2 = contact.geom2
                
                # Check if either geometry belongs to the obstacle
                if geom1 in obstacle_geom_ids or geom2 in obstacle_geom_ids:
                    if geom1 in robot_geom_ids or geom2 in robot_geom_ids:
                        obstacle_collision = True
                        break
            
            ncon_t = 1 if obstacle_collision else 0
            
            # 4. Generate New Action using Cartesian IK (every 30 steps)
            if step % 30 == 0:
                # Bounding box in front of the robot where the obstacle lives
                target_x = np.random.uniform(0.3, 0.6)
                target_y = np.random.uniform(-0.3, 0.3)
                target_z = np.random.uniform(0.1, 0.5)
                u_t = get_ik_goal(model, data, np.array([target_x, target_y, target_z]))
                
            # Apply control and step
            data.ctrl[act_idx] = u_t
            mj.mj_step(model, data)
            
            # 5. Store in buffers
            b_img.append(img_t)
            b_q.append(q_t)
            b_qd.append(qd_t)
            b_u.append(u_t)         # The goal we gave the controller
            b_dist.append([min_dist])
            b_ncon.append([ncon_t]) # Crucial for anchoring the safety loss
            
        if ep % 10 == 0:
            print(f"Episode {ep}/{NUM_EPISODES} collected.")

    # ================== SAVE TO HDF5 ==================
    print(f"Saving {len(b_q)} transitions to HDF5...")
    with h5py.File(SAVE_PATH, 'w') as hf:
        hf.create_dataset('images', data=np.array(b_img, dtype=np.uint8), compression="gzip")
        hf.create_dataset('q', data=np.array(b_q, dtype=np.float32))
        hf.create_dataset('qd', data=np.array(b_qd, dtype=np.float32))
        hf.create_dataset('u', data=np.array(b_u, dtype=np.float32))
        hf.create_dataset('min_dist', data=np.array(b_dist, dtype=np.float32))
        hf.create_dataset('ncon', data=np.array(b_ncon, dtype=np.int32))
        
    print("Data collection complete and saved!")

if __name__ == "__main__":  
    collect_data()