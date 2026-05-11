import mujoco as mj
import numpy as np
import torch

# ===================== DISTANCE FUNCTION =====================


def get_robot_collision_geom_ids(model):
    ids = []
    for i in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        # if model.geom_contype[i] != 0 or model.geom_conaffinity[i] != 0:           # selects floor also
        if name and "_c" in name:   # collision geoms only
            ids.append(i)
    return ids



def get_obstacle_geom_ids(model):
    ids = []
    for i in range(model.ngeom):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, i)
        if name and "obstacle" in name:
            ids.append(i)
    return ids





def compute_min_dist_from_q(model, data, q, robot_ids, obs_ids, set_state=True):
    if isinstance(q, torch.Tensor):
        q = q.detach().cpu().numpy()

    if set_state:
        old = data.qpos.copy()
        data.qpos[:7] = q[:7]
        mj.mj_forward(model, data)

    distmax = 5.0
    min_dist = distmax

    for r in robot_ids:
        for o in obs_ids:
            d = mj.mj_geomDistance(model, data, r, o, distmax, None)
             # model --> set robot to specific configr from above FK to compute min for that pose
            if np.isnan(d):
                d = distmax   
            min_dist = min(min_dist, d)

    # apply FK again to recompute geom poses after
    #  restoring the original joint configuration so the simulator state remains consistent.
    if set_state:
        data.qpos[:] = old
        mj.mj_forward(model, data)

    return float(min_dist)



def compute_distance_gradient(model, data_tmp, q,
                             robot_geom_ids,
                             obstacle_geom_ids,
                             eps=1e-4):

    grad = np.zeros_like(q)

    # 🔥 CREATE TEMP DATA (CRITICAL FIX)
    data_tmp = mj.MjData(model)

    # base distance
    d0 = compute_min_dist_from_q(
        model, data_tmp, q,
        robot_geom_ids,
        obstacle_geom_ids,
        set_state= False
    )

    for i in range(len(q)):
        q_perturb = q.copy()
        q_perturb[i] += eps

        d1 = compute_min_dist_from_q(
            model, data_tmp, q_perturb,
            robot_geom_ids,
            obstacle_geom_ids,
            set_state= False
        )

        grad[i] = (d1 - d0) / eps

    print("d0:", d0, "d1:", d1)

    return grad



