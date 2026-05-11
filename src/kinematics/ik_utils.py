import mujoco as mj
import numpy as np 

def get_ik_goal(model, data, target_pos):
    """
    Uses Differential Inverse Kinematics to find a joint configuration (q_goal)
    that reaches target_pos while minimizing unnecessary wrist rotation.
    """
    site_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "ee_site")
    
    # Save state to not disrupt the active simulation
    q_init = data.qpos[:7].copy()
    
    jacp = np.zeros((3, model.nv))
    for _ in range(50): # Max 50 IK iterations
        err = target_pos - data.site_xpos[site_id]
        if np.linalg.norm(err) < 0.01:
            break
            
        mj.mj_jacSite(model, data, jacp, None, site_id)
        J = jacp[:, :7]
        
        # Damped least squares for stability
        dq = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(3)) @ err
        data.qpos[:7] += dq * 0.5
        mj.mj_kinematics(model, data)
        
    q_goal = data.qpos[:7].copy()
    
    # Restore simulation state
    data.qpos[:7] = q_init
    mj.mj_kinematics(model, data)
    
    return q_goal