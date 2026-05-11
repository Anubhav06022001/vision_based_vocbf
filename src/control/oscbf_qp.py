import numpy as np
import scipy.sparse as sp
import osqp
import mujoco as mj

def oscbf_qp(u_ref, q, model, data, A_list, b_list, site_id, Wo=1.0, Wj=0.1):
    n = len(q)
    
    # 1. Get Jacobian (3x7 for position, 6x7 for pose)
    Jp = np.zeros((3, model.nv))
    Jr = np.zeros((3, model.nv))
    mj.mj_jacSite(model, data, Jp, Jr, site_id)
    J = Jp[:, :n] # Position task
    
    # 2. Compute Null-space Projection Matrix
    # N = I - J_pinv @ J
    J_pinv = np.linalg.pinv(J)
    N = np.eye(n) - J_pinv @ J
    
    # 3. Construct OSCBF Objective (Eq. 25 & 26 in paper) [cite: 196]
    # P = J^T * Wo * J + N^T * Wj * N
    P = (J.T @ (Wo * np.eye(3)) @ J) + (N.T @ (Wj * np.eye(n)) @ N)
    P = 2 * P # OSQP format
    
    # q_vec = -2 * P @ u_nom [cite: 196]
    q_vec = -P @ u_ref
    
    P_sparse = sp.csc_matrix(P)

    # 4. Constraints (Stay the same)
    if len(A_list) > 0:
        A_mat = np.vstack(A_list)
        b_vec = np.array(b_list)
        A_sparse = sp.csc_matrix(A_mat)
        l = b_vec # Lower bound: grad @ u >= b
        u = np.inf * np.ones(len(b_vec))
    else:
        return u_ref # Fallback

    solver = osqp.OSQP()
    solver.setup(P=P_sparse, q=q_vec, A=A_sparse, l=l, u=u, verbose=False)
    res = solver.solve()
    return res.x if res.info.status == 'solved' else u_ref



















# import numpy as np
# import scipy.sparse as sp
# import osqp
# import mujoco as mj

# def get_ee_jacobian(model, data, site_id):
#     Jp = np.zeros((3, model.nv))
#     Jr = np.zeros((3, model.nv))

#     mj.mj_jacSite(model, data, Jp, Jr, site_id)

#     return Jp[:, :7]   # only position Jacobian (3x7)


# def compute_task_direction(u_ref, q):
#     d = u_ref - q
#     norm = np.linalg.norm(d)
#     if norm < 1e-6:
#         return np.zeros_like(d)
#     return d / norm


# def compute_projection_matrices(d_hat):
#     n = len(d_hat)

#     # along task direction
#     P_task = np.outer(d_hat, d_hat)

#     # orthogonal complement
#     P_orth = np.eye(n) - P_task

#     return P_task, P_orth


# def oscbf_qp(u_ref, q, A_list, b_list, lambda_align=1.0):
#     """
#     OSCBF QP:
#     min ||P_task (u - u_ref)||^2 + λ ||P_orth (u - u_ref)||^2
#     s.t. A u >= b
#     """

#     n = len(u_ref)

#     # ---------- Task direction ----------
#     d_hat = compute_task_direction(u_ref, q)
#     P_task, P_orth = compute_projection_matrices(d_hat)

#     # ---------- Cost ----------
#     P = 2 * (P_task + lambda_align * P_orth)
#     q_vec = -2 * (P_task + lambda_align * P_orth) @ u_ref

#     P_sparse = sp.csc_matrix(P)

#     # ---------- Constraints ----------
#     if len(A_list) > 0:
#         A = -np.vstack(A_list)
#         b = -np.array(b_list)

#         A_sparse = sp.csc_matrix(A)
#         l = -np.inf * np.ones(len(b))
#         u = b
#     else:
#         A_sparse = sp.csc_matrix((0, n))
#         l = np.array([])
#         u = np.array([])

#     # ---------- Solve ----------
#     solver = osqp.OSQP()
#     solver.setup(
#         P=P_sparse,
#         q=q_vec,
#         A=A_sparse,
#         l=l,
#         u=u,
#         verbose=False
#     )

#     res = solver.solve()

#     return res.x