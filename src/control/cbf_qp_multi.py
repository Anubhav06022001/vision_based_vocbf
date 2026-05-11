import osqp
import numpy as np
from archive.src.control.qp_solver import solve_qp
import scipy.sparse as sp
from src.control.task_alignment import compute_task_aligned_P
#   TODO proeprly

def cbf_qp_multi(u_ref, A_list, b_list):
    P = 2 * np.eye(len(u_ref))
    q = -2 * u_ref

    A = np.vstack(A_list) 
    b = np.array(b_list)  

    u = -np.linalg.solve(P, q)

    for i in range(len(b)):
        Ai = A[i]                    
        violation = (Ai @ u) - b[i]       # since Au >=b but we define Au <= b becuase of sign mismatch in grad_B , fix later once pipeline is done

        if violation < 0:
            denom = Ai @ Ai          
            lambda_i = violation / denom
            u = u - 1*lambda_i * Ai

    return u


# ================ QP using library for modified QP top move in dirrection of goal ==================


# def cbf_qp_osqp(u_ref, q_current, A_list, b_list):

#     n = len(u_ref)

#     # ---- Task-aligned QP matrix ----
#     P_dense = compute_task_aligned_P(u_ref, q_current, lambda_align=5.0)
#     P = sp.csc_matrix(2 * P_dense)   
#     q = -2 * u_ref

#     # ---- Constraints ----
#     A = -np.vstack(A_list)
#     b = -np.array(b_list)

#     A_sparse = sp.csc_matrix(A)
#     l = -np.inf * np.ones(len(b))
#     u = b

#     solver = osqp.OSQP()
#     solver.setup(P=P, q=q, A=A_sparse, l=l, u=u, verbose=False)

#     res = solver.solve()
#     return res.x

# ================ QP using library ==================

def cbf_qp_osqp(u_ref, A_list, b_list):

    n = len(u_ref)

    P = 2*sp.eye(n).tocsc()
    q = -2*u_ref

    # covert constraints as per format expected by qsqp, Ax ≥ b  →  -Ax ≤ -b
    A = -np.vstack(A_list)
    b = -np.array(b_list)         

    # OSQP format:  l ≤ Ax ≤ u
    A_sparse = sp.csc_matrix(A)
    l = -np.inf*np.ones(len(b))
    u = b

    solver = osqp.OSQP()
    solver.setup(P=P, q=q, A=A_sparse, l=l, u=u, verbose=False)

    res = solver.solve()
    return res.x
   