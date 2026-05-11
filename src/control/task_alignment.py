import numpy as np


def compute_task_direction(u_ref, q_current, eps=1e-8):
    """
    Compute normalized task direction.

    d_task = u_ref - q_current

    Returns:
        d_hat: unit vector (n,)
    """
    d = u_ref - q_current
    norm = np.linalg.norm(d)

    if norm < eps:
        return np.zeros_like(d)

    return d / norm


def compute_projection_matrices(d_hat):
    """
    Compute projection matrices:
        P_task = d d^T
        P_orth = I - P_task

    Args:
        d_hat: (n,) normalized direction

    Returns:
        P_task: (n, n)
        P_orth: (n, n)
    """
    n = len(d_hat)

    if np.allclose(d_hat, 0):
        return np.zeros((n, n)), np.eye(n)

    P_task = np.outer(d_hat, d_hat)
    P_orth = np.eye(n) - P_task

    return P_task, P_orth


def compute_task_aligned_P(u_ref, q_current, lambda_align=0.5):
    """
    Compute modified QP matrix:

        P = I + lambda * P_orth

    Args:
        u_ref: (n,)
        q_current: (n,)
        lambda_align: float

    Returns:
        P: (n, n)
    """
    n = len(u_ref)

    d_hat = compute_task_direction(u_ref, q_current)
    _, P_orth = compute_projection_matrices(d_hat)

    P = np.eye(n) + lambda_align * P_orth

    return P


def compute_task_deviation(u, u_ref, q_current):
    """
    Debug utility:
    Computes how much deviation is orthogonal vs aligned.

    Returns:
        aligned_mag, orth_mag
    """
    d_hat = compute_task_direction(u_ref, q_current)
    P_task, P_orth = compute_projection_matrices(d_hat)

    v = u - u_ref

    v_task = P_task @ v
    v_orth = P_orth @ v

    return np.linalg.norm(v_task), np.linalg.norm(v_orth)