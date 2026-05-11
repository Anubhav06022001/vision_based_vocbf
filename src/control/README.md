# 🎛 Control 

This folder contains quadratic programming (QP) based control solvers used for enforcing Control Barrier Function (CBF) safety constraints.

---

## 📂 Files


### 🛡 `cbf_qp_multi.py`

Implements multi-constraint CBF-QP control.

- Combines multiple safety constraints
- Computes safe control input close to reference control

---
### 🛡 `task_alignment.py`
- Task-alignment QP shaping:
  Adds directional consistency to the CBF-QP by penalizing control deviations orthogonal to the goal direction.
  Prevents drift and improves task-consistent behavior under active constraints.

---
## thrown to archive
### ⚙ `qp_solver.py`

Provides a simple quadratic program solver.

- Solves constrained control optimization problems
- Supports single linear safety constraint
- Used for enforcing CBF safety conditions

---
