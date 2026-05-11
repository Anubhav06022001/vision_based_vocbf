# 🚧 Value-Guided Control Barrier Functions (V-OCBF)

An implementation of **Value-Guided Control Barrier Functions (V-OCBF)** for **safe robot control**, inspired by recent advances in learning-based safety filters.

This repository extends classical control barrier formulations into **vision-based, high-dimensional settings**, enabling **end-to-end safety from pixels**.

---

## ✨ Overview

**Vision-Based V-OCBF** removes the need for explicit geometric modeling and external perception pipelines by learning **safety certificates directly from visual observations**.

> Instead of computing safety from structured states, we infer it from *what the robot sees*.

---

## 📁 Repository Structure

```text
vision_based_vocbf/
├── README.md                   # Project overview & documentation
├── assets/                     # Meshes, MuJoCo scene.xml
├── data/
│   └── vision_vocbf/
│       ├── transition_dataset.pkl   # State transitions + distances
│       └── images/                  # HDF5 or PNG frames
│
├── src/
│   ├── perception/
│   │   ├── encoder_net.py           # CNN / ViT encoder
│   │   └── vision_utils.py          # Preprocessing & augmentation
│   │
│   ├── learning/
│   │   ├── vision_barrier_learner.py # Image-based CBF learner
│   │   └── expectile.py              # Expectile loss (reused)
│   │
│   ├── dynamics/
│   │   ├── vision_dynamics_net.py    # Latent dynamics model
│   │   └── train_vision_dynamics.py
│   │
│   ├── control/
│   │   └── vision_cbf_qp.py          # QP safety filter
│   │
│   └── safety/
│       └── distance_utils.py         # MuJoCo distance utilities
│
├── experiments/
│   ├── collect_vision_data.py        # Dataset collection
│   └── test_vision_vocbf.py          # Evaluation pipeline
│
├── models/
│   ├── vision_encoder.pt
│   └── vision_vocbf.pt
│
└── .gitignore
```

---


## Project Pipeline Overview

| Module              | Step Folder                         | What it Does                                                                 | Key Files                          |
|--------------------|-------------------------------------|------------------------------------------------------------------------------|------------------------------------|
| Data Collection    | `experiments/`                      | Runs the MuJoCo simulation to generate offline datasets                      | `collect_vision_data.py`           |
| Offline Learning   | `src/perception, /src/learning/`      | Trains the CNN encoder and the Neural Barrier Function offline               | `encoder_net.py`, `vision_barrier_learner.py` |
| Inference / Control| `src/dynamics`, `/src/control/`         | Runs the live Franka simulation and computes QP-based safety filters         | `vision_dynamics_net.py`, `vision_cbf_qp.py`  |


## 🚀 Pipeline Usage

### 1. Vision Data Collection

We collect transitions using a custom Cartesian IK sampler to ensure the wrist-camera maintains focus on the workspace. Ground-truth collisions (`data.ncon > 0`) are deliberately allowed to anchor the safety network.

```bash
export MUJOCO_GL=egl
python experiments/collect_vision_data.py
```

2. Extract Normalization StatsComputes and saves state normalization arrays for stable training.Bashpython src/dataset/saved_stats.py
3. Train the Latent Dynamics V-OCBFTrains the 3-part network using a combined loss function: Mean Squared Error (MSE) for the latent dynamics, and Expectile Loss for the Value-Guided safety barrier.Bashpython src/learning/train_vision_vocbf.py
🧩 Key Architecture InsightYour QP solver mathematically requires the gradient mapping how an action changes safety. Because PyTorch cannot backpropagate through MuJoCo's renderer, we split the network:Encoder: $I_t \to z_t$Latent Dynamics: $z_{t+1} = f(z_t, x_t, u_t)$Barrier: $B_{t+1} = g(z_{t+1})$When the QP solver requests a safety gradient, PyTorch tracks the computational graph backwards from $B_{t+1}$, through the predicted visual state $z_{t+1}$, directly to the commanded action $u_t$. Safety is no longer computed — it is perceived and predicted.

---

## ⚙️ Requirements

* MuJoCo
* Python ≥ 3.9
* NumPy
* PyTorch

---

## 🧩 Key Insight

> **Safety is no longer computed — it is *perceived*.**

This framework bridges **control theory + deep learning + vision**, pushing toward **generalizable robot safety in unstructured environments**.









