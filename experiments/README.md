# 🧪 Experiments

This directory contains executable scripts for running **MuJoCo simulations**, **collecting data**, and **testing policies**.

---

## 📁 Files

### 🔹 `collect_vision_data.py`

The primary data generation script.

- Runs a randomized policy on the Franka manipulator  
- Renders RGB frames from the wrist camera (`128x128`)  
- Logs synchronized:
  - states (`q`, `qd`)
  - actions (`u`)
  - safety labels (`min_dist`)

**Execution**  
Run this first to generate the offline dataset required for training the Vision Encoder and V-OCBF networks.

**Output**  

data/vision_vocbf/transition_dataset.h5


---

## 🚀 Inference Testing

**Script**  

experiments/test_vision_vocbf.py


**Function**  
Runs the Franka Panda in a closed-loop simulation. It uses the learned `VisionBarrierNet` to filter nominal control goals.

**Visuals**  
Features an on-screen **Inset View** showing the live feed from the robot's wrist camera, highlighting how visual proximity to obstacles triggers the safety filter.