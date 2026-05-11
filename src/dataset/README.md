# Vision Dataset Module

This module handles the loading and preprocessing of MuJoCo transition data for Vision-OCBF training.

## Features
* **RAM Caching:** Bypasses disk I/O bottlenecks by loading the entire HDF5 file directly into memory during initialization.
* **Auto-Normalization:** Dynamically computes mean and standard deviation for physical states ($q$, $\dot{q}$) using a data subset to speed up training convergence.