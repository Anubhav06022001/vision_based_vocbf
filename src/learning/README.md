# Learning Module

This directory handles the offline training pipeline for the safety certificates.

## Files
* `train_vision_vocbf.py`: The main training loop. It streams the `(Image, State, Next_Image, Next_State)` tuples from the HDF5 dataset to train the `VisionBarrierNet`. Uses Expectile Regression (tau=0.9) to learn conservative safety margins directly from visual transitions without requiring a learned dynamics model during training.
* `expectile.py`: Contains the asymmetric expectile loss function.