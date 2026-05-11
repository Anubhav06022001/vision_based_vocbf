# Perception Module

This directory contains the visual processing backbone for the Vision V-OCBF framework.

## Files
* `encoder_net.py`: Contains the `VisionEncoder` class. It uses a 4-layer CNN to compress `128x128` RGB images from the wrist camera into a dense `64D` latent representation. `LayerNorm` is applied to the output to ensure stable gradients during the offline expectile regression training.