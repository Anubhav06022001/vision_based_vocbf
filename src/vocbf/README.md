# V-OCBF Models

This directory contains the neural network architectures for learning the Value-Guided Control Barrier Function.

## Files
* `vision_barrier_net.py`: Contains the `VisionBarrierNet`. It instantiates the CNN encoder and fuses the 64D visual latent vector with the 14D physical state ($q, \dot{q}$) to predict the safety margin $B$.