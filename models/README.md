# Vision V-OCBF Model

This directory contains the trained PyTorch weights for the Vision-based Value-Guided Control Barrier Function (`vision_vocbf.pt`).

## Architecture Details
* **Vision Encoder:** A lightweight 4-layer CNN that compresses `128x128x3` RGB images from the wrist camera into a 64D latent representation.
* **Barrier MLP:** A multi-layer perceptron that fuses the 64D visual latent vector with the 14D physical state ($q$, $\dot{q}$) to output the safety margin $B(x)$.

## Usage
Load these weights during the real-time inference phase to evaluate the visual safety gradients ($\nabla B$) for the QP solver.