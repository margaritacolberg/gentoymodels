# GenerativeToyModels: Diffusion-Based Generative Models for Boltzmann Sampling

This repository contains toy model implementations of two generative methods
for sampling from equilibrium (Boltzmann) distributions using diffusion models:
Adjoint Sampling, and Denoising Diffusion Probabilistic Models (DDPM).

* **Adjoint Sampling** (2D): An efficient algorithm for training drift
  functions in stochastic differential equations (SDEs), based on the
*Reciprocal Adjoint Matching* approach from Havens et al. (2025).

* **DDPM** (1D): A Denoising Diffusion Probabilistic Model, implemented using a
  forward and reverse process described by Nakkiran et al. (2024).

These implementations serve as simple learning tools to better understand how
diffusion models can sample from complex distributions, such as Gaussian
mixtures or quadratic wells.

## Dependencies

All non-standard Python libraries can be installed using

```
pip install -r requirements.txt
```

and were tested with Python 3.13.5.

## File Overview

- `adjoint.py`: 2D toy implementation of adjoint sampling using the *Reciprocal
  Adjoint Matching* algorithm from Havens et al. See the top-level comment for
examples on how to run it with different systems.

- `buffer.py`: Replay buffer for storing Euler-Maruyama trajectory endpoints
  and energy gradients used for training the drift function in adjoint
sampling.

- `clipper.py`: Clips gradients whose L2 norm exceeds a user-defined threshold.

- `ddpm.py`: 1D toy implementation of DDPM based on Nakkiran et al.'s
  *Step-by-Step Diffusion* tutorial.

- `model.py`: A simple MLP used by both the `adjoint.py` and `ddpm.py` scripts.

- `systems.py`: Defines energy functions, energy gradients, and their whitened
  versions for the Gaussian mixture model.

- `test.py`: Basic test suite for `adjoint.py`.

## References

* Havens et al. (2025) [Adjoint Sampling: Highly Scalable Diffusion Samplers
  via Adjoint Matching](https://arxiv.org/abs/2504.11713)

* Nakkiran et al. (2024) [Step-by-Step Diffusion: An Elementary
  Tutorial](https://arxiv.org/abs/2406.08929)
