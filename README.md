# GenToyModels: Diffusion-Based Generative Models for Boltzmann Sampling

This repository contains toy model implementations of generative methods for
sampling from equilibrium (Boltzmann) distributions using diffusion models:
Adjoint Sampling (and its extension, the Adjoint Schrödinger Bridge Sampler
(ASBS)), and Denoising Diffusion Probabilistic Models (DDPM).

* **Adjoint Sampling** (2D): An efficient algorithm for training drift
  functions in stochastic differential equations (SDEs), based on the
*Reciprocal Adjoint Matching* approach from Havens et al. (2025).

* **Adjoint Schrödinger Bridge Sampler** (2D): An extension of adjoint sampling
  that allows non-Dirac initial distributions. ASBS alternates forward and
backward half-bridge optimizations in an *Iterative Proportional Fitting*
procedure to approximate the kinetic-optimal Schrödinger bridge.

* **DDPM** (1D): A Denoising Diffusion Probabilistic Model, implemented using a
  forward and reverse process described by Nakkiran et al. (2024).

These implementations serve as simple learning tools for understanding how
diffusion models can sample from complex distributions such as Gaussian
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

- `asbs.py`: 2D toy implementation of ASBS using an *Iterative Proportional
  Fitting* algorithm (Liu et al.).

- `buffer.py`: Replay buffer for storing Euler-Maruyama trajectory endpoints
  and energy gradients used for training the drift function in adjoint
sampling and ASBS.

- `clipper.py`: Clipper for trimming gradients whose L2 norm exceeds a
  user-defined threshold.

- `ddpm.py`: 1D toy implementation of DDPM based on Nakkiran et al.'s
  *Step-by-Step Diffusion* tutorial.

- `model.py`: A simple MLP used by the `adjoint.py`, `asbs.py`, and `ddpm.py`
  scripts.

- `systems.py`: An abstraction for energy functions, energy gradients, and
  their whitened versions for the Gaussian mixture model.

- `test_adjoint.py`: Basic test suite for `adjoint.py`. To execute, run `pytest
  -v`.

## References

* Liu et al., (2025), [Adjoint Schrödinger Bridge
  Sampler](https://arxiv.org/abs/2506.22565)

* Havens et al., (2025), [Adjoint Sampling: Highly Scalable Diffusion Samplers
  via Adjoint Matching](https://arxiv.org/abs/2504.11713)

* Nakkiran et al., (2024), [Step-by-Step Diffusion: An Elementary
  Tutorial](https://arxiv.org/abs/2406.08929)
