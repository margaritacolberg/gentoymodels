# GenToyModels: Diffusion-Based Generative Toy Models

This repository contains toy implementations of diffusion-based generative
models that serve as simple learning tools for understanding how these models
can sample from different target distributions such as Gaussian mixtures or
quadratic wells. Included are models for 1D and 2D systems, ranging from
standard diffusion such as the Denoising Diffusion Probabilistic Model (DDPM)
to flow-based methods such as Flow Matching (FM), and methods specialized for
energy-based distributions (Adjoint Sampling (AS), and its extension, the
Adjoint Schrödinger Bridge Sampler (ASBS)).

* **AS** (2D): An efficient algorithm for training drift functions in
  stochastic differential equations (SDEs), based on the *Reciprocal Adjoint
Matching* approach from Havens et al. (2025).

* **ASBS** (2D): An extension of AS that allows non-Dirac initial
  distributions. ASBS alternates forward and backward half-bridge optimizations
in an *Iterative Proportional Fitting* procedure to approximate the
kinetic-optimal Schrödinger bridge.

* **DDPM** (1D): Implemented using a forward and reverse process described by
  Nakkiran et al. (2024).

* **FM** (2D): A continuous-time generative model that learns to match
  distributions via velocity fields. The implementation follows the FM
framework described by Lipman et al. (2023) and Albergo & Vanden-Eijnden
(2023), with formulas adapted from the SimpleFold preliminary by Wang et al.
(2025).

## Dependencies

All non-standard Python libraries can be installed using

```
pip install -r requirements.txt
```

and were tested with Python 3.13.5.

## File Overview

- `adjoint.py`: 2D toy implementation of AS using the *Reciprocal Adjoint
  Matching* algorithm from Havens et al. See the top-level comment for examples
on how to run it with different systems.

- `asbs.py`: 2D toy implementation of ASBS using an *Iterative Proportional
  Fitting* algorithm (Liu et al.).

- `buffer.py`: Replay buffer for storing Euler-Maruyama trajectory endpoints
  and energy gradients used for training the drift function in AS and ASBS.

- `clipper.py`: Clipper for trimming gradients whose L2 norm exceeds a
  user-defined threshold.

- `ddpm.py`: 1D toy implementation of DDPM based on Nakkiran et al.'s
  *Step-by-Step Diffusion* tutorial.

- `fm.py`: 2D toy implementation of FM based on the theory of FM described in
  the SimpleFold paper (Wang et al.).

- `model.py`: A simple MLP used by the `adjoint.py`, `asbs.py`, and `ddpm.py`
  scripts.

- `systems.py`: An abstraction for energy functions, energy gradients, and
  their whitened versions for the Gaussian mixture model.

- `test_adjoint.py`: Basic test suite for `adjoint.py`. To execute, run `pytest
  -v`.

## References

* Havens et al., (2025), [Adjoint Sampling: Highly Scalable Diffusion Samplers
  via Adjoint Matching](https://arxiv.org/abs/2504.11713)

* Liu et al., (2025), [Adjoint Schrödinger Bridge
  Sampler](https://arxiv.org/abs/2506.22565)

* Nakkiran et al., (2024), [Step-by-Step Diffusion: An Elementary
  Tutorial](https://arxiv.org/abs/2406.08929)

* Wang et al., (2025), [SimpleFold: Folding Proteins is Simpler than You
  Think](https://arxiv.org/abs/2509.18480)
