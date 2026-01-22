# GenToyModels: Diffusion-Based Generative Toy Models

This repository contains toy implementations of diffusion-based generative
models for sampling from target distributions. It includes a 1D implementation
of the [Denoising Diffusion Probabilistic Model
(DDPM)](https://arxiv.org/abs/2406.08929), 2D implementations of [Adjoint
Sampling (AS)](https://arxiv.org/abs/2504.11713) and the [Adjoint Schrödinger
Bridge Sampler (ASBS)](https://arxiv.org/abs/2506.22565), and a simplified
[FlexiFlow](https://arxiv.org/abs/2511.17249) model for generating butane
conformers.

## Dependencies

All non-standard Python libraries can be installed using

```
pip install -r requirements.txt
```

and were tested with Python 3.13.5.

## File Overview

- `adjoint.py`: 2D implementation of AS for Gaussian mixtures and quadratic
  wells. See the top-level comment for examples on how to run these systems.

- `asbs.py`: 2D implementation of ASBS for Gaussian mixtures.

- `buffer.py`: Replay buffer for storing Euler-Maruyama trajectory endpoints
  and energy gradients used for training the drift function in AS and ASBS.

- `c4h10.json`: Input file for `flexiflow.py` containing structural information
  about butane.

- `clipper.py`: Clipper for trimming gradients whose L2 norm exceeds a
  user-defined threshold.

- `data_analysis.py`: Calculate and plot bond lengths, bond angles, and
  dihedral angles for generated molecular conformers.

- `ddpm.py`: 1D implementation of DDPM for Gaussians.

- `flexiflow.py`: 3D implementation of FlexiFlow without joint sampling of
  conformers, using a single MLP instead of SemlaFlow for simplicity.

- `model.py`: A simple MLP used by the `adjoint.py`, `asbs.py`, `ddpm.py`, and
  `flexiflow.py` scripts.

- `molecule_conformers.py`: Generates target distributions of molecular
  conformers for training. Also extracts atomic numbers and charges for
building molecular graphs, and computes theoretical bond lengths, bond angles,
and dihedrals for validation.

- `systems.py`: An abstraction for energy functions, energy gradients, and
  their whitened versions for Gaussian mixtures.

- `test_adjoint.py`: Basic test suite for `adjoint.py`. To execute, run `pytest
  -v`.
