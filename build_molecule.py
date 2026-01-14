import numpy as np
import torch


def get_molecule_positions():
    C = torch.tensor([0.0, 0.0, 0.0])

    # ideal tetrahedral directions
    a = 1.11 / np.sqrt(3)  # C-H bond length approx. 1.11 Angstroms
    b = 1.78 / np.sqrt(3)  # C-Cl bond length approx. 1.78 Angstroms
    H = torch.tensor([
        [ a,  a,  a],
        [ a, -a, -a],
        [-a,  a, -a],
    ], dtype=torch.float32)
    Cl = torch.tensor([
        [-b, -b,  b],
    ], dtype=torch.float32)

    coords = torch.vstack([C, H, Cl])  # (5, 3)

    # subtract geometric center
    coords = coords - coords.mean(dim=0, keepdim=True)

    return coords
