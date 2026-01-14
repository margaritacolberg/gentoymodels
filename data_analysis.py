import numpy as np
import matplotlib.pyplot as plt


def plot_loss_vs_epoch(epochs, loss):
    plt.plot(np.arange(epochs), loss)
    plt.title('Loss vs. epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.close()


def get_rmsd(x_ref, x_gen):
    # x_ref: (n_atoms, 3), x_gen: (n_traj, n_atoms, 3)
    diff = x_ref - x_gen
    rmsd = np.sqrt(np.sum(diff**2, axis=(1, 2)) / x_ref.size)

    return rmsd


def plot_rmsd(rmsd):
    plt.hist(rmsd, bins=200)
    plt.xlabel('RMSD')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('rmsd.png')
    plt.close()


def get_bond_lengths(x_gen, pairs):
    """
    Parameters:
        x_gen (np.array):
            ML-generated atomic positions with shape (n_traj, n_atoms, 3)
        pairs (list of pairs of int):
            list of (i, j) atom index pairs defining bonds
    Returns:
        bonds (np.array):
            bond lengths array in Angstroms, flattened over trajectories and
            pairs, with shape (n_traj * n_pairs,)
    """
    bonds = []
    for (i, j) in pairs:
        d = np.linalg.norm(x_gen[:, j, :] - x_gen[:, i, :], axis=1)
        bonds.append(d)

    return np.concatenate(bonds)


def plot_bond_lengths(bonds):
    plt.hist(bonds, bins=200)
    plt.xlabel('Bond length ($\\AA$)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('bond_lengths.png')
    plt.close()


def get_bond_angles(x_gen, triplets):
    """
    Parameters:
        x_gen (np.array):
            ML-generated atomic positions with shape (n_traj, n_atoms, 3)
        triplets (list of tuple of int):
            list of (i, j, k) atom index triplets defining angles, where j is
            the central atom and the angle is formed by i-j-k
    Returns:
        angles (np.array):
            bond angles array in degrees, flattened over trajectories and
            triplets, with shape (n_traj * n_triplets,)
    """
    angles = []
    for (i, j, k) in triplets:
        vec1 = x_gen[:, i, :] - x_gen[:, j, :]  # (n_traj, 3)
        vec2 = x_gen[:, k, :] - x_gen[:, j, :]

        dot = np.sum(vec1 * vec2, axis=1)
        vec1_norm = np.linalg.norm(vec1, axis=1)
        vec2_norm = np.linalg.norm(vec2, axis=1)

        cos_theta = dot / (vec1_norm * vec2_norm)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)
        angles.append(theta)  # radians

    return np.concatenate(angles) * (180 / np.pi)


def plot_bond_angles(bond_angles):
    plt.hist(bond_angles, bins=200)
    plt.xlabel(f'Bond angles (degrees)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('bond_angles.png')
    plt.close()
