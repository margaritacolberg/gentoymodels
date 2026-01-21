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
    plt.hist(bonds, bins=400)
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
    plt.hist(bond_angles, bins=400)
    plt.xlabel(f'Bond angles (degrees)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('bond_angles.png')
    plt.close()


def get_dihedral_angles(x_gen, quartets):
    """
    Parameters:
        x_gen (np.array):
            ML-generated atomic positions with shape (n_traj, n_atoms, 3)
        quartets (list of tuple of int):
            list of (i, j, k, l) atom index quartets defining dihedrals,
            where the angle is around the bond j-k
    Returns:
        dihedrals (np.array):
            dihedral angles array in degrees, flattened over trajectories and
            quartets, with shape (n_traj * n_quartets,)
    """
    dihedrals = []
    for (i, j, k, l) in quartets:
        b1 = x_gen[:, j, :] - x_gen[:, i, :]
        b2 = x_gen[:, k, :] - x_gen[:, j, :]
        b3 = x_gen[:, l, :] - x_gen[:, k, :]

        # normal vectors to planes
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        n1_norm = np.linalg.norm(n1, axis=1)
        n2_norm = np.linalg.norm(n2, axis=1)

        n1_unit = n1 / n1_norm[:, None]
        n2_unit = n2 / n2_norm[:, None]

        b2_norm = np.linalg.norm(b2, axis=1)
        b2_unit = b2 / b2_norm[:, None]

        # vector perpendicular to n1_unit and b2_unit, and lies on the same
        # plane as i-j-k
        m1 = np.cross(n1_unit, b2_unit)

        x = np.sum(n1 * n2, axis=1)
        # calculate coordinates of n2 in the orthonormal frame formed by
        # n1_unit, b2_unit, and m1
        y = np.sum(m1 * n2, axis=1)

        phi = np.arctan2(y, x)

        dihedrals.append(phi)

    return np.concatenate(dihedrals) * (180 / np.pi)


def plot_dihedral_angles(dihedral_angles):
    plt.hist(dihedral_angles, bins=400)
    plt.xlabel(f'Dihedral angles (degrees)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('dihedral_angles.png')
    plt.close()
