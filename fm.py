# Length units: degrees
# Angle units: Angstroms

import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import *


def main(args):
    system_dim = 15  # CH4 has 5 atoms, positions are in 3D

    # seed numpy and torch RNGs for reproducibility
    seed = 22
    torch.manual_seed(seed)

    # inputs: x, y, z, t
    input_size = system_dim + 1
    # output: v_x, v_y, v_z
    output_size = system_dim

    model, loss_fnc, optimizer = load_model(
        input_size, args.hidden_size, output_size, args.num_hidden, args.lr
    )

    ch4_pos = get_ch4_positions()  # ch4_pos.shape == torch.Size([15])

    loss = np.zeros(args.epochs)
    for i in range(args.epochs):
        x_0 = torch.randn(args.batch_size, system_dim)

        # final distribution: noisy CH4 conformations
        # noise.shape == torch.Size([B, 15])
        noise = 0.001 * torch.randn(args.batch_size, system_dim)
        # broadcast to torch.Size([B, 15])
        x_1 = ch4_pos + noise

        t = torch.rand((args.batch_size, 1))

        # linear interpolant path
        x_t = (1 - t) * x_0 + t * x_1
        # dx_t / dt = v_target; velocity is constant along the linear path
        v_target = x_1 - x_0

        optimizer.zero_grad()
        features = torch.cat((x_t, t), dim=1)
        prediction = model(features)
        batch_loss = loss_fnc(input=prediction, target=v_target)
        batch_loss.backward()
        optimizer.step()

        loss[i] = batch_loss.item()

        if i % 10 == 0:
            print(f'Epoch {i}, Loss: {batch_loss.item():.5f}')

    plot_loss_vs_epoch(args.epochs, loss)

    n_traj = 3000
    n_steps = 300

    x_1_val = validate(n_traj, n_steps, system_dim, model)
    x_1_val = np.array([reorder_ch4(xi).flatten() for xi in x_1_val])

    rmsd = get_rmsd(ch4_pos.numpy(), x_1_val)
    bond_lengths = get_bond_lengths(x_1_val)
    print(f'Average C-H bond length is {np.mean(bond_lengths)} Angstoms')
    print(f'SD of C-H bond length is {np.std(bond_lengths)} Angstoms')
    bond_angles = get_bond_angles(x_1_val)
    print(f'Average H-C-H bond angle is {np.mean(bond_angles)} degrees')
    print(f'SD of H-C-H bond angle is {np.std(bond_angles)} degrees')

    plot_rmsd(rmsd)
    plot_bond_lengths(bond_lengths)
    plot_bond_angles(bond_angles)


def load_model(input_dim, hidden_dim, output_dim, num_hidden, lr):
    model = MLP(
        input_dim, hidden_dim, output_dim, num_hidden, activation='gelu'
    )
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def get_ch4_positions():
    C = torch.tensor([0.0, 0.0, 0.0])

    # ideal tetrahedral directions
    a = 1.09 / np.sqrt(3)  # C-H bond length approx. 1.09 Angstroms
    H = torch.tensor([
        [ a,  a,  a],
        [ a, -a, -a],
        [-a,  a, -a],
        [-a, -a,  a],
    ], dtype=torch.float32)

    coords = torch.vstack([C, H])  # (5, 3)

    # subtract geometric center
    coords = coords - coords.mean(dim=0, keepdim=True)

    return coords.view(-1)


def plot_loss_vs_epoch(epochs, loss):
    plt.plot(np.arange(epochs), loss)
    plt.title('Loss vs. epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.close()


def validate(n_traj, n_steps, system_dim, model):
    dt = 1.0 / n_steps

    x_t = torch.randn(n_traj, system_dim)
    t = torch.zeros(n_traj, 1)

    for _ in range(n_steps):
        features = torch.cat((x_t, t), dim=1)
        with torch.no_grad():
            v_theta = model(features)

        x_t += dt * v_theta
        t += dt

    return x_t.numpy()


def get_rmsd(x_ref, x_gen):
    assert x_ref.size == x_gen[0].size, (
        'size mismatch between reference and generated coordinates'
    )

    diff = x_ref - x_gen
    rmsd = np.sqrt(np.sum(diff**2, axis=1) / x_ref.size)

    return rmsd


def plot_rmsd(rmsd):
    plt.hist(rmsd, bins=200)
    plt.xlabel('CH4 RMSD')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('rmsd.png')
    plt.close()


def reorder_ch4(x):
    x = x.reshape(5, 3)
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)  # (5, 5)
    # each row in d matrix corresponds to distance between atom and all other
    # atoms; sort distances, ignoring self-distance
    sort_d = np.sort(d, axis=1)[:, 1:5]  # (5, 4)
    # min std will always be for C, since C's nearest neighbours are 4 H, but
    # for each H, the nearest neighbors are 1 C, 3 H which gives larger std
    carbon_ind = np.argmin(sort_d.std(axis=1))
    order = [carbon_ind] + [i for i in range(5) if i != carbon_ind]

    return x[order]


def get_bond_lengths(x_gen):
    # x_gen: (n_traj, 15) flattened, 5 atoms * 3 coords
    x_gen = x_gen.reshape(-1, 5, 3)  # (n_traj, 5, 3)
    C_pos = x_gen[:, 0, :]  # (n_traj, 3)
    H_pos = x_gen[:, 1:, :]  # (n_traj, 4, 3)

    # calculate bond length for each H and trajectory
    d = np.linalg.norm(H_pos - C_pos[:, np.newaxis, :], axis=2)  # (n_traj, 4)

    return d.flatten()


def plot_bond_lengths(bond_lengths):
    plt.hist(bond_lengths, bins=200)
    plt.xlabel('CH4 bond lengths ($\\AA$)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('bond_lengths.png')
    plt.close()


def get_bond_angles(x_gen):
    x_gen = x_gen.reshape(-1, 5, 3)
    C_pos = x_gen[:, 0, :]
    H_pos = x_gen[:, 1:, :]

    num_H = H_pos[0].shape[0]

    angles = []
    for i in range(num_H):
        for j in range(i+1, num_H):
            vec1 = H_pos[:, i, :] - C_pos  # (n_traj, 3)
            vec2 = H_pos[:, j, :] - C_pos

            dot = np.sum(vec1 * vec2, axis=1)  # (n_traj, )
            vec1_norm = np.linalg.norm(vec1, axis=1)  # (n_traj, )
            vec2_norm = np.linalg.norm(vec2, axis=1)

            cos_theta = dot / (vec1_norm * vec2_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)

            theta = np.arccos(cos_theta)
            angles.append(theta)  # radians

    return np.concatenate(angles) * (180 / np.pi)


def plot_bond_angles(bond_angles):
    plt.hist(bond_angles, bins=200)
    plt.xlabel('CH4 bond angles (degrees)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('bond_angles.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_hidden', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=2048)

    args = parser.parse_args()

    main(args)
