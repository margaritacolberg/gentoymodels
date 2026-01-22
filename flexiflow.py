# Length units: Angstroms
# Angle units: degrees

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.preprocessing import OneHotEncoder

from model import *
import molecule_conformers as mc
import data_analysis as da


def main(args):
    with open(args.json, 'r') as input_json:
        data = json.load(input_json)

    # seed numpy and torch RNGs for reproducibility
    seed = data['seed']
    torch.manual_seed(seed)

    n_atoms = data['n_atoms']
    system_dim = n_atoms * 3  # positions are in 3D

    atom_vocab = data['atom_vocab']  # H, C, N, O, Cl
    charge_vocab = data['charge_vocab']
    heavy_atom_num = set(atom_vocab[1:])

    vocab_size = len(atom_vocab) + len(charge_vocab)

    # inputs: x, y, z, graph nodes, graph edges, t
    input_size = system_dim + (n_atoms * vocab_size) + n_atoms**2 + 1
    # output: v_x, v_y, v_z
    output_size = system_dim

    model, loss_fnc, optimizer = load_model(
        input_size, args.hidden_size, output_size, args.num_hidden, args.lr
    )

    mol = mc.get_mol(args.smiles)
    Z, Q = mc.get_Z_Q(mol)
    bonds = mc.get_bonds(mol)
    angles = mc.get_angles(mol)
    dihedrals = mc.get_heavy_atom_dihedrals(mol, heavy_atom_num)

    target_dihedral_angles = np.array(
        data['target_dihedral_angles'], dtype=float
    )
    target_torsions = [
        (i, j, k, l, angle)
        for (i, j, k, l) in dihedrals
        for angle in target_dihedral_angles
    ]

    conformers = mc.get_mol_conformers(args.n_confs, mol, target_torsions)

    graph = convert_to_graph(
        atom_vocab, charge_vocab, Z, Q, data['cutoff'], conformers[0]
    )
    V_batch, E_batch = make_graph_batch(graph, args.batch_size)

    loss = np.zeros(args.epochs)
    for i in range(args.epochs):
        x_0 = torch.randn(args.batch_size, n_atoms, 3)

        ind = torch.randint(0, args.n_confs, (args.batch_size,))

        # final distribution: noisy conformations
        # noise represents harmonic vibrations, thermal fluctuations, or
        # experimental uncertainty
        noise = data['pos_noise_std'] * torch.randn(
            args.batch_size, n_atoms, 3
        )
        x_1 = conformers[ind] + noise

        t = torch.rand((args.batch_size, 1)).unsqueeze(1)

        # linear interpolant path
        x_t = (1 - t) * x_0 + t * x_1
        # dx_t / dt = v_target; velocity is constant along the linear path
        v_target = x_1 - x_0

        x_t = x_t.flatten(start_dim=1)
        v_target = v_target.flatten(start_dim=1)

        optimizer.zero_grad()
        features = torch.cat((x_t, V_batch, E_batch, t.squeeze(1)), dim=1)
        prediction = model(features)
        batch_loss = loss_fnc(input=prediction, target=v_target)
        batch_loss.backward()
        optimizer.step()

        loss[i] = batch_loss.item()

        if i % 10 == 0:
            print(f'Epoch {i}, Loss: {batch_loss.item():.5f}')

    da.plot_loss_vs_epoch(args.epochs, loss)

    n_traj = data['n_traj']
    n_steps = data['n_steps']

    x_1_val = validate(n_traj, n_steps, graph, n_atoms, model)

    bonds_val = da.get_bond_lengths(x_1_val, bonds)
    angles_val = da.get_bond_angles(x_1_val, angles)
    dihedrals_val = da.get_dihedral_angles(x_1_val, dihedrals)

    da.plot_bond_lengths(bonds_val)
    da.plot_bond_angles(angles_val)
    da.plot_dihedral_angles(dihedrals_val)


def load_model(input_dim, hidden_dim, output_dim, num_hidden, lr):
    model = MLP(
        input_dim, hidden_dim, output_dim, num_hidden, activation='gelu'
    )
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def convert_to_graph(atom_vocab, charge_vocab, Z, Q, cutoff, coords):
    atom_to_index = {z: i for i, z in enumerate(atom_vocab)}
    one_hot_Z = encode_element_types(Z, atom_vocab, atom_to_index)

    charge_to_index = {q: i for i, q in enumerate(charge_vocab)}
    one_hot_Q = encode_element_types(Q, charge_vocab, charge_to_index)

    # nodes
    V = np.concatenate([one_hot_Z, one_hot_Q], axis=1)

    assert coords.ndim == 2 and coords.shape[1] == 3

    # edges
    n_atoms = coords.shape[0]
    E = np.zeros((n_atoms, n_atoms), dtype=int)

    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            d = np.linalg.norm(coords[i] - coords[j])
            # cutoff in Angstroms, slightly longer than longest bond
            if d <= cutoff:
                E[i, j] = 1
                E[j, i] = 1

    return {'V': V, 'E': E}


def encode_element_types(elements, element_vocab, element_to_index):
    n_elements = len(elements)
    n_types = len(element_vocab)

    one_hot = np.zeros((n_elements, n_types), dtype=int)

    for i, e in enumerate(elements):
        if e not in element_to_index:
            raise ValueError(f'Unknown atom or charge type, {e}')

        one_hot[i, element_to_index[e]] = 1

    return one_hot


def make_graph_batch(graph, batch_size):
    V = torch.from_numpy(
        np.broadcast_to(
            graph['V'],
            (batch_size, graph['V'].shape[0], graph['V'].shape[1])
        ).copy()
    ).flatten(start_dim=1)

    E = torch.from_numpy(
        np.broadcast_to(
            graph['E'],
            (batch_size, graph['E'].shape[0], graph['E'].shape[1])
        ).copy()
    ).flatten(start_dim=1)

    return V, E


def validate(n_traj, n_steps, graph, n_atoms, model):
    dt = 1.0 / n_steps

    V_batch, E_batch = make_graph_batch(graph, n_traj)

    x_t = torch.randn(n_traj, n_atoms, 3).flatten(start_dim=1)
    t = torch.zeros(n_traj, 1)

    for _ in range(n_steps):
        features = torch.cat((x_t, V_batch, E_batch, t), dim=1)
        with torch.no_grad():
            v_theta = model(features)

        x_t += dt * v_theta
        t += dt

    x_t = x_t.reshape(n_traj, n_atoms, 3)

    return x_t.numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('json', help='json input file')
    parser.add_argument('n_confs', type=int)
    parser.add_argument('smiles', type=str)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_hidden', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=7000)
    parser.add_argument('--batch_size', type=int, default=2048)

    args = parser.parse_args()

    main(args)
