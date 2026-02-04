import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.distributions import Dirichlet

from data_analysis import plot_loss_vs_epoch
from flexiflow import load_model
from model import *


def main(args):
    # seed numpy and torch RNGs for reproducibility
    seed = 22
    torch.manual_seed(seed)

    # inputs: K categories, t
    input_size = args.K + 1
    # outputs: K velocities
    output_size = args.K

    model, loss_fnc, optimizer = load_model(
        input_size, args.hidden_size, output_size, args.num_hidden, args.lr
    )

    p_1 = torch.tensor([0.6, 0.3, 0.1])
    psi_1 = torch.sqrt(p_1)

    loss = np.zeros(args.epochs)
    for i in range(args.epochs):
        # p_0: shape (B, K)
        p_0 = Dirichlet(torch.ones(args.K)).sample((args.batch_size,))
        psi_0 = torch.sqrt(p_0)

        # clamp between [-1, 1] to prevent acos from returning NaN (a
        # precaution; values in positive orthant are not expected to be in
        # [-1, 0) range)
        dot = torch.sum(psi_0 * psi_1, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
        # clamp to prevent NaN for inv_sin_theta
        theta = torch.acos(dot).clamp(1e-5)

        t = torch.rand(args.batch_size, 1)

        inv_sin_theta = 1 / torch.sin(theta[:, None])
        a_t = torch.sin((1 - t) * theta[:, None])
        b_t = torch.sin(t * theta[:, None])
        psi_t = inv_sin_theta * (a_t * psi_0 + b_t * psi_1)

        # d(psi_t) / dt = v_target; velocity is constant along the geodesic
        da_t = -theta[:, None] * torch.cos((1 - t) * theta[:, None])
        db_t = theta[:, None] * torch.cos(t * theta[:, None])
        v_target = inv_sin_theta * (da_t * psi_0 + db_t * psi_1)
        # training label should be tangential component of velocity only;
        # remove radial component
        v_target = v_target - torch.sum(
            v_target * psi_t, dim=1, keepdim=True
        ) * psi_t

        optimizer.zero_grad()
        features = torch.cat((psi_t, t), dim=1)
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

    p_1_val = validate(n_traj, n_steps, args.K, model)
    plot_prob(p_1_val)


def validate(n_traj, n_steps, K, model):
    dt = 1.0 / n_steps

    p_0 = Dirichlet(torch.ones(K)).sample((n_traj,))
    psi_t = torch.sqrt(p_0)

    t = torch.zeros(n_traj, 1)

    for _ in range(n_steps):
        features = torch.cat((psi_t, t), dim=1)
        with torch.no_grad():
            v_theta = model(features)

        psi_t += dt * v_theta
        t += dt

    return psi_t.numpy()**2


def plot_prob(p_pred):
    plt.hist(p_pred, bins=300, density=True)
    plt.xlabel('$p_1$')
    plt.ylabel('Density')
    plt.title('Predicted SFM $p_1$ distribution')
    plt.tight_layout()
    plt.savefig(f'sfm_probs.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_hidden', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=2048)

    args = parser.parse_args()

    main(args)
