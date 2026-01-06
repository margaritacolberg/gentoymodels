# Gaussian to a Gaussian (1D) example, taken from:
# https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from model import *


def main(args):
    mu = 2.0

    # inputs: x_t, t
    input_size = 2
    # output: v_theta(x_t, t)
    output_size = 1

    model, loss_fnc, optimizer = load_model(
        input_size, args.hidden_size, output_size, args.num_hidden, args.lr
    )

    loss = np.zeros(args.epochs)
    for i in range(args.epochs):
        x_0 = torch.randn(args.batch_size)
        t = torch.rand(args.batch_size)
        x_t = x_0 + mu * t
        v_target = torch.full((args.batch_size, 1), mu)

        optimizer.zero_grad()
        features = torch.stack((x_t, t), dim=1)
        prediction = model(features)
        batch_loss = loss_fnc(input=prediction, target=v_target)
        batch_loss.backward()
        optimizer.step()

        loss[i] = batch_loss

        if i % 10 == 0:
            print(f'Epoch {i}, Loss: {batch_loss.item():.5f}')

    plot_loss_vs_epoch(args.epochs, loss)

    n_traj = 3000
    n_steps = 300
    dt = 1.0 / n_steps

    x_1 = []
    for _ in range(n_traj):
        x_t = torch.randn(1, 1)
        t = torch.zeros(1, 1)

        for i in range(n_steps):
            features = torch.cat((x_t, t), dim=1)
            with torch.no_grad():
                v_theta = model(features)

            x_t += dt * v_theta
            t += dt
            
            if i == n_steps-1:
                x_1.append(x_t)

    x_1 = np.array(x_1).squeeze()
    x_th = np.random.normal(loc=mu, scale=1.0, size=n_traj)

    plot_dist(x_1, x_th)


def load_model(input_dim, hidden_dim, output_dim, num_hidden, lr):
    model = MLP(
        input_dim, hidden_dim, output_dim, num_hidden, activation='relu'
    )
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def plot_loss_vs_epoch(epochs, loss):
    plt.plot(np.arange(epochs), loss)
    plt.title('Loss vs. epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('loss.png')
    plt.close()


def plot_dist(x_pred, x_th):
    plt.hist(x_pred, bins=300, density=True, label=r'ML-predicted')
    plt.hist(x_th, bins=300, density=True, \
            label=r'Theoretical $\mathcal{N}(\mu, 1)$')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'fm_dists.png')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_hidden', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()

    main(args)
