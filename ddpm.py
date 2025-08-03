# ddpm.py is a 1D toy implementation of DDPM, based on Pseudocode 1 and 2 on p.
# 11 from Nakkiran et al.'s "Step-by-Step Diffusion: An Elementary Tutorial";
# in this program, a neural net is trained to denoise x_{t+dt} to predict x_t
# during the forward process, and the forward process is then reversed for
# sampling

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from model import *


def main(args):
    dt = 0.1
    t_0 = 0.
    t_1 = 1.
    num_t = int(t_1 / dt)
    t_path = np.linspace(t_0, t_1, num_t)

    # var of Boltzmann distribution
    tau = 1
    # var of Gaussian distribution
    sigma_q = 1

    batch_size = 256

    # inputs: x_{t+dt}, t+dt
    input_size = 2
    # output: f(x_{t+dt}, t+dt)
    output_size = 1

    model, loss_fnc, optimizer = load_model(
        input_size, args.hidden_size, output_size, args.lr
    )

    for i in range(args.epochs):
        forward_process(
            model, loss_fnc, optimizer, tau,
            batch_size, t_0, t_1, dt, sigma_q, i
        )

    x0_val = []
    # create a distribution of x0 estimates
    for _ in range(10000):
        x0_val.append(reverse_process(model, sigma_q, t_1, num_t, dt))

    x_th = np.linspace(-5.0, 5.0, 100)
    mu = np.exp(-0.5 * x_th**2 / tau)

    # normalize Boltzmann distribution
    mu /= np.trapezoid(mu, x_th)

    mean_th = np.trapezoid(mu * x_th, x_th)
    var_th = np.trapezoid(mu * (x_th - mean_th)**2, x_th)

    x_calc = []
    for _ in range(5000):
        # noise sample at t = 1
        x = np.random.normal(0, sigma_q)
        for t in t_path[1:]:
            # reverse one diffusion step using Bayes-derived f(x, t)
            x = sanity_check(x, t, sigma_q)
        x_calc.append(x)

    print(
        'mean: {}, std: {} of ML-predicted Boltzmann distribution'.format(
            np.mean(x0_val), np.std(x0_val)
        )
    )
    print(
        'mean: {}, std: {} of theoretical Boltzmann distribution'.format(
            mean_th, np.sqrt(var_th)
        )
    )

    plt.hist(x0_val, bins=500, density=True, label=r'ML $\mu$')
    plt.hist(x_calc, bins=500, density=True, label=r'Calculated')
    plt.plot(x_th, mu, label='Theoretical')
    plt.title('1D Boltzmann distribution')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.legend()
    plt.savefig('boltz_ddpm.png')
    plt.close()


def load_model(input_dim, hidden_dim, output_dim, lr):
    model = MLP(
        input_dim, hidden_dim, output_dim, num_hidden=4, activation='relu'
    )
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, loss_fnc, optimizer


def forward_process(
    model, loss_fnc, optimizer, tau,
    batch_size, t_0, t_1, dt, sigma_q, i
):
    x0 = np.random.normal(0, np.sqrt(tau), size=batch_size)
    # sample t in [0, 1 - dt] to avoid exceeding time range when
    # calculating xt_plus_dt
    t = np.random.uniform(t_0, t_1 - dt, size=batch_size)
    xt = x0 + np.random.normal(0, sigma_q * np.sqrt(t))
    xt_plus_dt = xt + np.random.normal(0, sigma_q * np.sqrt(dt))
    t_plus_dt = t + dt

    assert len(xt_plus_dt) == len(t_plus_dt), (
        'len of xt_plus_dt and t_plus_dt vectors do not match'
    )

    xt = torch.tensor(xt, dtype=torch.float32)
    xt_plus_dt = torch.tensor(xt_plus_dt, dtype=torch.float32)
    t_plus_dt = torch.tensor(t_plus_dt, dtype=torch.float32)

    optimizer.zero_grad()
    features = torch.stack((xt_plus_dt, t_plus_dt), dim=1)
    prediction = model(features).squeeze()
    batch_loss = loss_fnc(input=prediction, target=xt)
    batch_loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f'Epoch {i}, Loss: {batch_loss.item():.5f}')


def reverse_process(model, sigma_q, t_1, num_t, dt):
    x = np.random.normal(0, sigma_q, size=1)
    t = np.array([t_1])

    for i in range(1, num_t+1):
        x_tensor = torch.tensor(x, dtype=torch.float32)
        t_tensor = torch.tensor(t, dtype=torch.float32)

        features = torch.stack((x_tensor, t_tensor), dim=1)

        with torch.no_grad():
            f_theta = model(features).detach().numpy().flatten()

        eta = np.random.normal(0, sigma_q * np.sqrt(dt))

        x = f_theta + eta
        t = np.array([t_1 - (i * dt)])

    return x[0]


def sanity_check(x_t, t, sigma_q):
    mean = x_t / (1 + sigma_q**2 * t)
    var = (sigma_q**2 * t) / (1 + sigma_q**2 * t)

    return np.random.normal(mean, np.sqrt(var))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=300)

    args = parser.parse_args()

    main(args)
