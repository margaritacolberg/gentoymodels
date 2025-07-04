# adjoint.py is a 1D toy implementation of adjoint sampling, based on
# pseudocode on p. 6 from Havens et al.'s preprint "Adjoint Sampling: Highly
# Scalable Diffusion Samplers via Adjoint Matching"; in this program, a drift
# function in an SDE is trained to sample from a 1D Boltzmann distribution
# using the adjoint matching algorithm

import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

from model import *


def main(args):
    dt = 0.001
    t_0 = 0.
    t_1 = 1.
    num_t = int(t_1 / dt)
    t = np.linspace(t_0, t_1, num_t) 

    tau = 1
    buffer = []

    batch_size = 64 
    n_paths = 200
    n_inner_loop = 20

    # use var-exploding schedule; add dt to avoid zero noise at t = 0
    sigma = lambda time: time + dt

    u = np.zeros(num_t)

    # inputs: Xt, t
    input_size = 2
    # output: u(Xt, t)
    output_size = 1

    model, optimizer = load_model(input_size, args.hidden_size, output_size, \
            args.lr)

    for j in range(args.epochs):
        X1_list = []
        grad_g_list = []
        for _ in range(n_paths):
            X1 = euler_maruyama(t, model, num_t, dt, sigma)

            # to get var of X1, integrate sigma(t)^2 from 0 to 1 to find cumulative
            # var of full path
            var = np.sum(sigma(t)**2 * dt)
            # E(x) = 1/2 * x^2; base distribution is multivariate normal
            # probability density function with mean 0, since there is no drift in
            # base process 
            grad_g = float((-X1 / var) + (X1 / tau))

            X1_list.append(X1)
            grad_g_list.append(grad_g)

        for pair in zip(X1_list, grad_g_list):
            buffer.append(pair)

        accum_loss = 0
        for k in range(n_inner_loop):
            sample_buffer = random.choices(buffer, k=batch_size)
            sample_t = random.choices(t, k=batch_size)

            Xt, label, weight = get_Xt_label_weight(sample_t, sample_buffer, \
                    sigma, dt)

            assert len(Xt) == len(sample_t), \
                    'len of Xt and t vectors do not match'

            Xt = torch.tensor(Xt, dtype=torch.float32)
            sample_t = torch.tensor(sample_t, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            weight = torch.tensor(weight, dtype=torch.float32)

            optimizer.zero_grad()
            features = torch.stack((Xt, sample_t), dim=1)
            prediction = model(features)
            diff = prediction - label
            batch_loss = torch.mean(weight * (diff**2))
            accum_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        print('Epoch: {}, train loss: {}'.format(j, accum_loss))

    X1_val = []
    # validate drift by comparing distribution of X1 to Boltzmann distribution
    for i in range(1000):
        X1_val.append(euler_maruyama(t, model, num_t, dt, sigma))

    x_th = np.linspace(-5.0, 5.0, 100)
    mu = np.exp(-0.5 * x_th**2 / tau)

    # normalize Boltzmann distribution
    mu /= np.trapezoid(mu, x_th)

    mean_th = np.trapezoid(mu * x_th, x_th)
    var_th = np.trapezoid(mu * (x_th - mean_th)**2, x_th)

    print('Mean: {}, std: {} of ML-predicted Boltzmann distribution'.format( \
            np.mean(X1_val), np.std(X1_val)))
    print('Mean: {}, std: {} of theoretical Boltzmann distribution'.format( \
            mean_th, np.sqrt(var_th)))

    plt.hist(X1_val, bins=500, density=True, label=r'ML $\mu$')
    plt.plot(x_th, mu, label='Theoretical')
    plt.title('Validate Boltzmann distribution')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x)$')
    plt.legend()
    plt.savefig('boltz_adjoint.png')


def load_model(input_dim, hidden_dim, output_dim, lr):
    model = MLP(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, optimizer


def euler_maruyama(t, model, num_t, dt, sigma):
    # Gaussian noise term for infinitesimal step of Brownian motion
    dB = np.random.normal(0, np.sqrt(dt), size=num_t) 

    # Dirac distribution at t = 0
    x = np.zeros(num_t)

    # Euler-Maruyama with no gradient
    # https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
    for i in range(num_t - 1):
        x_i_tensor = torch.tensor([x[i]], dtype=torch.float32)
        t_i_tensor = torch.tensor([t[i]], dtype=torch.float32)

        features = torch.stack((x_i_tensor, t_i_tensor), dim=1)

        with torch.no_grad():
            u_theta = model(features).item()

        x[i+1] = x[i] + (u_theta * dt) + (sigma(t[i]) * dB[i])

    return float(x[-1])


def get_Xt_label_weight(sample_t, sample_buffer, sigma, dt):
    # choose Xt from a distribution of paths passing through Xt that
    # all end at X1; the distribution at t = 1 is a Dirac delta with
    # mean X1 and var 0
    var_t = lambda t: (t**3 / 3) + t**2 * dt + t * dt**2
    var_tot = (1 / 3) + dt + dt**2 
    mean_Xt = lambda t, X1: (var_t(t) * X1) / var_tot
    var_Xt = lambda t: (var_t(t) * (var_tot - var_t(t))) / var_tot

    Xt = []
    label = []
    weight = []

    for i in range(len(sample_t)):
        X1 = sample_buffer[i][0]
        grad_g = sample_buffer[i][1]

        Xt.append(np.random.normal(mean_Xt(sample_t[i], X1), \
                np.sqrt(var_Xt(sample_t[i]))))

        label.append(-sigma(sample_t[i]) * grad_g)

        # weight to improve numerical stability
        weight.append(0.5 / sigma(sample_t[i])**2)
      
    return Xt, label, weight


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)

    args = parser.parse_args()

    main(args)
