import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import multivariate_normal
import torch

import adjoint as adj
from model import *
from buffer import *
from clipper import *
from systems import *


def main(args):
    # seed numpy, Python, and torch RNGs for reproducibility
    seed = 22
    py_rng, np_rng = adj.seed_all(seed)

    dt = 0.008
    t_0 = 0.
    t_1 = 1.
    num_t = int(t_1 / dt)
    t = np.linspace(t_0, t_1, num_t)

    freqs = 3

    # cache Fourier embeddings
    t_fourier = torch.tensor(adj.fourier(t, freqs), dtype=torch.float32)
    t_1_fourier = torch.tensor(adj.fourier([1.0], freqs), dtype=torch.float32)

    tau = 1.0
    buffer_adj = BatchBuffer(buffer_size=args.n_paths)
    buffer_crt = BatchBuffer(buffer_size=args.n_paths)
    clipper = Clipper(max_norm=args.max_score_norm)

    # GMM parameters
    mu1 = np.array([2.0, 2.0])
    cov1 = np.array([[2.0, 0.0], [0.0, 3.0]])
    mu2 = np.array([-4.0, -4.0])
    cov2 = np.array([[1.0, 0.0], [0.0, 2.0]])

    w1, w2 = 0.5, 0.5

    system, m, W, gmm = adj.create_normalized_gmm_system(
        [mu1, mu2], [cov1, cov2], [w1, w2], tau
    )

    avg_w_grad_E = np.zeros(args.epochs_adj)

    # geometric noise schedule
    sigma_max = 1.0
    sigma_min = 1e-3
    sigma_diff = sigma_max / sigma_min
    sigma = lambda time: (
        sigma_min * sigma_diff**(1 - time) * (2 * np.log(sigma_diff))**0.5
    )

    x_vec_dim = 2

    # inputs: Xt
    input_size = x_vec_dim
    # output: u(Xt, t)
    output_size = x_vec_dim

    model_adj, optimizer_adj = adj.load_model(
        input_size + 2 * freqs, args.hidden_size, output_size, args.num_hidden,
        args.lr
    )

    model_crt, optimizer_crt = adj.load_model(
        input_size, args.hidden_size, output_size, args.num_hidden, args.lr
    )

    loss_crt = np.zeros(args.epochs_crt)
    drift = []
    for i in range(args.stage):
        print('stage =', i)
        for j in range(args.epochs_adj):
            # Gaussian noise term for infinitesimal step of Brownian motion
            dB = np_rng.normal(
                0, np.sqrt(dt), size=(num_t, args.n_paths, x_vec_dim)
            )

            X0, X1 = euler_maruyama_wrap(
                t, model_adj, num_t, dt, sigma, x_vec_dim, args.n_paths, dB,
                np_rng, t_fourier
            )

            # to get var of X1, integrate sigma(t)^2 from 0 to 1 to find
            # cumulative var of full path
            var = np.sum(sigma(t)**2 * dt)

            X1_tensor = torch.tensor(X1, dtype=torch.float32)

            with torch.no_grad():
                h_phi = model_crt(X1_tensor)

            w_grad_E = system.gradenergy(X1)
            a_t = clipper.clip(w_grad_E) + h_phi.detach().numpy()

            buffer_adj.add(X0, X1, a_t)

            accum_loss_adj = 0
            for _ in range(args.n_inner_loop):
                accum_loss_adj += train_drift(
                    buffer_adj, sigma, dt, sigma_max, sigma_diff, x_vec_dim,
                    args.batch_size, np_rng, freqs, model_adj, optimizer_adj
                )

            if i == args.stage - 1:
                drift.append(
                    adj.drift_per_epoch(freqs, X1, model_adj, t_1_fourier)
                )
                avg_w_grad_E[j] = np.mean(system.gradenergy(X1))

            print(
                'adj epoch: {}, train loss: {}'.format(
                    j, accum_loss_adj / args.n_inner_loop
                )
            )

        for k in range(args.epochs_crt):
            dB = np_rng.normal(
                0, np.sqrt(dt), size=(num_t, args.n_paths, x_vec_dim)
            )

            X0, X1 = euler_maruyama_wrap(
                t, model_adj, num_t, dt, sigma, x_vec_dim, args.n_paths, dB,
                np_rng, t_fourier
            )

            buffer_crt.add(X0, X1)

            for _ in range(args.n_inner_loop):
                accum_loss_crt = train_corrector(
                    buffer_crt, sigma, var, x_vec_dim, args.batch_size, np_rng,
                    model_crt, optimizer_crt
                )

            print(
                'crt epoch: {}, train_loss: {}'.format(
                    k, accum_loss_crt / args.n_inner_loop
                )
            )

    # validate drift by comparing distribution of X1 to Boltzmann distribution
    X0_val, X1_val = euler_maruyama_wrap(
        t, model_adj, num_t, dt, sigma, x_vec_dim, 3000, None, np_rng,
        t_fourier
    )

    X0_x_slice = X0_val[:, 0]
    X0_y_slice = X0_val[:, 1]

    epochs = np.arange(0, args.epochs_adj, 1)
    print('whitened mean:', np.mean(X1_val, axis=0))
    print('whitened cov:', np.cov(X1_val.T))

    X1_unwhitened = np.linalg.solve(W, X1_val.T).T + m

    X1_x_slice = X1_unwhitened[:, 0]
    X1_y_slice = X1_unwhitened[:, 1]

    x_th, y_th, boltz = adj.plot_theoretical_contour(
        args.energy_type, tau, gmm
    )
    boltz_slice, th_mean, th_std = adj.theoretical_1D_slice_stats(
        x_th, y_th, boltz
    )
    ml_mean, ml_std = adj.ml_1D_slice_stats(X1_x_slice)
    adj.plot_1D_slices(args.energy_type, x_th, boltz_slice, X1_x_slice)

    # training check: should be approx. 0 if sampler learns correct drift
    adj.plot_avg_w_grad_E_vs_epoch(epochs, avg_w_grad_E)

    adj.plot_ml_contour(args.energy_type, X0_x_slice, X0_y_slice, 'X0')
    adj.plot_ml_contour(args.energy_type, X1_x_slice, X1_y_slice, 'X1')

    # training check: should be approx. 0 if sampler learns correct drift
    adj.plot_drift_vs_epoch(epochs, drift)

    header = [['batch_size', 'n_inner_loop', 'hidden_size', 'lr', 'epochs_adj',
               'epochs_crt', 'n_paths', 'energy_type', 'ml_mean', 'ml_std',
               'th_mean', 'th_std']]
    output = [[args.batch_size, args.n_inner_loop, args.hidden_size, args.lr,
               args.epochs_adj, args.epochs_crt, args.n_paths,
               args.energy_type, ml_mean, ml_std, th_mean, th_std]]

    with open(args.csv, 'w') as output_csv:
        writer = csv.writer(output_csv)
        writer.writerows(header)
        writer.writerows(output)


def euler_maruyama_wrap(
    t, model, num_t, dt, sigma, x_vec_dim, n_paths, dB, np_rng, t_fourier
):
    x = np.zeros((num_t, n_paths, x_vec_dim))

    # Gaussian distribution at t = 0
    mu0 = [2.0, -1.0]
    cov0 = [1.5, 0.5]
    x[0] = np_rng.normal(mu0, cov0, size=(n_paths, x_vec_dim))

    X1 = adj.euler_maruyama(
        t, model, num_t, dt, sigma, x_vec_dim, n_paths, dB, np_rng, x,
        t_fourier
    )

    return x[0], X1


def get_Xt_label_weight(
    sample_t, sample_buffer, sigma, dt, sigma_max, sigma_diff, x_vec_dim,
    batch_size, np_rng
):
    sample_t = np.array(sample_t)
    assert sample_t.ndim == 1, (
        'expected sample_t to be 1D, got shape {}'.format(sample_t.shape)
    )

    # sample Xt from the Brownian bridge distribution (the conditional law of a
    # Brownian path given start X0 and end X1); at t = 1 this collapses to a
    # Dirac delta at X1
    X0 = sample_buffer[:, 0:x_vec_dim]
    X1 = sample_buffer[:, x_vec_dim: 2*x_vec_dim]
    a_t = sample_buffer[:, 2*x_vec_dim: 3*x_vec_dim]

    c_s = sigma_diff**(-2 * sample_t)
    c_1 = sigma_diff**(-2)

    # conditional mean and var of X_t given X_0 and X_1 under the Brownian base
    # process, derived using Bayes' rule and completing the square on Gaussian
    # densities
    alpha = ((c_s - 1) / (c_1 - 1))[:, None]
    mean_Xt = (1.0 - alpha) * X0 + alpha * X1
    var_Xt = sigma_max**2 * (c_s - 1) * (c_s - c_1) / (c_1 - 1)

    if np.any(var_Xt < 0):
        raise ValueError('one or more t values have negative var')

    std_Xt = np.sqrt(var_Xt)[:, None]

    Xt = np_rng.normal(loc=mean_Xt, scale=std_Xt)

    sigmas = sigma(sample_t)[:, None]
    label = -sigmas * a_t
    # weight for numerical stability
    weight = np.full((batch_size, x_vec_dim), 1.0) / (sigmas**2)

    return Xt, label, weight


def train_drift(
    buffer, sigma, dt, sigma_max, sigma_diff, x_vec_dim, batch_size, np_rng,
    freqs, model, optimizer
):
    sample_buffer = buffer.sample(batch_size, np_rng)
    # sample t uniformly in [0,1) independent of Euler-Maruyama grid (since
    # this grid is coarse) to train drift at arbitrary times
    sample_t = np_rng.random(batch_size)

    Xt, label, weight = get_Xt_label_weight(
        sample_t, sample_buffer, sigma, dt, sigma_max, sigma_diff, x_vec_dim,
        batch_size, np_rng
    )

    batch_loss = adj.training_step(
        Xt, sample_t, label, weight, freqs, model, optimizer
    )

    return batch_loss


def train_corrector(
    buffer, sigma, var, x_vec_dim, batch_size, np_rng, model, optimizer
):
    sample_buffer = buffer.sample(batch_size, np_rng)

    X0 = sample_buffer[:, :x_vec_dim]
    X1 = sample_buffer[:, x_vec_dim:]

    label = -(X1 - X0) / var

    X1 = torch.tensor(X1, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)

    optimizer.zero_grad()
    prediction = model(X1)
    diff = prediction - label
    batch_loss = torch.mean(diff**2)
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='csv output file')
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_hidden', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--stage', type=int, default=10)
    parser.add_argument('--epochs_adj', type=int, default=200)
    parser.add_argument('--epochs_crt', type=int, default=50)
    parser.add_argument('--n_inner_loop', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_paths', type=int, default=1000)
    parser.add_argument('--max_score_norm', type=float, default=50.0)
    parser.add_argument('--energy_type', type=str, default='well',
                        choices=['well', 'gmm'])

    args = parser.parse_args()

    main(args)
