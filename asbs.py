# if energy is a simple harmonic well, run
# python adjoint.py well.csv
#
# if energy is -log of the Gaussian mixture model (GMM), run
# python adjoint.py gmm.csv --energy_type 'gmm'

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import multivariate_normal
import torch

from model import *
from buffer import *
from clipper import *
from systems import *


def main(args):
    # seed numpy, Python, and torch RNGs for reproducibility
    seed = 22
    py_rng, np_rng = seed_all(seed)

    dt = 0.008
    t_0 = 0.
    t_1 = 1.
    num_t = int(t_1 / dt)
    t = np.linspace(t_0, t_1, num_t)

    tau = 1.0
    buffer_adj = BatchBuffer(buffer_size=args.n_paths)
    buffer_crt = BatchBuffer(buffer_size=args.n_paths)
    clipper = Clipper(max_norm=args.max_score_norm)

    if args.energy_type == 'gmm':
        # GMM parameters
        mu1 = np.array([2.0, 2.0])
        cov1 = np.array([[2.0, 0.0], [0.0, 3.0]])
        mu2 = np.array([-4.0, -4.0])
        cov2 = np.array([[1.0, 0.0], [0.0, 2.0]])

        w1, w2 = 0.5, 0.5

        # whitening: transform GMM so that target density has mean 0 and
        # identity covariance to stabilize training
        m, W = mean_std_for_norm(mu1, mu2, cov1, cov2, w1, w2)

        gmm = GMMSystem(
            mus=[mu1, mu2], covs=[cov1, cov2], log_w=np.log(np.array([w1, w2]))
        )

        affine = AffineSystem(gmm, W=W, m=m)

        system = TemperatureSystem(affine, t=tau)

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

    freqs = 3

    model_adj, optimizer_adj = load_model(
        input_size + 2 * freqs, args.hidden_size, output_size, args.num_hidden,
        args.lr
    )

    model_crt, optimizer_crt = load_model(
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

            X0, X1 = euler_maruyama(
                t, model_adj, num_t, dt, sigma, freqs, x_vec_dim, args.n_paths,
                dB, np_rng
            )

            # to get var of X1, integrate sigma(t)^2 from 0 to 1 to find
            # cumulative var of full path
            var = np.sum(sigma(t)**2 * dt)

            X1_tensor = torch.tensor(X1, dtype=torch.float32)

            h_phi = model_crt(X1_tensor)

            if args.energy_type == 'gmm':
                w_grad_E = system.gradenergy(X1)
                w_grad_E = clipper.clip(w_grad_E) + h_phi.detach().numpy()
            else:
                w_grad_g = (X1 / tau) + h_phi.detach().numpy()

            buffer_adj.add(X0, X1, w_grad_E)

            accum_loss_adj = 0
            for _ in range(args.n_inner_loop):
                accum_loss_adj += train_drift(
                    buffer_adj, sigma, dt, sigma_max, sigma_diff, x_vec_dim,
                    args.batch_size, np_rng, freqs, model_adj, optimizer_adj
                )

            if i == args.stage - 1:
                drift.append(drift_per_epoch(freqs, X1, model_adj))

                if args.energy_type == 'gmm':
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

            X0, X1 = euler_maruyama(
                t, model_adj, num_t, dt, sigma, freqs, x_vec_dim, args.n_paths,
                dB, np_rng
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
    X0_val, X1_val = euler_maruyama(
        t, model_adj, num_t, dt, sigma, freqs, x_vec_dim, 2000, None, np_rng
    )

    X0_x_slice = X0_val[:, 0]
    X0_y_slice = X0_val[:, 1]

    epochs = np.arange(0, args.epochs_adj, 1)
    if args.energy_type == 'gmm':
        print('whitened mean:', np.mean(X1_val, axis=0))
        print('whitened cov:', np.cov(X1_val.T))

        X1_unwhitened = np.linalg.solve(W, X1_val.T).T + m

        X1_x_slice = X1_unwhitened[:, 0]
        X1_y_slice = X1_unwhitened[:, 1]

        x_th, y_th, boltz = plot_theoretical_contour(
            args.energy_type, tau, gmm
        )
        ml_mean, ml_std, th_mean, th_std = plot_1D_slices(
            args.energy_type, x_th, y_th, boltz, X1_x_slice
        )

        # training check: should be approx. 0 if sampler learns correct drift
        plt.plot(epochs, avg_w_grad_E)
        plt.title('Average whitened gradient of the energy vs. epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Avg. whitened grad. E')
        plt.tight_layout()
        plt.savefig('avg_w_grad_E.png')
        plt.close()
    else:
        X1_x_slice = X1_val[:, 0]
        X1_y_slice = X1_val[:, 1]

        x_th, y_th, boltz = plot_theoretical_contour(
            args.energy_type, tau, None
        )
        ml_mean, ml_std, th_mean, th_std = plot_1D_slices(
            args.energy_type, x_th, y_th, boltz, X1_x_slice
        )

    plot_initial_final_contours(args.energy_type, X0_x_slice, X0_y_slice, 'X0')
    plot_initial_final_contours(args.energy_type, X1_x_slice, X1_y_slice, 'X1')

    # training check: should be approx. 0 if sampler learns correct drift
    drift = np.array(drift)
    mean_drift_dim0 = drift.mean(axis=1)[:, 0]
    plt.plot(epochs, mean_drift_dim0)
    plt.title('Drift vs. epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Drift')
    plt.tight_layout()
    plt.savefig('drift.png')
    plt.close()

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


def seed_all(seed):
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    return py_rng, np_rng


def load_model(input_dim, hidden_dim, output_dim, num_hidden, lr):
    model = MLP(
        input_dim, hidden_dim, output_dim, num_hidden, activation='gelu'
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, optimizer


def euler_maruyama(
    t, model, num_t, dt, sigma, freqs, x_vec_dim, n_paths, dB, np_rng
):
    if dB is None:
        dB = np_rng.normal(0, np.sqrt(dt), size=(num_t, n_paths, x_vec_dim))

    x = np.zeros((num_t, n_paths, x_vec_dim))

    # Gaussian distribution at t = 0
    x[0] = np_rng.standard_normal((n_paths, x_vec_dim))

    t_fourier = fourier(t, freqs)
    t_fourier = torch.tensor(t_fourier, dtype=torch.float32)

    # Euler-Maruyama with no gradient
    for i in range(num_t - 1):
        x_i = torch.tensor(x[i], dtype=torch.float32)
        t_i = t_fourier[i].repeat(n_paths, 1)

        features = torch.cat([x_i, t_i], dim=1)

        with torch.no_grad():
            u_theta = model(features).numpy()

        x[i+1] = x[i] + (sigma(t[i]) * u_theta * dt) + (sigma(t[i]) * dB[i])

    return x[0], x[-1]


def mean_std_for_norm(mu1, mu2, cov1, cov2, w1, w2):
    m = w1 * mu1 + w2 * mu2

    # derived from law of total variance
    cov_gmm = (
        w1 * (cov1 + np.outer(mu1 - m, mu1 - m))
        + w2 * (cov2 + np.outer(mu2 - m, mu2 - m))
    )

    D, Q = np.linalg.eigh(cov_gmm)
    D_sqrt = np.diag(1 / np.sqrt(D))
    W = D_sqrt @ Q.T  # PCA-whitening

    return m, W


def get_Xt_label_weight(
    sample_t, sample_buffer, sigma, dt, sigma_max,
    sigma_diff, x_vec_dim, batch_size, np_rng
):
    sample_t = np.array(sample_t)
    assert sample_t.ndim == 1, (
        'expected sample_t to be 1D, got shape {}'.format(sample_t.shape)
    )

    # choose Xt from a distribution of paths passing through Xt that all end at
    # X1; the distribution at t = 1 is a Dirac delta with mean X1 and var 0
    X0_dim = 2
    X1 = sample_buffer[:, X0_dim:X0_dim + x_vec_dim]
    grad_E = sample_buffer[:, X0_dim + x_vec_dim:X0_dim + 2 * x_vec_dim]

    c_s = sigma_diff**(-2 * sample_t)
    c_1 = sigma_diff**(-2)

    # conditional mean and var of X_t given X_1 under the Brownian base
    # process, derived using Bayes' rule and completing the square on Gaussian
    # densities
    mean_Xt = ((c_s - 1) / (c_1 - 1))[:, None] * X1
    var_Xt = sigma_max**2 * (c_s - 1) * (c_s - c_1) / (c_1 - 1)

    if np.any(var_Xt < 0):
        raise ValueError('one or more t values have negative var')

    std_Xt = np.sqrt(var_Xt)[:, None]

    Xt = np_rng.normal(loc=mean_Xt, scale=std_Xt)

    sigmas = sigma(sample_t)[:, None]
    label = -sigmas * grad_E
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

    assert len(Xt) == len(sample_t), (
        'len of Xt and t vectors do not match'
    )

    sample_t = fourier(sample_t, freqs)

    Xt = torch.tensor(Xt, dtype=torch.float32)
    sample_t = torch.tensor(sample_t, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.float32)
    weight = torch.tensor(weight, dtype=torch.float32)

    optimizer.zero_grad()
    features = torch.cat((Xt, sample_t), dim=1)
    prediction = model(features)
    diff = prediction - label
    batch_loss = torch.mean(weight * (diff**2))
    batch_loss.backward()
    optimizer.step()

    return batch_loss.item()


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


# Fourier transform applied to time to expand the input space and help the
# network avoid struggles with learning large changes in drift; lower frequency
# times are for learning smaller drift changes, and larger frequency times are
# for learning larger drift changes
def fourier(t, freqs):
    t = np.asarray(t)
    t = t[:, None]

    i = np.arange(1, freqs+1)

    angles = i * np.pi * t
    emb_cos = np.cos(angles)
    emb_sin = np.sin(angles)

    # interleave cos and sin
    emb = np.empty((t.shape[0], 2 * freqs), dtype=np.float64)
    emb[:, 0::2] = emb_cos
    emb[:, 1::2] = emb_sin

    return emb


def drift_per_epoch(freqs, X1, model):
    # only need time at the endpoint of each path
    t_1_fourier = fourier([1.0], freqs)
    t_1 = torch.tensor(t_1_fourier, dtype=torch.float32).repeat(X1.shape[0], 1)

    X1 = torch.tensor(X1, dtype=torch.float32)

    features = torch.cat((X1, t_1), dim=1)

    with torch.no_grad():
        u_theta = model(features).numpy()

    return u_theta


def plot_theoretical_contour(energy_type, tau, system):
    npoints = 500

    x = np.linspace(-8.0, 8.0, npoints)
    y = np.linspace(-8.0, 8.0, npoints)
    X, Y = np.meshgrid(x, y)

    if energy_type == 'gmm':
        XY = np.column_stack([X.ravel(), Y.ravel()])
        boltz = np.apply_along_axis(
            lambda x: np.exp(-system.energy(x) / tau), 1, XY
        )
        boltz = boltz.reshape(npoints, npoints)
    else:
        E = 0.5 * (X**2 + Y**2)
        boltz = np.exp(-E / tau)

    # normalize Boltzmann distribution
    boltz /= np.trapezoid(np.trapezoid(boltz, x, axis=1), y, axis=0)

    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, boltz, levels=50)
    plt.colorbar(label=r'$\mu(x)$')
    plt.title(
        f'2D Boltzmann distribution from theoretical data ({energy_type})'
    )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'boltz_2d_{energy_type}.png')
    plt.close()

    return x, y, boltz


def plot_1D_slices(energy_type, x, y, boltz, X1_x_slice):
    # turn p(x, y) into p(x)
    boltz_slice = np.trapezoid(boltz, y, axis=0)
    boltz_slice /= np.trapezoid(boltz_slice, x)

    plt.hist(X1_x_slice, bins=200, density=True, label=r'ML-predicted')
    plt.plot(x, boltz_slice, label='Theoretical')
    plt.title(f'1D slice of Boltzmann distribution at y = 0 ({energy_type})')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x, y=0)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'boltz_{energy_type}_slice.png')
    plt.close()

    mean = np.trapezoid(boltz_slice * x, x)
    var = np.trapezoid(boltz_slice * (x - mean)**2, x)

    return np.mean(X1_x_slice), np.std(X1_x_slice), mean, np.sqrt(var)


def plot_initial_final_contours(energy_type, X_x_slice, X_y_slice, tag):
    output_name = f'boltz_2d_ml_{energy_type}_' + tag + '.png'

    plt.figure(figsize=(8, 6))
    plt.hist2d(
        X_x_slice, X_y_slice, bins=70, range=[[-10, 10], [-10, 10]],
        density=True, alpha=0.9
    )
    plt.colorbar(label=r'$\mu(x)$')
    plt.title(
        f'2D Boltzmann distribution from ML samples ({energy_type})'
    )
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(output_name)
    plt.close()


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
