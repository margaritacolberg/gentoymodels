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
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize

from model import *


def main(args):
    dt = 0.01
    t_0 = 0.
    t_1 = 1.
    num_t = int(t_1 / dt)
    t = np.linspace(t_0, t_1, num_t) 

    tau = 1
    buffer = []

    batch_size = 256
    n_paths = 100
    n_inner_loop = 10

    if args.energy_type == 'gmm':
        # Gaussian mixture model parameters
        #mu1 = np.array([2.0, 2.0])
        #cov1 = np.array([[2.0, 0.0], [0.0, 3.0]])

        mu1 = np.array([-4.0, -4.0])
        cov1 = np.array([[1.0, 0.0], [0.0, 2.0]])

#        mu2 = np.array([-4.0, -4.0])
#        cov2 = np.array([[1.0, 0.0], [0.0, 2.0]])

        # weights
#        w1, w2 = 0.5, 0.5
        w1 = 1.0

        sigma_max = 1.5

        # whitening
        #mean_gmm, W = mean_std_for_norm(mu1, mu2, cov1, cov2, w1, w2)
        mean_gmm, W = mean_std_for_norm(mu1, cov1, w1)

        mu1_white = W @ (mu1 - mean_gmm)
#        mu2_white = W @ (mu2 - mean_gmm)
        cov1_white = W @ cov1 @ W.T
#        cov2_white = W @ cov2 @ W.T
    else:
        sigma_max = 1.0

    # geometric noise schedule
    sigma_min = 0.05
    sigma_diff = sigma_max / sigma_min
    sigma = lambda time: sigma_min * sigma_diff**(1 - time) \
            * (2 * np.log(sigma_diff))**0.5

    u = np.zeros(num_t)

    x_vec_dim = 2

    # inputs: Xt
    input_size = x_vec_dim
    # output: u(Xt, t)
    output_size = x_vec_dim

    freqs = 3

    model, optimizer = load_model(input_size + 2 * freqs, args.hidden_size, \
            output_size, args.lr)

    for j in range(args.epochs):
        for _ in range(n_paths):
            X1 = euler_maruyama(t, model, num_t, dt, sigma, freqs, x_vec_dim)

            # to get var of X1, integrate sigma(t)^2 from 0 to 1 to find
            # cumulative var of full path
            var = np.sum(sigma(t)**2 * dt)

            if args.energy_type == 'gmm':
                #grad_g = calculate_grad_g(X1_norm, mu1_white, mu2_white, \
                 #       cov1_white, cov2_white, w1, w2, tau, var, x_vec_dim, \
                  #      mean_gmm, W)
                grad_g = calculate_grad_g(X1, mu1_white, \
                        cov1_white, w1, tau, var, x_vec_dim, \
                        mean_gmm, W)
            else:
                grad_g = (-X1 / var) + (X1 / tau)

            pair = np.concatenate([X1, grad_g])
            buffer.append(pair)

        accum_loss = 0
        for k in range(n_inner_loop):
            sample_buffer = random.choices(buffer, k=batch_size)
            sample_t = random.choices(t, k=batch_size)

            Xt, label, weight = get_Xt_label_weight(sample_t, sample_buffer, \
                    sigma, dt, sigma_max, sigma_diff, x_vec_dim)

            assert len(Xt) == len(sample_t), \
                    'len of Xt and t vectors do not match'

            sample_t = fourier(sample_t, freqs)

            Xt = torch.tensor(Xt, dtype=torch.float32)
#            Xt = Xt - Xt.mean(dim=0, keepdim=True)
            sample_t = torch.tensor(sample_t, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            weight = torch.tensor(weight, dtype=torch.float32)

            optimizer.zero_grad()
            features = torch.cat((Xt, sample_t), dim=1)
            prediction = model(features)
            diff = prediction - label
            batch_loss = torch.mean(weight * (diff**2))
            accum_loss += batch_loss.item()
            batch_loss.backward()
            optimizer.step()

        print('Epoch: {}, train loss: {}'.format(j, accum_loss))

    X1_val = []
    # validate drift by comparing distribution of X1 to Boltzmann distribution
    for _ in range(1000):
        X1_val.append(euler_maruyama(t, model, num_t, dt, sigma, freqs, \
                x_vec_dim))

    X1_val = np.array(X1_val, dtype=np.float32)
    print("Whitened mean:", np.mean(X1_val, axis=0))
    print("Whitened cov:", np.cov(X1_val.T))

    print("mean and std")
    print(np.mean(X1_val[:,0]), np.std(X1_val[:,0]))
    print(np.mean(X1_val[:,1]), np.std(X1_val[:,1]))

#    if args.energy_type == 'gmm':
#        W_inv = np.linalg.inv(W)
#        X1_val = (W_inv @ X1_val.T).T + mean_gmm

    X1_x_slice = X1_val[:, 0]
    X1_y_slice = X1_val[:, 1]

    if args.energy_type == 'gmm':
        #plot_theoretical_distribution('gmm', tau, X1_x_slice, mu1, mu2, \
         #       cov1, cov2, w1, w2)
        plot_theoretical_distribution('gmm', tau, X1_x_slice, mu1, None, \
                cov1, None, w1, None, W, mean_gmm)
    else:
        plot_theoretical_distribution('well', tau, X1_x_slice, None, None, \
                None, None, None, None)

    plt.figure(figsize=(8, 6))
    plt.hist2d(X1_x_slice, X1_y_slice, bins=50, range=[[-10, 10], [-10, 10]], \
            density=True)
    plt.colorbar(label=r'$\mu(x)$')
    plt.title(f'2D Boltzmann distribution from ML samples ({args.energy_type})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'boltz_2d_ml_{args.energy_type}.png')
    plt.close()


def load_model(input_dim, hidden_dim, output_dim, lr):
    model = MLP_adjoint(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, optimizer


def euler_maruyama(t, model, num_t, dt, sigma, freqs, x_vec_dim):
    # Gaussian noise term for infinitesimal step of Brownian motion
    dB = np.random.normal(0, np.sqrt(dt), size=(num_t, x_vec_dim))

    # Dirac distribution at t = 0
    x = np.zeros((num_t, x_vec_dim))

    t_fourier = fourier(t, freqs)

    # Euler-Maruyama with no gradient
    # https://ipython-books.github.io/134-simulating-a-stochastic-differential-equation/
    for i in range(num_t - 1):
        x_i_tensor = torch.tensor(x[i], dtype=torch.float32)
        t_i_tensor = torch.tensor(t_fourier[i], dtype=torch.float32)

        features = torch.cat((x_i_tensor, t_i_tensor), dim=0)

        with torch.no_grad():
            u_theta = model(features).numpy()

        x[i+1] = x[i] + (u_theta * dt) + (sigma(t[i]) * dB[i])

    return x[-1]


'''
def calculate_grad_g(x, mu1, mu2, cov1, cov2, w1, w2, tau, var, x_vec_dim, \
        mean_gmm, W):
    inv_cov1 = np.linalg.inv(cov1)
    inv_cov2 = np.linalg.inv(cov2)

    p1 = multivariate_normal.pdf(x, mean=mu1, cov=cov1)
    p2 = multivariate_normal.pdf(x, mean=mu2, cov=cov2)

    p = w1 * p1 + w2 * p2

    grad_p1 = -p1 * (inv_cov1 @ (x - mu1))
    grad_p2 = -p2 * (inv_cov2 @ (x - mu2))

    grad_p = w1 * grad_p1 + w2 * grad_p2

    grad_E = -tau * grad_p / p

    # base distribution is multivariate normal probability density function
    # with mean 0, since there is no drift in base process
    mu_base = np.zeros(x_vec_dim)
    cov_base = var * np.eye(x_vec_dim)
    inv_cov_base = np.linalg.inv(cov_base)

    grad_log_p_base = -inv_cov_base @ (x - mu_base)

    return grad_log_p_base + grad_E
    '''

def calculate_grad_g(x, mu1, cov1, w1, tau, var, x_vec_dim, \
        mean_gmm, W):
    x = W @ (x - mean_gmm)

    inv_cov1 = np.linalg.inv(cov1)

    p1 = multivariate_normal.pdf(x, mean=mu1, cov=cov1)

    p = w1 * p1

    grad_p1 = -p1 * (inv_cov1 @ (x - mu1))

    grad_p = w1 * grad_p1

    grad_E = -tau * grad_p / p

    grad_E = W.T @ grad_E

    # base distribution is multivariate normal probability density function
    # with mean 0, since there is no drift in base process
    mu_base = np.zeros(x_vec_dim)
    cov_base = var * np.eye(x_vec_dim)
    inv_cov_base = np.linalg.inv(cov_base)

    grad_log_p_base = -inv_cov_base @ (x - mu_base)

    return grad_log_p_base + grad_E

'''
def mean_std_for_norm(mu1, mu2, cov1, cov2, w1, w2):
    mean_gmm = w1 * mu1 + w2 * mu2

    # derived from law of total variance
    cov_gmm = w1 * (cov1 + np.outer(mu1 - mean_gmm, mu1 - mean_gmm)) + \
            w2 * (cov2 + np.outer(mu2 - mean_gmm, mu2 - mean_gmm))

#    inv_cov_gmm = np.linalg.inv(cov_gmm)

#    D, Q = np.linalg.svd(inv_cov_gmm)
#    D_sqrt = np.diag(np.sqrt(D + 1e-5))
#    W = D_sqrt @ Q.T

#    U, Lambda, _ = np.linalg.svd(cov_gmm)
#    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))

    return mean_gmm, W
    '''

def mean_std_for_norm(mu1, cov1, w1):
    mean_gmm = w1 * mu1

    # derived from law of total variance
    cov_gmm = w1 * (cov1 + np.outer(mu1 - mean_gmm, mu1 - mean_gmm))

    inv_cov_gmm = np.linalg.inv(cov_gmm)

    D, Q = np.linalg.eigh(inv_cov_gmm)
    D_sqrt = np.diag(np.sqrt(D + 1e-5))
    W = D_sqrt @ Q.T

#    U, Lambda, _ = np.linalg.svd(cov_gmm)
#    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))

    return mean_gmm, W


def get_Xt_label_weight(sample_t, sample_buffer, sigma, dt, sigma_max, \
        sigma_diff, x_vec_dim):
    # choose Xt from a distribution of paths passing through Xt that
    # all end at X1; the distribution at t = 1 is a Dirac delta with
    # mean X1 and var 0
    c_s = lambda t: sigma_diff**(-2 * t)
    c_1 = sigma_diff**(-2)
    mean_Xt = lambda t, X1: (c_s(t) - 1) / (c_1 - 1) * X1
    var_Xt = lambda t: sigma_max**2 * (c_s(t) - 1) * (c_s(t) - c_1) / (c_1 - 1)

    Xt = []
    label = []
    weight = []

    for i in range(len(sample_t)):
        X1 = sample_buffer[i][:x_vec_dim]
        grad_g = sample_buffer[i][x_vec_dim:]
        t_i = sample_t[i]
        var = var_Xt(t_i)
        sigma_i = sigma(t_i)

        if np.any(var < 0):
            raise ValueError('Negative var {} at t = {}'.format(var, t_i))

        Xt.append(np.random.normal(mean_Xt(t_i, X1), \
                np.sqrt(np.maximum(var, np.float32(0.0)))))

        label.append(-sigma_i * grad_g)

        # weight to improve numerical stability
        weight.append(np.full(x_vec_dim, 0.5 / sigma_i**2))
      
    Xt = np.array(Xt, dtype=np.float32)
    label = np.array(label, dtype=np.float32)
    weight = np.array(weight, dtype=np.float32)

    return Xt, label, weight


def fourier(t, freqs):
    t_fourier = []
    for i in range(len(t)):
        t_fourier_i = []
        for j in range(1, freqs + 1):
            t_val = t[i]
            t_fourier_i.append(np.cos(j * np.pi * t[i]))
            t_fourier_i.append(np.sin(j * np.pi * t[i]))

        t_fourier.append(t_fourier_i)

    return np.array(t_fourier)


def plot_theoretical_distribution(energy_type, tau, X1_x_slice, mu1, mu2, \
        cov1, cov2, w1, w2, W, mean_gmm):
    npoints = 800

    x = np.linspace(-10.0, 10.0, npoints)
    y = np.linspace(-10.0, 10.0, npoints)
    X, Y = np.meshgrid(x, y)

    if energy_type == 'gmm':
        XY = np.column_stack([X.ravel(), Y.ravel()])
        XY_white = (W @ (XY.T - mean_gmm[:, None])).T
        mu1_white = W @ (mu1 - mean_gmm)
        cov1_white = W @ cov1 @ W.T

        p1 = multivariate_normal.pdf(XY_white, mean=mu1_white, cov=cov1_white)
    #    p2 = multivariate_normal.pdf(XY, mean=mu2, cov=cov2)
    #    p = w1 * p1 + w2 * p2
        p = w1 * p1

        boltz = p.reshape(npoints, npoints)
    else:
        E = 0.5 * (X**2 + Y**2)
        boltz = np.exp(-E / tau)

    # normalize Boltzmann distribution
    boltz /= np.trapezoid(np.trapezoid(boltz, x, axis=1), y, axis=0)

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, boltz, levels=50)
    plt.colorbar(label=r'$\mu(x)$')
    plt.title(f'2D Boltzmann distribution ({energy_type})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'boltz_2d_{energy_type}.png')
    plt.close()

    # turn p(x, y) into p(x)
    boltz_slice = np.trapezoid(boltz, y, axis=0)
    boltz_slice /= np.trapezoid(boltz_slice, x)

    x_white = (W @ (x.T - mean_gmm[:, None])).T

    plt.hist(X1_x_slice, bins=100, density=True, label=r'ML-predicted')
    plt.plot(x_white, boltz_slice, label='Theoretical')
    plt.title(f'1D slice of Boltzmann distribution at y = 0 ({energy_type})')
    plt.xlabel('x')
    plt.ylabel(r'$\mu(x, y=0)$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'boltz_{energy_type}_slice.png')
    plt.close()

    mean = np.trapezoid(boltz_slice * x, x)
    var = np.trapezoid(boltz_slice * (x - mean)**2, x)

    print('Mean: {}, std: {} of ML-predicted distribution'.format( \
            np.mean(X1_x_slice), np.std(X1_x_slice)))
    print('Mean: {}, std: {} of theoretical distribution'.format( \
            mean, np.sqrt(var)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--energy_type', type=str, default='well', \
            choices=['well', 'gmm'])

    args = parser.parse_args()

    main(args)
