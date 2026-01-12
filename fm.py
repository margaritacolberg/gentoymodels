import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.mixture import GaussianMixture

from model import *


def main(args):
    # seed numpy and torch RNGs for reproducibility
    seed = 22
    torch.manual_seed(seed)

    mu1 = [2.0, 2.0]
    cov1 = [[2.0, 0.0], [0.0, 3.0]]
    mu2 = [-4.0, -4.0]
    cov2 = [[1.0, 0.0], [0.0, 2.0]]

    w1, w2 = 0.5, 0.5

    gmm = GaussianMixture(n_components=2, random_state=seed)
    gmm.weights_ = np.array([w1, w2])
    gmm.means_ = np.array([mu1, mu2])
    gmm.covariances_ = np.array([cov1, cov2])

    # inputs: x_t_x, x_t_y, t
    input_size = 3
    # output: v_x(x_t_x, x_t_y, t), v_y(x_t_x, x_t_y, t)
    output_size = 2

    model, loss_fnc, optimizer = load_model(
        input_size, args.hidden_size, output_size, args.num_hidden, args.lr
    )

    loss = np.zeros(args.epochs)
    for i in range(args.epochs):
        x_0 = torch.randn(args.batch_size, 2)
        x_1_np, _ = gmm.sample(args.batch_size)
        x_1 = torch.from_numpy(x_1_np).float()
        t = torch.rand(args.batch_size, 1)

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

    x_1 = validate(n_traj, n_steps, model)

    x_slice = x_1[:, 0]
    y_slice = x_1[:, 1]

    plot_ml_contour(x_slice, y_slice)
    plot_theoretical_contour(gmm)


def load_model(input_dim, hidden_dim, output_dim, num_hidden, lr):
    model = MLP(
        input_dim, hidden_dim, output_dim, num_hidden, activation='gelu'
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


def validate(n_traj, n_steps, model):
    dt = 1.0 / n_steps

    x_t = torch.randn(n_traj, 2)
    t = torch.zeros(n_traj, 1)

    for _ in range(n_steps):
        features = torch.cat((x_t, t), dim=1)
        with torch.no_grad():
            v_theta = model(features)

        x_t += dt * v_theta
        t += dt

    return x_t.numpy()


def plot_ml_contour(x_slice, y_slice):
    plt.figure(figsize=(8, 6))
    plt.hist2d(
        x_slice, y_slice, bins=70, range=[[-10, 10], [-10, 10]], density=True,
        alpha=0.9
    )
    plt.colorbar(label=r'Density')
    plt.title(f'Flow matching samples (t=1)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'fm_2d_ml.png')
    plt.close()


def plot_theoretical_contour(gmm):
    npoints = 500

    x = np.linspace(-8.0, 8.0, npoints)
    y = np.linspace(-8.0, 8.0, npoints)
    X, Y = np.meshgrid(x, y)
    XY = np.column_stack([X.ravel(), Y.ravel()])

    Z = np.zeros(XY.shape[0])
    for i in range(gmm.n_components):
        mu = gmm.means_[i]
        cov = gmm.covariances_[i]
        diff = XY - mu
        inv_cov = np.linalg.inv(cov)
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        Z_i = np.exp(exponent) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
        Z += gmm.weights_[i] * Z_i

    Z = Z.reshape(npoints, npoints)

    plt.figure(figsize=(10, 8))
    plt.contourf(x, y, Z, levels=50)
    plt.colorbar(label=r'Density')
    plt.title(f'Target GMM density')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'fm_2d_th.png')
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
