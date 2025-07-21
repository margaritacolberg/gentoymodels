import numpy as np

from adjoint import *
from systems import *

def main():
    mu1 = np.array([2.0, 2.0])
    cov1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([-2.0, -2.0])
    cov2 = np.array([[1.0, 0.0], [0.0, 1.0]])

    w1, w2 = 0.5, 0.5

    m, W = mean_std_for_norm(mu1, mu2, cov1, cov2, w1, w2)

    tau = 1.0

    gmm = GMMSystem(mus=[mu1, mu2], covs=[cov1, cov2], \
            log_w=np.log(np.array([w1, w2])))

    affine = AffineSystem(gmm, W=W, m=m)

    system = TemperatureSystem(affine, t=tau)

    points = [
        np.array([-1.0, -0.5]),       # between the modes
        np.array([0.0, 0.0]),         # near origin
        np.array([2.0, 2.0]),         # at mu1
        np.array([-4.0, -4.0]),       # at mu2
        np.array([-3.0, -2.0]),       # off-axis
    ]

    npoints = len(points)
    grad_E = np.zeros((npoints, 2))
    grad_fd = np.zeros((npoints, 2))
    w_grad_E = np.zeros((npoints, 2))
    w_grad_fd = np.zeros((npoints, 2))
    for i in range(npoints):
        grad_E[i] = gmm.gradenergy(points[i])
        grad_fd[i] = finite_difference(lambda x: gmm.energy(x), points[i])

        assert np.allclose(grad_E[i], grad_fd[i], atol=1e-6)

        y = W @ (points[i] - m)
        E1 = affine.energy(y)
        E2 = gmm.energy(points[i])

        assert np.allclose(E1, E2, atol=1e-6)

        w_grad_E[i] = affine.gradenergy(y)
        w_grad_fd[i] = finite_difference(lambda x: affine.energy(x), y)

        assert np.allclose(w_grad_E[i], w_grad_fd[i], atol=1e-6)

    x, y, boltz = plot_theoretical_contour('gmm', tau, gmm)
    mean, std = plot_1D_slice('gmm', x, y, boltz)
    mean_slice, std_slice = marginal_params(mu1, mu2, cov1, cov2, w1, w2)

    assert np.allclose(mean, mean_slice, atol=1e-6)
    assert np.allclose(std, std_slice, atol=1e-6)


def finite_difference(f, x, eps=1e-6):
    grad_fd = np.zeros_like(x)

    # central difference
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad_fd[i] = (f(x_plus) - f(x_minus)) / (2 * eps)

    return grad_fd


def plot_1D_slice(energy_type, x, y, boltz):
    # turn p(x, y) into p(x)
    boltz_slice = np.trapezoid(boltz, y, axis=0)
    boltz_slice /= np.trapezoid(boltz_slice, x)

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

    return mean, np.sqrt(var)


def marginal_params(mu1, mu2, cov1, cov2, w1, w2):
    dim = 0

    weights = [w1, w2]
    mus = [mu1, mu2]
    covs = [cov1, cov2]

    mean_slice = sum(w * mu[dim] for w, mu in zip(weights, mus))

    var_slice = sum(
        w * (cov[dim, dim] + (mu[dim] - mean_slice)**2)
        for w, mu, cov in zip(weights, mus, covs)
    )

    std_slice = np.sqrt(var_slice)

    return mean_slice, std_slice


if __name__ == "__main__":
    main()
