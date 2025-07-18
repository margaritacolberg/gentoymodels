import numpy as np

from adjoint import *

def main():
    mu1 = np.array([2.0, 2.0])
    cov1 = np.array([[2.0, 0.0], [0.0, 3.0]])
    mu2 = np.array([-4.0, -4.0])
    cov2 = np.array([[1.0, 0.0], [0.0, 2.0]])

    w1, w2 = 0.5, 0.5

    m, W = mean_std_for_norm(mu1, mu2, cov1, cov2, w1, w2)

    mu1_white = W @ (mu1 - m)
    cov1_white = W @ cov1 @ W.T
    mu2_white = W @ (mu2 - m)
    cov2_white = W @ cov2 @ W.T

    tau = 1.0

    points = [
        np.array([-1.0, -0.5]),       # between the modes
        np.array([0.0, 0.0]),         # near origin
        np.array([2.0, 2.0]),         # at mu1
        np.array([-4.0, -4.0]),       # at mu2
        np.array([-3.0, -2.0]),       # off-axis
    ]

    npoints = len(points)
    for i in range(npoints):
        E1 = calculate_whitened_E(points[i], mu1_white, mu2_white, \
                cov1_white, cov2_white, w1, w2, tau, W, m)
        y = W @ (points[i] - m)
        E2 = calculate_whitened_E_v2(y, W, m, mu1, mu2, cov1, cov2, w1, w2, \
                tau)
        assert np.allclose(E1, E2, atol=1e-6)

    grad_E = np.zeros((npoints, 2))
    grad_fd = np.zeros((npoints, 2))
    w_grad_E = np.zeros((npoints, 2))
    w_grad_fd = np.zeros((npoints, 2))
    for i in range(npoints):
        grad_E[i] = calculate_grad_E(points[i], mu1, mu2, cov1, cov2, w1, w2, \
                tau)

        grad_fd[i] = finite_difference(
            lambda x: calculate_E(x, mu1, mu2, cov1, cov2, w1, w2, tau),
            points[i]
        )

        w_grad_E[i] = calculate_whitened_grad_E(points[i], mu1_white, \
                mu2_white, cov1_white, cov2_white, w1, w2, tau, W, m)

        w_grad_fd[i] = finite_difference(
            lambda y: calculate_whitened_E(y, mu1_white, mu2_white, \
                    cov1_white, cov2_white, w1, w2, tau, W, m),
            points[i]
        )

        assert np.allclose(grad_E[i], w_grad_E[i], atol=1e-6)
        assert np.allclose(grad_fd[i], w_grad_fd[i], atol=1e-6)


def calculate_E(x, mu1, mu2, cov1, cov2, w1, w2, tau):
    p1 = multivariate_normal.pdf(x, mean=mu1, cov=cov1)
    p2 = multivariate_normal.pdf(x, mean=mu2, cov=cov2)

    p = w1 * p1 + w2 * p2

    return -tau * np.log(p)


def calculate_whitened_E(x, mu1, mu2, cov1, cov2, w1, w2, tau, W, m):
    y = W @ (x - m)
    return calculate_E(y, mu1, mu2, cov1, cov2, w1, w2, tau)


def calculate_whitened_E_v2(y, W, m, mu1, mu2, cov1, cov2, w1, w2, tau):
    y = np.linalg.solve(W, y) + m
    p1 = multivariate_normal.pdf(y, mean=mu1, cov=cov1)
    p2 = multivariate_normal.pdf(y, mean=mu2, cov=cov2)

    p = w1 * p1 + w2 * p2
    _, logabsdet = np.linalg.slogdet(W)

    return -tau * (np.log(p) - logabsdet)


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


if __name__ == "__main__":
    main()
