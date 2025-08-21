import numpy as np

from adjoint import *
from buffer import *
from clipper import *
from systems import *


# helper functions
def finite_difference(f, x, eps=1e-6):
    grad_fd = np.zeros_like(x)
    n_points, dim = x.shape

    for n in range(n_points):
        for i in range(dim):
            x_plus = x.copy()
            x_minus = x.copy()

            x_plus[n, i] += eps
            x_minus[n, i] -= eps

            f_plus = f(x_plus)[n].item()
            f_minus = f(x_minus)[n].item()

            grad_fd[n, i] = (f_plus - f_minus) / (2 * eps)

    return grad_fd


def marginal_params(mus, covs, weights, dim=0):
    mean_slice = sum(w * mu[dim] for w, mu in zip(weights, mus))

    var_slice = sum(
        w * (cov[dim, dim] + (mu[dim] - mean_slice)**2)
        for w, mu, cov in zip(weights, mus, covs)
    )

    return mean_slice, np.sqrt(var_slice)


# tests
def test_gmm_gradients():
    mu1 = np.array([2.0, 2.0])
    mu2 = np.array([-2.0, -2.0])
    cov1, cov2 = np.eye(2), np.eye(2)
    w1, w2 = 0.5, 0.5

    mus = [mu1, mu2]
    covs = [cov1, cov2]
    weights = [w1, w2]

    system, m, W, gmm = create_normalized_gmm_system(
        mus, covs, weights, tau=1.0
    )

    points = np.array([
        [-1.0, -0.5],       # between the modes
        [0.0, 0.0],         # near origin
        [2.0, 2.0],         # at mu1
        [-4.0, -4.0],       # at mu2
        [-3.0, -2.0],       # off-axis
    ])

    # test single point gradient
    single_point = np.array([[-1.0, -0.5]])
    grad_E = gmm.gradenergy(single_point)
    grad_fd = finite_difference(gmm.energy, single_point)
    assert np.allclose(grad_E, grad_fd, atol=1e-6)

    # test multiple points gradient
    grad_E = gmm.gradenergy(points)
    grad_fd = finite_difference(gmm.energy, points)
    assert np.allclose(grad_E, grad_fd, atol=1e-6)


def test_affine_system_energy():
    mu1 = np.array([2.0, 2.0])
    mu2 = np.array([-2.0, -2.0])
    cov1, cov2 = np.eye(2), np.eye(2)
    w1, w2 = 0.5, 0.5

    mus = [mu1, mu2]
    covs = [cov1, cov2]
    weights = [w1, w2]

    system, m, W, gmm = create_normalized_gmm_system(
        mus, covs, weights, tau=1.0
    )
    affine = AffineSystem(gmm, W, m)

    points = np.array([
        [-1.0, -0.5],
        [0.0, 0.0],
        [2.0, 2.0],
        [-4.0, -4.0],
        [-3.0, -2.0],
    ])

    y = (points - m) @ W.T
    E1 = affine.energy(y)
    E2 = gmm.energy(points)
    assert np.allclose(E1, E2, atol=1e-6)

    w_grad_E = affine.gradenergy(y)
    w_grad_fd = finite_difference(affine.energy, y)
    assert np.allclose(w_grad_E, w_grad_fd, atol=1e-6)


def test_marginal_params():
    mu1 = np.array([2.0, 2.0])
    mu2 = np.array([-2.0, -2.0])
    cov1, cov2 = np.eye(2), np.eye(2)
    w1, w2 = 0.5, 0.5

    mus = [mu1, mu2]
    covs = [cov1, cov2]
    weights = [w1, w2]

    _, _, _, gmm = create_normalized_gmm_system(mus, covs, weights, tau=1.0)

    x, y, boltz = plot_theoretical_contour('gmm', tau=1.0, system=gmm)
    boltz_slice, mean, std = theoretical_1D_slice_stats(x, y, boltz)
    mean_slice, std_slice = marginal_params(mus, covs, weights)

    # allow looser tolerance due to numeric integration
    assert np.allclose(mean, mean_slice, atol=1e-3)
    assert np.allclose(std, std_slice, atol=1e-3)

    # check slice is normalized
    assert np.isclose(np.trapezoid(boltz_slice, x), 1.0, atol=1e-6)


def test_buffer_clearing():
    np_rng = np.random.default_rng(22)
    buffer = BatchBuffer(buffer_size=5)

    X1_batch = np.array([[1, 1], [2, 2], [3, 3]])
    grad_g_batch = np.array([[10, 10], [20, 20], [30, 30]])
    buffer.add(X1_batch, grad_g_batch)

    # add more items than buffer_size to see FIFO behaviour
    X1_batch2 = np.array([[4, 4], [5, 5], [6, 6]])
    grad_g_batch2 = np.array([[40, 40], [50, 50], [60, 60]])
    buffer.add(X1_batch2, grad_g_batch2)

    # the buffer should keep only the last 5 items
    # the first item, (X1=[1,1], grad_g=[10,10]), should be gone
    first_item = np.concatenate([[1, 1], [10, 10]])

    exists = any(np.array_equal(data, first_item) for data in buffer.data)

    assert not exists, (
        'test failed: first item should have been removed from buffer'
    )


def test_clipper():
    clipper = Clipper(max_norm=5.0)

    # case 1: vector norm smaller than max_norm (no change)
    small = np.array([[3.0, 4.0]])  # norm = 5.0, exactly at boundary
    out = clipper.clip(small)

    assert np.allclose(out, small)

    # case 2: vector norm larger than max_norm (must clip)
    big = np.array([[6.0, 8.0]])  # norm = 10.0 > max_norm
    out = clipper.clip(big)
    norm_out = np.linalg.norm(out)

    assert np.isclose(norm_out, 5.0, atol=1e-6)

    # case 3: multiple vectors; first vector norm larger than max_norm but
    # second vector norm smaller than max_norm (clip only the first)
    batch = np.array([[6.0, 8.0], [1.0, 2.0]])
    out = clipper.clip(batch)
    norms_out = np.linalg.norm(out, axis=-1)

    assert np.isclose(norms_out[0], 5.0, atol=1e-6)
    assert np.isclose(norms_out[1], np.linalg.norm([1.0, 2.0]))
