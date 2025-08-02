import numpy as np

from adjoint import *
from buffer import *
from clipper import *
from systems import *


def main():
    seed = 22
    py_rng, np_rng = seed_all(seed)

    mu1 = np.array([2.0, 2.0])
    cov1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu2 = np.array([-2.0, -2.0])
    cov2 = np.array([[1.0, 0.0], [0.0, 1.0]])

    w1, w2 = 0.5, 0.5

    m, W = mean_std_for_norm(mu1, mu2, cov1, cov2, w1, w2)

    tau = 1.0

    n_paths = 100

    gmm = GMMSystem(
        mus=[mu1, mu2], covs=[cov1, cov2], log_w=np.log(np.array([w1, w2]))
    )

    affine = AffineSystem(gmm, W=W, m=m)

    system = TemperatureSystem(affine, t=tau)

    points = np.array([
        [-1.0, -0.5],       # between the modes
        [0.0, 0.0],         # near origin
        [2.0, 2.0],         # at mu1
        [-4.0, -4.0],       # at mu2
        [-3.0, -2.0],       # off-axis
    ])

    single_point = np.array([
        [-1.0, -0.5],
    ])

    single_E = gmm.gradenergy(single_point)
    single_fd = finite_difference(gmm.energy, single_point)

    assert np.allclose(single_E, single_fd, atol=1e-6)

    grad_E = gmm.gradenergy(points)
    grad_fd = finite_difference(gmm.energy, points)

    assert np.allclose(grad_E, grad_fd, atol=1e-6)

    y = (points - m) @ W.T
    E1 = affine.energy(y)
    E2 = gmm.energy(points)

    assert np.allclose(E1, E2, atol=1e-6)

    w_grad_E = affine.gradenergy(y)
    w_grad_fd = finite_difference(affine.energy, y)

    assert np.allclose(w_grad_E, w_grad_fd, atol=1e-6)

    x, y, boltz = plot_theoretical_contour('gmm', tau, gmm)
    mean, std = plot_1D_slice('gmm', x, y, boltz)
    mean_slice, std_slice = marginal_params(mu1, mu2, cov1, cov2, w1, w2)

    assert np.allclose(mean, mean_slice, atol=1e-6)
    assert np.allclose(std, std_slice, atol=1e-6)

    plot_euler_maruyama_output(n_paths, np_rng)
    test_vectorized_euler_and_gradient(n_paths, np_rng, system)

    t = np.array([1.0, 2.0])
    t_fourier = fourier(t, 3)
    t_fourier_check = fourier_check(t, 3)
    assert np.allclose(t_fourier, t_fourier_check, atol=1e-6)

    test_buffer_clearing()
    test_clipper()


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


def plot_euler_maruyama_output(n_paths, np_rng):
    dt = 0.008
    t_0 = 0.
    t_1 = 1.
    num_t = int(t_1 / dt)
    t = np.linspace(t_0, t_1, num_t)

    # geometric noise schedule
    sigma_max = 1.5
    sigma_min = 0.05
    sigma_diff = sigma_max / sigma_min
    sigma = lambda time: (
        sigma_min * sigma_diff**(1 - time) * (2 * np.log(sigma_diff))**0.5
    )

    x_vec_dim = 2

    freqs = 3

    model, _ = load_model(x_vec_dim + 2 * freqs, 32, x_vec_dim, 0.001)

    X1 = euler_maruyama(
        t, model, num_t, dt, sigma, freqs,
        x_vec_dim, n_paths, None, np_rng
    )

    X1_x_slice = X1[:, 0]
    X1_y_slice = X1[:, 1]

    plt.figure(figsize=(8, 6))
    plt.hist2d(
        X1_x_slice, X1_y_slice, bins=50, range=[[-10, 10], [-10, 10]],
        density=True
    )
    plt.colorbar(label=r'$\mu(x)$')
    plt.title(f'Initial distribution of points from Euler-Maruyama paths')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(f'euler_maruyama.png')
    plt.close()


def test_vectorized_euler_and_gradient(n_paths, np_rng, system):
    # set seed for reproducibility
    np.random.seed(42)

    dt = 0.008
    t_0 = 0.
    t_1 = 1.
    num_t = int(t_1 / dt)
    t = np.linspace(t_0, t_1, num_t)

    sigma_max = 1.5
    sigma_min = 0.05
    sigma_diff = sigma_max / sigma_min
    sigma = lambda time: (
        sigma_min * sigma_diff**(1 - time) * (2 * np.log(sigma_diff))**0.5
    )

    x_vec_dim = 2

    freqs = 3

    model, _ = load_model(x_vec_dim + 2 * freqs, 32, x_vec_dim, 0.001)

    # fix the noise across all paths
    dB_shared = np_rng.normal(0, np.sqrt(dt), size=(num_t, n_paths, x_vec_dim))

    var = np.sum(sigma(t)**2 * dt)

    X1_serial = []
    grad_g_serial = []
    for i in range(n_paths):
        x1 = euler_maruyama(
            t, model, num_t, dt, sigma, freqs,
            x_vec_dim, 1, dB_shared[:, i:i+1, :], np_rng
        )
        X1_serial.append(x1)

        w_grad_E_serial = system.gradenergy(x1)
        grad_g_serial.append(
            calculate_grad_g(x1, var, x_vec_dim, w_grad_E_serial)
        )

    X1_serial = np.concatenate(X1_serial, axis=0)
    grad_g_serial = np.concatenate(grad_g_serial, axis=0)

    X1_vectorized = euler_maruyama(
        t, model, num_t, dt, sigma, freqs,
        x_vec_dim, n_paths, dB_shared, np_rng
    )

    w_grad_E_vectorized = system.gradenergy(X1_vectorized)
    grad_g_vectorized = calculate_grad_g(
        X1_vectorized, var, x_vec_dim, w_grad_E_vectorized
    )

    assert np.allclose(X1_serial, X1_vectorized, atol=1e-6)
    assert np.allclose(grad_g_serial, grad_g_vectorized, atol=1e-6)


# implement FT in a naive but straightforward way using nested for loops;
# intentionally simple but computationally expensive, designed only to verify
# the vectorized version in the main program
def fourier_check(t, freqs):
    t_fourier = []
    for i in range(len(t)):
        t_fourier_i = []
        for j in range(1, freqs + 1):
            t_fourier_i.append(np.cos(j * np.pi * t[i]))
            t_fourier_i.append(np.sin(j * np.pi * t[i]))

        t_fourier.append(t_fourier_i)

    return np.array(t_fourier)


def test_buffer_clearing():
    np_rng = np.random.default_rng(42)
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


if __name__ == '__main__':
    main()
