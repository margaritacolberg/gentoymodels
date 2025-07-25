import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


class System:

	def __init__(self):
		pass

	def energy(self, x):
		pass

	def gradenergy(self, x):
		pass


class GMMSystem(System):

    def __init__(self, mus, covs, log_w):
        self.mus = np.stack(mus)                  # (K, D)
        self.covs = np.stack(covs)                # (K, D, D)
        self.inv_covs = np.linalg.inv(self.covs)  # (K, D, D)
        self.log_w = log_w

        self.max_w = np.max(self.log_w)
        self.log_norm_weights = self.log_w - self.max_w - \
                np.log(np.sum(np.exp(self.log_w - self.max_w)))

    def energy(self, x):
        log_p_i = np.stack([
            np.atleast_1d(multivariate_normal.logpdf(x, mean=mu, cov=cov))
            for (mu, cov) in zip(self.mus, self.covs)
        ], axis=1)

        weighted_log_p_i = self.log_norm_weights + log_p_i

        return -logsumexp(weighted_log_p_i, axis=1, keepdims=True)

    def gradenergy(self, x):
        log_p_i = np.stack([
            np.atleast_1d(multivariate_normal.logpdf(x, mean=mu, cov=cov))
            for (mu, cov) in zip(self.mus, self.covs)
        ], axis=1)

        weighted_log_p_i = self.log_norm_weights + log_p_i

        max_p = np.max(weighted_log_p_i, axis=1, keepdims=True)
        unnorm_q = np.exp(weighted_log_p_i - max_p)
        q_i = unnorm_q / np.sum(unnorm_q, axis=1, keepdims=True)

        # both diff, grad_log_p_i are of shape (N, K, D)
        diff = x[:, None, :] - self.mus[None, :, :]
        grad_log_p_i = -np.einsum('kij,nkj->nki', self.inv_covs, diff)

        return -np.sum(q_i[:, :, None] * grad_log_p_i, axis=1)


class AffineSystem(System):

    def __init__(self, subsystem, W, m):
        self.subsystem = subsystem
        self.W = W
        self.m = m

    def energy(self, y):
        ss = self.subsystem
        y = np.atleast_2d(y)

        x = np.stack([
            np.linalg.solve(self.W, y_i) + self.m
            for y_i in y
        ])

        return ss.energy(x)

    def gradenergy(self, y):
        ss = self.subsystem
        y = np.atleast_2d(y)

        x = np.stack([
            np.linalg.solve(self.W, y_i) + self.m
            for y_i in y
        ])

        grad_x = ss.gradenergy(x)
        grad_y = np.stack([np.linalg.solve(self.W.T, g) for g in grad_x])

        return grad_y


class TemperatureSystem(AffineSystem):

	def __init__(self, subsystem, t):
		self.subsystem = subsystem
		self.t = t

	def energy(self, x):
		ss = self.subsystem
		return ss.energy(x) / self.t

	def gradenergy(self, x):
		ss = self.subsystem
		return ss.gradenergy(x) / self.t
