import numpy as np
from scipy.stats import multivariate_normal


class System:

	def __init__(self):
		pass

	def energy(self, x):
		pass

	def gradenergy(self, x):
		pass


class GMMSystem():

    def __init__(self, mus, covs, log_w):
        self.mus = mus
        self.covs = covs
        self.log_w = log_w

    def energy(self, x):
        log_p_i = np.array([
            multivariate_normal.logpdf(x, mean=mu, cov=cov)
            for (mu, cov) in zip(self.mus, self.covs)
        ])

        max_w = np.max(self.log_w)
        log_norm_weights = self.log_w - max_w - \
                np.log(np.sum(np.exp(self.log_w - max_w))) 

        weighted_log_p_i = log_norm_weights + log_p_i

        max_p = np.max(weighted_log_p_i)
        return -np.log(np.sum(np.exp(weighted_log_p_i - max_p))) - max_p

    def gradenergy(self, x):
        log_p_i = np.array([
            multivariate_normal.logpdf(x, mean=mu, cov=cov)
            for (mu, cov) in zip(self.mus, self.covs)
        ])

        max_w = np.max(self.log_w)
        log_norm_weights = self.log_w - max_w - \
                np.log(np.sum(np.exp(self.log_w - max_w))) 

        weighted_log_p_i = log_norm_weights + log_p_i

        max_p = np.max(weighted_log_p_i)
        unnorm_q = np.exp(weighted_log_p_i - max_p)
        q_i = unnorm_q / np.sum(unnorm_q)

        grad_log_p_i = np.array([
            -np.linalg.solve(cov, x - mu)
            for (mu, cov) in zip(self.mus, self.covs)
        ])

        return -np.sum(q_i[:, None] * grad_log_p_i, axis=0)


class AffineSystem(System):

    def __init__(self, subsystem, W, m):
        self.subsystem = subsystem
        self.W = W
        self.m = m

    def energy(self, y):
        ss = self.subsystem
        x = np.linalg.solve(self.W, y) + self.m
        return ss.energy(x)

    def gradenergy(self, y):
        ss = self.subsystem
        x = np.linalg.solve(self.W, y) + self.m
        return np.linalg.solve(self.W.T, ss.gradenergy(x))


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
