from collections import deque
import numpy as np


class BatchBuffer:

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.data = deque(maxlen=buffer_size)  # automatically drops old data

    def add(self, X1_batch, grad_g_batch):
        # both X1_batch, grad_g_batch are of shape (n_paths, x_vec_dim)
        pairs = np.concatenate([X1_batch, grad_g_batch], axis=1)
        for row in pairs:
            self.data.append(row)

    def sample(self, batch_size, np_rng):
        # uniform random sample with replacement
        idx = np_rng.integers(low=0, high=len(self.data), size=batch_size)
        return np.stack([self.data[i] for i in idx], axis=0)
