import numpy as np


class Clipper:

    def __init__(self, max_norm=None):
        # if max_norm is None, clipping is disabled
        self.max_norm = max_norm

    def clip(self, scores: np.ndarray) -> np.ndarray:
        # clips scores so their L2 norm does not exceed max_norm; scores can be
        # of shape (N, D) or (D, )
        if self.max_norm is None:
            return scores

        scores = np.atleast_2d(scores)
        # if scores is of shape (N, D), reduce along D dimension and keep the
        # 2D shape, (N, 1)
        norms = np.linalg.norm(scores, axis=-1, keepdims=True)
        # scaling factor for each score vector: 1.0 if norm <= max_norm, else
        # scale down so norm = max_norm
        coef = np.minimum(1.0, self.max_norm / (norms + 1e-6))

        return scores * coef
