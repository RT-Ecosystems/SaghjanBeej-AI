
import numpy as np
from collections import deque

class FluidMemory:
    def __init__(self, seed_dim=1024, novelty_threshold=0.3, history_size=1000):
        self.seed_dim = seed_dim
        self.seed = np.zeros(seed_dim)
        self.novelty_threshold = novelty_threshold
        self.history = deque(maxlen=history_size)

    def _compute_summary(self, batch_vectors):
        mean_vec = np.mean(batch_vectors, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 1e-8: mean_vec /= norm
        return mean_vec

    def _is_novel(self, summary_vec):
        if not self.history: return True
        similarities = [np.dot(summary_vec, old) for old in self.history]
        return max(similarities) < (1 - self.novelty_threshold)

    def ingest(self, data_batch):
        summary = self._compute_summary(data_batch)
        if self._is_novel(summary):
            self.seed = 0.9 * self.seed + 0.1 * summary
            norm = np.linalg.norm(self.seed)
            if norm > 1e-8: self.seed /= norm
        self.history.append(summary)
        return self.seed.copy()

    def get_memory_state(self):
        return self.seed.copy()
