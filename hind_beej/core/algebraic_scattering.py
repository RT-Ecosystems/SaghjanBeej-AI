
import numpy as np

class AlgebraicScattering:
    def __init__(self, pattern_dim=1024, num_patterns=10000, learning_rate=0.05):
        self.pattern_dim = pattern_dim
        self.num_patterns = num_patterns
        self.lr = learning_rate
        self.patterns = np.random.randn(num_patterns, pattern_dim) * 0.01
        self.usage = np.zeros(num_patterns)

    def encode_query(self, query_vector):
        norm = np.linalg.norm(query_vector)
        if norm < 1e-8: return query_vector
        q = query_vector / norm
        return np.sign(q) * np.sqrt(np.abs(q))

    def scatter(self, query_wave):
        similarities = np.dot(self.patterns, query_wave)
        best_idx = np.argmax(similarities)
        self.usage[best_idx] += 1
        return self.patterns[best_idx].copy()

    def update_patterns(self, seed_vector):
        for i in range(self.num_patterns):
            diff = seed_vector - self.patterns[i]
            self.patterns[i] += self.lr * diff + np.random.randn(self.pattern_dim) * 0.001
        norms = np.linalg.norm(self.patterns, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        self.patterns /= norms

    def get_top_k(self, query_wave, k=5):
        similarities = np.dot(self.patterns, query_wave)
        top_indices = np.argsort(similarities)[-k:][::-1]
        return [(idx, similarities[idx], self.patterns[idx].copy()) for idx in top_indices]
