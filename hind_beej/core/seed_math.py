
import numpy as np
import hashlib

class SeedMath:
    def __init__(self, seed_dim=1024):
        self.seed_dim = seed_dim
        np.random.seed(42)
        self.base_seed = np.random.randn(seed_dim) * 0.02
        self.word_vectors = {}

    def _hash_word(self, word):
        h = hashlib.sha256(word.encode('utf-8')).digest()
        seed = int.from_bytes(h[:8], 'big') % (2**32)
        rng = np.random.RandomState(seed)
        return rng.randn(self.seed_dim) * 0.1

    def get_word_vector(self, word):
        if word not in self.word_vectors:
            self.word_vectors[word] = self._hash_word(word)
        return self.word_vectors[word]

    def text_to_vector(self, text):
        words = text.strip().split()
        if not words: return np.zeros(self.seed_dim)
        vecs = [self.get_word_vector(w) for w in words]
        avg = np.mean(vecs, axis=0)
        norm = np.linalg.norm(avg)
        if norm > 1e-8: avg /= norm
        return avg

    def get_seed(self):
        return self.base_seed.copy()
