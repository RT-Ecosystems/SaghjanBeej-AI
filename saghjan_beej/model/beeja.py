
import numpy as np
from ..core.fluid_memory import FluidMemory
from ..core.algebraic_scattering import AlgebraicScattering
from ..core.seed_math import SeedMath

class SaghjanBeejModel:
    def __init__(self, seed_dim=1024, num_patterns=10000):
        self.seed_math = SeedMath(seed_dim)
        self.memory = FluidMemory(seed_dim)
        self.scattering = AlgebraicScattering(seed_dim, num_patterns)
        self.memory.seed = self.seed_math.get_seed()
        self.sentence_store = []  # (text, vector)

    def train(self, data_streamer, epochs=1, scatter_update_every=10):
        print("प्रशिक्षण शुरू...")
        for epoch in range(epochs):
            batch_count = 0
            for batch_vectors, batch_texts in data_streamer:
                self.memory.ingest(batch_vectors)
                for text, vec in zip(batch_texts, batch_vectors):
                    self.sentence_store.append((text, vec))
                batch_count += 1
                if batch_count % scatter_update_every == 0:
                    self.scattering.update_patterns(self.memory.get_memory_state())
            print(f"  युग {epoch+1}/{epochs} पूर्ण। स्टोर में वाक्य: {len(self.sentence_store)}")

    def ask(self, question_text):
        if not self.sentence_store:
            return "[ज्ञान खाली है]"
        q_vec = self.seed_math.text_to_vector(question_text)
        q_wave = self.scattering.encode_query(q_vec)
        answer_vec = self.scattering.scatter(q_wave)
        # सबसे समान वाक्य खोजें
        best_text = max(self.sentence_store, key=lambda item: np.dot(item[1], answer_vec))
        return best_text[0]
