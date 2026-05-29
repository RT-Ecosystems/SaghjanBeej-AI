
import threading, queue, numpy as np

class DataStreamer:
    def __init__(self, file_paths, batch_size=64, streaming=True, seed_math=None):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.streaming = streaming
        self.seed_math = seed_math
        self._queue = queue.Queue(maxsize=10)
        if streaming:
            self._start_workers()

    def _start_workers(self):
        def worker():
            for path in self.file_paths:
                with open(path, "r", encoding="utf-8") as f:
                    batch_texts = []
                    for line in f:
                        line = line.strip()
                        if line: batch_texts.append(line)
                        if len(batch_texts) >= self.batch_size:
                            self._queue.put(batch_texts)
                            batch_texts = []
                    if batch_texts:
                        self._queue.put(batch_texts)
            self._queue.put(None)
        threading.Thread(target=worker, daemon=True).start()

    def __iter__(self):
        return self

    def __next__(self):
        batch_texts = self._queue.get()
        if batch_texts is None:
            raise StopIteration
        if self.seed_math:
            batch_vectors = np.array([self.seed_math.text_to_vector(t) for t in batch_texts])
        else:
            batch_vectors = np.random.randn(len(batch_texts), 1024)
        return batch_vectors, batch_texts
