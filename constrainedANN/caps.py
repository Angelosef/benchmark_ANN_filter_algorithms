import numpy as np
import os
from constrainedANN import filter_index_py

class CapsIndex:
    def __init__(self, base_vecs, props, nc=100, algo="kmeans", mode=1, metric="L2"):
        self.base_vecs = np.ascontiguousarray(base_vecs)
        self.index = filter_index_py.FilterIndex(self.base_vecs, nc, props, algo, mode)
        
        self.metric = metric
        self.mode = mode
        self.index_path = None
        return

    def save_index(self, index_path):
        self.index_path = index_path
        os.makedirs(index_path, exist_ok=True)
        self.index.get_index(self.metric, self.index_path, self.mode)
        
        return
    
    def load_index(self):
        self.index.loadIndex(self.index_path)
        return
    
    def query(self, query_vecs, constraints, k, n_probe):
        indices = self.index.query(query_vecs, constraints, k, n_probe)
        mask = (indices != -1)
    
        safe_indices = np.where(mask, indices, 0)
        neighbor_vecs = self.base_vecs[safe_indices]
        
        if self.metric == "L2":
            diff = query_vecs[:, np.newaxis, :] - neighbor_vecs
            distances = np.sum(np.square(diff), axis=2)
        else:
            raise ValueError(f"Metric {self.metric} not implemented in Python wrapper.")
        
        distances[~mask] = np.inf

        return distances, indices

