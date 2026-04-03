from ACORN.acorn_class import IndexACORNFlat
import numpy as np
from src.algorithms.baseIndex import BaseANNIndex
from src.algorithms.utils import build_inverted_attribute_index

class AcornBuildParameters:
    def __init__(self, M=32, gamma=12, M_beta=32):
        self.M = M
        self.gamma = gamma
        self.M_beta = M_beta

class AcornQueryParameters:
    def __init__(self, efSearch=10):
        self.efSearch = efSearch

class Acorn(BaseANNIndex):
    def __init__(self, dim, metric):
        super().__init__(dim, metric)
        self.algo_name = "Acorn"
    
    def name(self):
        return self.algo_name

# --- BUILD STRATEGIES ---

@Acorn.register_build("structured")
def build_structured(self, vectors, attributes, parameters):
    self.base_vectors = vectors
    self.base_attributes = attributes
    self.inverted_index = build_inverted_attribute_index(self.base_attributes)
    self.build_params = parameters
    self.batch_size = 64
    self.index = IndexACORNFlat(
        self.dim,
        self.build_params.M,
        self.build_params.gamma,
        attributes,
        self.build_params.M_beta
    )

# --- QUERY STRATEGIES ---
@Acorn.register_query("structured", "conjunction")
def query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_params = parameters
    nq = len(vectors)
    nb = self.base_attributes.shape[0]
    D = np.full((len(vectors), k), np.inf, dtype='float32')
    I = np.full((len(vectors), k), -1, dtype='int64')

    self.index.efSearch = self.query_params.efSearch

    for start_idx in range(0, nq, self.batch_size):
        end_idx = min(start_idx + self.batch_size, nq)
        batch_size = end_idx - start_idx
        filter_map = np.zeros((batch_size, nb), dtype='int8')

        for i in range(start_idx, end_idx):
            rel_i = i - start_idx
            current_filter = filters[i]
            valid_ids = None

            for dim_idx, required_val in enumerate(current_filter):
                # -1 acts as a wildcard; skip this dimension
                if required_val == -1:
                    continue
                
                key = (dim_idx, required_val)
                ids_for_this_dim = set(self.inverted_index.get(key, []))

                if valid_ids is None:
                    valid_ids = ids_for_this_dim
                else:
                    valid_ids.intersection_update(ids_for_this_dim)
            if valid_ids is None:
                valid_ids = np.arange(nb)
            filter_map[rel_i][list(valid_ids)] = 1
        
        batch_q_vecs = vectors[start_idx:end_idx]
        D_batch, I_batch = self.index.search(batch_q_vecs, k, filter_map)
        D[start_idx:end_idx] = D_batch
        I[start_idx:end_idx] = I_batch

    return D, I

@Acorn.register_query("structured", "CNF")
def query_structured_CNF(self, vectors, filters, k, parameters):
    self.query_params = parameters
    nq = len(vectors)
    nb = self.base_attributes.shape[0]
    D = np.full((len(vectors), k), np.inf, dtype='float32')
    I = np.full((len(vectors), k), -1, dtype='int64')

    self.index.efSearch = self.query_params.efSearch

    for start_idx in range(0, nq, self.batch_size):
        end_idx = min(start_idx + self.batch_size, nq)
        batch_size = end_idx - start_idx
        filter_map = np.zeros((batch_size, nb), dtype='int8')

        for i in range(start_idx, end_idx):
            rel_i = i - start_idx
            current_filter = filters[i]
            valid_ids = None

            for dim_idx, valid_vals in enumerate(current_filter):
                dim_union = set()
                for val in valid_vals:
                    if val == -1: continue 
                    key = (dim_idx, val)
                    if key in self.inverted_index:
                        dim_union.update(self.inverted_index[key])
                
                if valid_ids is None:
                    valid_ids = dim_union
                else:
                    valid_ids.intersection_update(dim_union)
                    
            filter_map[rel_i][list(valid_ids)] = 1
        
        batch_q_vecs = vectors[start_idx:end_idx]
        D_batch, I_batch = self.index.search(batch_q_vecs, k, filter_map)
        D[start_idx:end_idx] = D_batch
        I[start_idx:end_idx] = I_batch

    return D, I
