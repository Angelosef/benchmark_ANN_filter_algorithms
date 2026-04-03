import faiss
import numpy as np
from src.algorithms.baseIndex import BaseANNIndex
from src.algorithms.utils import build_inverted_attribute_index

class BruteForceIdFilter(BaseANNIndex):
    def __init__(self, dim, metric):
        super().__init__(dim, metric)
        self.algo_name = "BruteForceIdFilter"
    
    def name(self):
        return self.algo_name

# --- BUILD STRATEGIES ---

@BruteForceIdFilter.register_build("structured")
def build_structured(self, vectors, attributes, parameters):
    self.base_vectors = vectors
    self.base_attributes = attributes
    self.inverted_index = build_inverted_attribute_index(self.base_attributes)
    
    sub_index = faiss.IndexFlatL2(self.dim)
    self.index = faiss.IndexIDMap(sub_index)
    
    ids = np.arange(len(vectors)).astype('int64')
    self.index.add_with_ids(self.base_vectors, ids)

@BruteForceIdFilter.register_build("sparse")
def build_sparse(self, vectors, attributes, parameters):
    self.base_vectors = vectors
    self.base_attributes_csc = attributes.tocsc()
    
    sub_index = faiss.IndexFlatL2(self.dim)
    self.index = faiss.IndexIDMap(sub_index)
    
    ids = np.arange(len(vectors)).astype('int64')
    self.index.add_with_ids(self.base_vectors, ids)

# --- QUERY STRATEGIES ---
@BruteForceIdFilter.register_query("structured", "conjunction")
def query_structured_conjunction(self, vectors, filters, k, parameters):
    D = np.full((len(vectors), k), np.inf, dtype='float32')
    I = np.full((len(vectors), k), -1, dtype='int64')

    for q_idx, q_vec in enumerate(vectors):
        # current_filter is a 1D array: [dim_val_0, dim_val_1, ...]
        current_filter = filters[q_idx]
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
            
            if not valid_ids:
                break

        if valid_ids is not None and not valid_ids:
            continue

        params = faiss.SearchParameters() 
        if valid_ids is not None:
            id_array = np.array(list(valid_ids), dtype='int64')
            params.sel = faiss.IDSelectorBatch(len(id_array), faiss.swig_ptr(id_array))

        dist, indices = self.index.search(q_vec.reshape(1, -1), k, params=params)
        D[q_idx] = dist[0]
        I[q_idx] = indices[0]

    return D, I

@BruteForceIdFilter.register_query("structured", "CNF")
def query_structured_CNF(self, vectors, filters, k, parameters):
    D_updated = np.full((len(vectors), k), np.inf, dtype='float32')
    I_updated = np.full((len(vectors), k), -1, dtype='int64')

    for q_idx, q_vec in enumerate(vectors):
        current_filter = filters[q_idx]
        valid_ids = None

        # OR between values, AND between dimensions
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
            if not valid_ids: break

        if valid_ids is not None and not valid_ids: continue

        # Use SearchParameters (Note: IndexIDMap handles the selector)
        params = faiss.SearchParameters() 
        if valid_ids is not None:
            id_array = np.array(list(valid_ids), dtype='int64')
            params.sel = faiss.IDSelectorBatch(len(id_array), faiss.swig_ptr(id_array))

        dist, indices = self.index.search(q_vec.reshape(1, -1), k, params=params)
        D_updated[q_idx] = dist[0]
        I_updated[q_idx] = indices[0]

    return D_updated, I_updated

@BruteForceIdFilter.register_query("sparse", "conjunction")
def query_sparse_conjunction(self, vectors, filters, k, parameters):
    D_updated = np.full((len(vectors), k), np.inf, dtype='float32')
    I_updated = np.full((len(vectors), k), -1, dtype='int64')

    for q_idx, q_vec in enumerate(vectors):
        f_start = filters.indptr[q_idx]
        f_end = filters.indptr[q_idx+1]
        required_tags = filters.indices[f_start:f_end]

        if len(required_tags) == 0:
            valid_ids_set = None 
        else:
            valid_ids_set = None
            for tag_col in required_tags:
                c_start = self.base_attributes_csc.indptr[tag_col]
                c_end = self.base_attributes_csc.indptr[tag_col + 1]
                doc_ids_with_tag = self.base_attributes_csc.indices[c_start:c_end]

                if valid_ids_set is None:
                    valid_ids_set = set(doc_ids_with_tag)
                else:
                    valid_ids_set.intersection_update(doc_ids_with_tag)
                if not valid_ids_set: break

        if valid_ids_set is not None and not valid_ids_set: continue

        params = faiss.SearchParameters()
        if valid_ids_set is not None:
            id_array = np.array(list(valid_ids_set), dtype='int64')
            params.sel = faiss.IDSelectorBatch(len(id_array), faiss.swig_ptr(id_array))

        dist, indices = self.index.search(q_vec.reshape(1, -1), k, params=params)
        D_updated[q_idx] = dist[0]
        I_updated[q_idx] = indices[0]

    return D_updated, I_updated