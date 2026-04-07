import faiss
import numpy as np
from src.algorithms.baseIndex import BaseANNIndex
from src.algorithms.utils import AttributeIndex
from src.algorithms.utils import HybridBloomEncoder

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
    self.attribute_index = AttributeIndex(attributes)
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
        valid_ids = self.attribute_index.get_valid_ids_conj(current_filter)
        
        params = faiss.SearchParameters() 
        params.sel = faiss.IDSelectorBatch(len(valid_ids), faiss.swig_ptr(valid_ids))

        dist, indices = self.index.search(q_vec.reshape(1, -1), k, params=params)
        D[q_idx] = dist[0]
        I[q_idx] = indices[0]

    return D, I

@BruteForceIdFilter.register_query("structured", "CNF")
def query_structured_CNF(self, vectors, filters, k, parameters):
    D = np.full((len(vectors), k), np.inf, dtype='float32')
    I = np.full((len(vectors), k), -1, dtype='int64')

    for q_idx, q_vec in enumerate(vectors):
        current_filter = filters[q_idx]
        valid_ids = self.attribute_index.get_valid_ids_cnf(current_filter)

        params = faiss.SearchParameters() 
        params.sel = faiss.IDSelectorBatch(len(valid_ids), faiss.swig_ptr(valid_ids))

        dist, indices = self.index.search(q_vec.reshape(1, -1), k, params=params)
        D[q_idx] = dist[0]
        I[q_idx] = indices[0]

    return D, I

@BruteForceIdFilter.register_query("sparse", "conjunction")
def query_sparse_conjunction(self, vectors, filters, k, parameters):
    D = np.full((len(vectors), k), np.inf, dtype='float32')
    I = np.full((len(vectors), k), -1, dtype='int64')

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
        D[q_idx] = dist[0]
        I[q_idx] = indices[0]

    return D, I

def build_structured_boolean(self, vectors, attributes, parameters):
    self.base_vectors = vectors
    self.attributes = attributes
    sub_index = faiss.IndexFlatL2(self.dim)
    self.index = faiss.IndexIDMap(sub_index)
    
    ids = np.arange(len(vectors)).astype('int64')
    self.index.add_with_ids(self.base_vectors, ids)

def query_structured_boolean(self, vectors, filters_list, k, parameters):
    nq = vectors.shape[0]
    D = np.full((nq, k), np.inf, dtype='float32')
    I = np.full((nq, k), -1, dtype='int64')

    for q_idx in range(nq):
        q_vec = vectors[q_idx].reshape(1, -1)
        required_bit_indices = filters_list[q_idx]
        
        if len(required_bit_indices) == 0:
            dist, indices = self.index.search(q_vec, k)
        else:
            mask = (self.attributes[:, required_bit_indices] == 1).all(axis=1)
            valid_ids = np.where(mask)[0].astype('int64')
            
            if len(valid_ids) == 0:
                continue

            selector = faiss.IDSelectorBatch(valid_ids)
            dist, indices = self.index.search(q_vec, k, params=faiss.SearchParameters(sel=selector))
        
        D[q_idx] = dist[0]
        I[q_idx] = indices[0]

    return D, I

@BruteForceIdFilter.register_build("sparse-")
def build_sparse_translator(self, vectors, attributes, parameters):
    print("using sparse build translator")
    m = 4000
    # k_min = 2
    # k_max = 35
    n_head = 1024
    k_tail = 3

    print("m = ", m)
    # print("k_min =", k_min)
    # print("k_max = ", k_max)
    print("n_head = ", n_head)
    print("k_tail = ", k_tail)

    self.attribute_encoder = HybridBloomEncoder(m, n_head, k_tail)
    encoded_attrs = self.attribute_encoder.encode_csr(attributes)
    
    return build_structured_boolean(self, vectors, encoded_attrs, parameters)

@BruteForceIdFilter.register_query("sparse-", "conjunction-")
def query_sparse_conjunction_translator(self, vectors, filters, k, parameters):
    print("using sparse query translator")
    nq = vectors.shape[0]
    all_query_bits = []

    for q_idx in range(nq):
        f_start = filters.indptr[q_idx]
        f_end = filters.indptr[q_idx+1]
        required_tags = filters.indices[f_start:f_end]
        
        target_bits = self.attribute_encoder.create_query_indices(required_tags)
        all_query_bits.append(target_bits)

    return query_structured_boolean(self, vectors, all_query_bits, k, parameters)
