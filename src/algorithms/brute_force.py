import faiss
import numpy as np
from src.algorithms.baseIndex import BaseANNIndex
from src.algorithms.utils import AttributeIndex, BitsetAttributeIndex
from src.algorithms.utils import HybridBloomEncoder, TagAssigner, TagEncoder
from concurrent.futures import ThreadPoolExecutor
import os

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

    def search_single(q_idx):
        q_vec = vectors[q_idx]
        current_filter = filters[q_idx]
        valid_ids = self.attribute_index.get_valid_ids_conj(current_filter)
        sel = faiss.IDSelectorBatch(
            len(valid_ids),
            faiss.swig_ptr(valid_ids)
        )

        params = faiss.SearchParametersIVF(sel=sel)

        dist, indices = self.index.search(
            q_vec.reshape(1, -1),
            k,
            params=params
        )

        params.sel = None
        del params
        del sel

        return q_idx, dist[0], indices[0]

    num_threads = os.cpu_count() or 1
    faiss.omp_set_num_threads(1)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(search_single, range(len(vectors)))

    faiss.omp_set_num_threads(num_threads)

    for q_idx, dist, indices in results:
        D[q_idx] = dist
        I[q_idx] = indices

    return D, I

@BruteForceIdFilter.register_query("structured", "CNF")
def query_structured_CNF(self, vectors, filters, k, parameters):
    D = np.full((len(vectors), k), np.inf, dtype='float32')
    I = np.full((len(vectors), k), -1, dtype='int64')


    def search_single(q_idx):
        q_vec = vectors[q_idx]
        current_filter = filters[q_idx]
        valid_ids = self.attribute_index.get_valid_ids_cnf(current_filter)
        sel = faiss.IDSelectorBatch(
            len(valid_ids),
            faiss.swig_ptr(valid_ids)
        )

        params = faiss.SearchParametersIVF(sel=sel)

        dist, indices = self.index.search(
            q_vec.reshape(1, -1),
            k,
            params=params
        )

        params.sel = None
        del params
        del sel

        return q_idx, dist[0], indices[0]

    num_threads = os.cpu_count() or 1
    faiss.omp_set_num_threads(1)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(search_single, range(len(vectors)))

    faiss.omp_set_num_threads(num_threads)

    for q_idx, dist, indices in results:
        D[q_idx] = dist
        I[q_idx] = indices

    return D, I

@BruteForceIdFilter.register_query("sparse", "conjunction")
def query_sparse_conjunction(self, vectors, filters, k, parameters):
    D = np.full((len(vectors), k), np.inf, dtype='float32')
    I = np.full((len(vectors), k), -1, dtype='int64')

    def search_single(q_idx):
        q_vec = vectors[q_idx]
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
                
                if not valid_ids_set:
                    break

        if valid_ids_set is not None and not valid_ids_set:
            return q_idx, np.full(k, np.inf, dtype='float32'), np.full(k, -1, dtype='int64')
        
        valid_ids = np.array(list(valid_ids_set), dtype='int64')
        params = faiss.SearchParametersIVF()

        sel = faiss.IDSelectorBatch(
            len(valid_ids),
            faiss.swig_ptr(valid_ids)
        )

        params = faiss.SearchParametersIVF(sel=sel)
        
        dist, indices = self.index.search(
            q_vec.reshape(1, -1),
            k,
            params=params
        )
        
        params.sel = None
        del params
        del sel

        return q_idx, dist[0], indices[0]

    num_threads = os.cpu_count() or 1
    faiss.omp_set_num_threads(1)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = executor.map(search_single, range(len(vectors)))

    faiss.omp_set_num_threads(num_threads)

    for q_idx, dist, indices in results:
        D[q_idx] = dist
        I[q_idx] = indices

    return D, I


@BruteForceIdFilter.register_build("sparse-")
def build_sparse_translator(self, vectors, attributes, parameters):
    # 500+ for 95%+ recall
    print("using sparse build translator")
    num_bins = 500
    print("num_bins = ", num_bins)

    tag_assigner = TagAssigner(attributes, num_bins)
    assignment = tag_assigner.get_assignment()
    self.attribute_encoder = TagEncoder(assignment, num_bins)
    encoded_attrs = self.attribute_encoder.get_encoded_data(attributes)
    
    return build_structured(self, vectors, encoded_attrs, parameters)

@BruteForceIdFilter.register_query("sparse-", "conjunction-")
def query_sparse_conjunction_translator(self, vectors, filters, k, parameters):
    print("using sparse query translator")
    encoded_filters = self.attribute_encoder.get_encoded_queries(filters)

    return query_structured_conjunction(self, vectors, encoded_filters, k, parameters)
