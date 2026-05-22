from src.algorithms.baseIndex import BaseANNIndex
import faiss
import numpy as np
from src.algorithms.utils import AttributeIndex
from concurrent.futures import ThreadPoolExecutor
import os

class IVFIdFilterBuildParameters:
    def __init__(self, nlist=100):
        self.nlist = nlist

class IVFIdFilterQueryParameters:
    def __init__(self, nprobe=10):
        self.nprobe = nprobe

class IVFIdFilter(BaseANNIndex):

    def __init__(self, dim, metric):
        super().__init__(dim, metric)
        self.algo_name = "IVFIdFilter"
    
    def name(self):
        return self.algo_name

@IVFIdFilter.register_build("structured")
def build_structured(self, vectors, attributes, parameters):
    self.build_parameters = parameters

    self.attribute_index = AttributeIndex(attributes)

    quantizer = faiss.IndexFlatL2(self.dim)
    self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.build_parameters.nlist)
    train_size = int(len(vectors) * 0.1)
    self.index.train(vectors[:train_size])
    self.index.add(vectors)
    return

@IVFIdFilter.register_query("structured", "conjunction")
def query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_parameters = parameters

    D = np.full((len(vectors), k), np.inf)
    I = np.full((len(vectors), k), -1)

    def search_single(q_idx):
        q_vec = vectors[q_idx]
        current_filter = filters[q_idx]
        valid_ids = self.attribute_index.get_valid_ids_conj(current_filter)
        sel = faiss.IDSelectorBatch(
            len(valid_ids),
            faiss.swig_ptr(valid_ids)
        )

        params = faiss.SearchParametersIVF(sel=sel)
        params.nprobe = self.query_parameters.nprobe

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


@IVFIdFilter.register_query("structured", "CNF")
def query_structured_CNF(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
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
        params.nprobe = self.query_parameters.nprobe

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

@IVFIdFilter.register_build("sparse")
def build_sparse(self, vectors, attributes, parameters):
    self.base_attributes_csc = attributes.tocsc()
    self.build_parameters = parameters

    quantizer = faiss.IndexFlatL2(self.dim)
    self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.build_parameters.nlist)
    train_size = int(len(vectors) * 0.1)
    self.index.train(vectors[:train_size])
    self.index.add(vectors)
    return

@IVFIdFilter.register_query("sparse", "conjunction")
def query_sparse_conjunction(self, vectors, filters, k, parameters):
    # filters is a CSR matrix where rows = query, cols = required tags
    self.query_parameters = parameters
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
        params.nprobe = self.query_parameters.nprobe

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