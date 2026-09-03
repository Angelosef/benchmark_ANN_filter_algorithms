import faiss
import numpy as np
from src.algorithms.baseIndex import BaseANNIndex
from src.algorithms.utils import AttributeIndex

class BruteForceIdFilter(BaseANNIndex):
    def __init__(self, dim, metric):
        super().__init__(dim, metric)
        self.algo_name = "BruteForceIdFilter"
    
    def name(self):
        return self.algo_name

    def save_to_files(self, ds_file, index_file):
        print("saving to files not implemented for brute force")

# --- BUILD STRATEGIES ---

@BruteForceIdFilter.register_build("structured")
def build_structured(self, vectors, attributes, parameters):
    self.attribute_index = AttributeIndex(attributes)
    sub_index = faiss.IndexFlatL2(self.dim)
    self.index = faiss.IndexIDMap(sub_index)
    
    ids = np.arange(len(vectors)).astype('int64')
    self.index.add_with_ids(vectors, ids)

@BruteForceIdFilter.register_build("sparse")
def build_sparse(self, vectors, attributes, parameters):
    self.base_attributes_csc = attributes.tocsc()
    
    sub_index = faiss.IndexFlatL2(self.dim)
    self.index = faiss.IndexIDMap(sub_index)
    
    ids = np.arange(len(vectors)).astype('int64')
    self.index.add_with_ids(vectors, ids)

# --- QUERY STRATEGIES ---
@BruteForceIdFilter.register_init_query("structured", "conjunction")
def init_query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
    return

@BruteForceIdFilter.register_query("structured", "conjunction")
def query_structured_conjunction(self, vector, filter, k):
    valid_ids = self.attribute_index.get_valid_ids_conj(filter)
    sel = faiss.IDSelectorBatch(
        len(valid_ids),
        faiss.swig_ptr(valid_ids)
    )

    params = faiss.SearchParametersIVF(sel=sel)

    dist, indices = self.index.search(
        vector.reshape(1, -1),
        k,
        params=params
    )

    params.sel = None
    del params
    del sel

    return dist[0], indices[0]

@BruteForceIdFilter.register_init_query("structured", "CNF")
def init_query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
    return

@BruteForceIdFilter.register_query("structured", "CNF")
def query_structured_CNF(self, vector, filter, k):
    
    valid_ids = self.attribute_index.get_valid_ids_cnf(filter)
    sel = faiss.IDSelectorBatch(
        len(valid_ids),
        faiss.swig_ptr(valid_ids)
    )

    params = faiss.SearchParametersIVF(sel=sel)

    dist, indices = self.index.search(
        vector.reshape(1, -1),
        k,
        params=params
    )

    params.sel = None
    del params
    del sel

    return dist[0], indices[0]

@BruteForceIdFilter.register_init_query("sparse", "conjunction")
def init_query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
    return

@BruteForceIdFilter.register_query("sparse", "conjunction")
def query_sparse_conjunction(self, vector, filter, k):
    required_tags = filter.indices

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
        return np.full(k, np.inf, dtype='float32'), np.full(k, -1, dtype='int64')
    
    valid_ids = np.array(list(valid_ids_set), dtype='int64')
    params = faiss.SearchParametersIVF()

    sel = faiss.IDSelectorBatch(
        len(valid_ids),
        faiss.swig_ptr(valid_ids)
    )

    params = faiss.SearchParametersIVF(sel=sel)
    
    dist, indices = self.index.search(
        vector.reshape(1, -1),
        k,
        params=params
    )
    
    params.sel = None
    del params
    del sel

    return dist[0], indices[0]

