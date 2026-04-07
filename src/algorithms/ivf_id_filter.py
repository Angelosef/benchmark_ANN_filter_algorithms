from src.algorithms.baseIndex import BaseANNIndex
import faiss
import numpy as np
from src.algorithms.utils import AttributeIndex

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
    self.base_vectors = vectors
    self.build_parameters = parameters

    self.attribute_index = AttributeIndex(attributes)

    quantizer = faiss.IndexFlatL2(self.dim)
    self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.build_parameters.nlist)
    train_size = int(len(self.base_vectors) * 0.1)
    self.index.train(self.base_vectors[:train_size])
    self.index.add(self.base_vectors)
    return

@IVFIdFilter.register_query("structured", "conjunction")
def query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_parameters = parameters

    D = np.full((len(vectors), k), np.inf)
    I = np.full((len(vectors), k), -1)

    for i, q in enumerate(vectors):
        f = filters[i]
        valid_ids = self.attribute_index.get_valid_ids_conj(f)
        
        selector = faiss.IDSelectorBatch(
            len(valid_ids),
            faiss.swig_ptr(valid_ids)
        )

        params = faiss.SearchParametersIVF()
        params.sel = selector
        params.nprobe = self.query_parameters.nprobe

        dist, indices = self.index.search(np.array([q]), k, params=params)

        D[i] = dist
        I[i] = indices

    return D, I


@IVFIdFilter.register_query("structured", "CNF")
def query_structured_CNF(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
    D_updated = np.full((len(vectors), k), np.inf, dtype='float32')
    I_updated = np.full((len(vectors), k), -1, dtype='int64')

    for q_idx, q_vec in enumerate(vectors):
        current_filter = filters[q_idx]
        valid_ids = self.attribute_index.get_valid_ids_cnf(current_filter)
        selector = faiss.IDSelectorBatch(
            len(valid_ids),
            faiss.swig_ptr(valid_ids)
        )

        params = faiss.SearchParametersIVF()
        params.sel = selector
        params.nprobe = self.query_parameters.nprobe

        dist, indices = self.index.search(q_vec.reshape(1, -1), k, params=params)

        D_updated[q_idx] = dist[0]
        I_updated[q_idx] = indices[0]

    return D_updated, I_updated

@IVFIdFilter.register_build("sparse")
def build_sparse(self, vectors, attributes, parameters):
    self.base_vectors = vectors
    self.base_attributes_csc = attributes.tocsc()
    self.build_parameters = parameters

    quantizer = faiss.IndexFlatL2(self.dim)
    self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.build_parameters.nlist)
    train_size = int(len(self.base_vectors) * 0.1)
    self.index.train(self.base_vectors[:train_size])
    self.index.add(self.base_vectors)
    return

@IVFIdFilter.register_query("sparse", "conjunction")
def query_sparse_conjunction(self, vectors, filters, k, parameters):
    # filters is a CSR matrix where rows = query, cols = required tags
    self.query_parameters = parameters
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
                
                if not valid_ids_set:
                    break

        if valid_ids_set is not None and not valid_ids_set:
            continue

        params = faiss.SearchParametersIVF()
        params.nprobe = self.query_parameters.nprobe

        if valid_ids_set is not None:
            id_array = np.array(list(valid_ids_set), dtype='int64')
            params.sel = faiss.IDSelectorBatch(len(id_array), faiss.swig_ptr(id_array))
        else:
            params.sel = None 

        dist, indices = self.index.search(q_vec.reshape(1, -1), k, params=params)

        D_updated[q_idx] = dist[0]
        I_updated[q_idx] = indices[0]

    return D_updated, I_updated