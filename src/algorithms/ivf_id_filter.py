from src.algorithms.baseIndex import BaseANNIndex
import faiss
import numpy as np
from src.algorithms.utils import intersect_sorted_lists, build_inverted_attribute_index

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
    self.base_attributes = attributes
    self.build_parameters = parameters

    self.inverted_index = build_inverted_attribute_index(self.base_attributes)

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

        candidate_lists = []

        for dim, val in enumerate(f):
            if val != -1:
                key = (dim, val)
                if key in self.inverted_index:
                    candidate_lists.append(self.inverted_index[key])
                else:
                    candidate_lists = []
                    break

        if candidate_lists:
            valid_ids = intersect_sorted_lists(candidate_lists)
        else:
            valid_ids = list(range(len(self.base_attributes)))

        if not valid_ids:
            continue

        id_array = np.ascontiguousarray(list(valid_ids), dtype='int64')
        selector = faiss.IDSelectorBatch(
            len(id_array),
            faiss.swig_ptr(id_array)
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
        # current_filter is a 2D matrix: [dim][valid_values]
        current_filter = filters[q_idx]
        valid_ids = None

        for dim_idx, valid_vals in enumerate(current_filter):
            dim_union = set()
            for val in valid_vals:
                # Assuming -1 or a specific flag is used for padding if rows aren't equal length
                if val == -1: continue 
                
                key = (dim_idx, val)
                if key in self.inverted_index:
                    dim_union.update(self.inverted_index[key])
            
            if valid_ids is None:
                valid_ids = dim_union
            else:
                valid_ids.intersection_update(dim_union)
            
            if not valid_ids:
                break

        if not valid_ids:
            continue

        id_array = np.ascontiguousarray(list(valid_ids), dtype='int64')
        selector = faiss.IDSelectorBatch(
            len(id_array),
            faiss.swig_ptr(id_array)
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