from src.algorithms.baseIndex import BaseANNIndex
import faiss
import numpy as np
from src.algorithms.utils import valid_structured_conjunction, valid_structured_CNF, valid_csr_conjunction

class HNSWPostfilterBuildParameters:
    def __init__(self, graph_degree=32, efConstruction=200):
        self.graph_degree = graph_degree
        self.efConstruction = efConstruction

class HNSWPostfilterQueryParameters:
    def __init__(self, efSearch, initial_k):
        self.efSearch = efSearch
        self.initial_k = initial_k

class HNSWPostfilter(BaseANNIndex):
    def __init__(self, dim, metric):
        super().__init__(dim, metric)
        self.algo_name = "HNSWPostfilter"
    
    def name(self):
        return self.algo_name

@HNSWPostfilter.register_build("structured")
def build_structured(self, vectors, attributes, parameters):
    self.base_vectors = vectors
    self.base_attributes = attributes
    self.build_parameters = parameters

    self.index = faiss.IndexHNSWFlat(self.dim, self.build_parameters.graph_degree, faiss.METRIC_L2)
    self.index.hnsw.efConstruction = self.build_parameters.efConstruction
    self.index.add(self.base_vectors)
    return

@HNSWPostfilter.register_query("structured", "conjunction")
def query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
    self.index.hnsw.efSearch = self.query_parameters.efSearch

    D, I = self.index.search(vectors, self.query_parameters.initial_k)

    D_updated = np.full((len(vectors), k), np.inf)
    I_updated = np.full((len(vectors), k), -1)

    for q_index, result_set in enumerate(I):
        valid_pairs = []  # (distance, index)
        filter = filters[q_index]

        for i, idx in enumerate(result_set):
            if valid_structured_conjunction(
                self.base_attributes[idx],
                filter
            ):
                valid_pairs.append((D[q_index][i], idx))

        valid_pairs.sort(key=lambda x: x[0])

        num_valid = min(len(valid_pairs), k)

        for j in range(num_valid):
            D_updated[q_index][j] = valid_pairs[j][0]
            I_updated[q_index][j] = valid_pairs[j][1]

    return D_updated, I_updated

@HNSWPostfilter.register_query("structured", "CNF")
def query_structured_CNF(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
    self.index.hnsw.efSearch = self.query_parameters.efSearch

    D, I = self.index.search(vectors, self.query_parameters.initial_k)

    D_updated = np.full((len(vectors), k), np.inf)
    I_updated = np.full((len(vectors), k), -1)

    for q_index, result_set in enumerate(I):
        valid_pairs = []
        filter = filters[q_index]

        for i, idx in enumerate(result_set):
            if valid_structured_CNF(
                self.base_attributes[idx],
                filter
            ):
                valid_pairs.append((D[q_index][i], idx))

        valid_pairs.sort(key=lambda x: x[0])

        num_valid = min(len(valid_pairs), k)

        for j in range(num_valid):
            D_updated[q_index][j] = valid_pairs[j][0]
            I_updated[q_index][j] = valid_pairs[j][1]

    return D_updated, I_updated

@HNSWPostfilter.register_build("sparse")
def build_sparse(self, vectors, attributes, parameters):
    self.base_vectors = vectors
    self.base_attributes = attributes
    self.base_attributes.sort_indices()
    self.build_parameters = parameters

    self.index = faiss.IndexHNSWFlat(self.dim, self.build_parameters.graph_degree, faiss.METRIC_L2)
    self.index.hnsw.efConstruction = self.build_parameters.efConstruction
    self.index.add(self.base_vectors)
    return

@HNSWPostfilter.register_query("sparse", "conjunction")
def query_structured_CNF(self, vectors, filters, k, parameters):
    # base attributes and query filters are supposed to be csr matrices
    filters.sort_indices()
    self.query_parameters = parameters
    self.index.hnsw.efSearch = self.query_parameters.efSearch

    D, I = self.index.search(vectors, self.query_parameters.initial_k)

    D_updated = np.full((len(vectors), k), np.inf)
    I_updated = np.full((len(vectors), k), -1)

    for q_index, result_set in enumerate(I):
        valid_pairs = []
        start = filters.indptr[q_index]
        end = filters.indptr[q_index+1]
        filter_indices = filters.indices[start:end]

        for i, idx in enumerate(result_set):

            if valid_csr_conjunction(
                self.base_attributes,
                idx,
                filter_indices
            ):
                valid_pairs.append((D[q_index][i], idx))

        valid_pairs.sort(key=lambda x: x[0])

        num_valid = min(len(valid_pairs), k)

        for j in range(num_valid):
            D_updated[q_index][j] = valid_pairs[j][0]
            I_updated[q_index][j] = valid_pairs[j][1]

    return D_updated, I_updated

