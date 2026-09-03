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

    def save_to_files(self, ds_file, index_file):
        if ds_file is None:
            ds_file = ""
        self.index.writeToFile(index_file, ds_file)

@HNSWPostfilter.register_build("structured")
def build_structured(self, vectors, attributes, parameters):
    self.base_attributes = attributes
    self.build_parameters = parameters

    self.index = faiss.IndexHNSWFlat(self.dim, self.build_parameters.graph_degree, faiss.METRIC_L2)
    self.index.hnsw.efConstruction = self.build_parameters.efConstruction
    self.index.add(vectors)
    return

@HNSWPostfilter.register_build_from_files("structured")
def build_from_files_structured(self, ds_file, index_file, attributes, parameters):
    self.base_attributes = attributes
    self.build_parameters = parameters

    self.index = faiss.IndexHNSWFlat(index_file, ds_file)

    return

@HNSWPostfilter.register_init_query("structured", "conjunction")
def init_query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
    self.index.hnsw.efSearch = self.query_parameters.efSearch
    return

@HNSWPostfilter.register_query("structured", "conjunction")
def query_structured_conjunction(self, vector, filter, k):
    
    D, I = self.index.search(vector.reshape(1, -1), self.query_parameters.initial_k)
    dist = D[0]
    indices = I[0]

    D_updated = np.full(k, np.inf)
    I_updated = np.full(k, -1)

    valid_pairs = []
    for i, idx in enumerate(indices):
        if valid_structured_conjunction(
            self.base_attributes[idx],
            filter
        ):
            valid_pairs.append((dist[i], idx))

    num_valid = min(len(valid_pairs), k)

    for i in range(num_valid):
        D_updated[i] = valid_pairs[i][0]
        I_updated[i] = valid_pairs[i][1]

    return D_updated, I_updated

@HNSWPostfilter.register_init_query("structured", "CNF")
def init_query_structured_CNF(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
    self.index.hnsw.efSearch = self.query_parameters.efSearch
    return

@HNSWPostfilter.register_query("structured", "CNF")
def query_structured_CNF(self, vector, filter, k):
    D, I = self.index.search(vector.reshape(1, -1), self.query_parameters.initial_k)
    dist = D[0]
    indices = I[0]

    D_updated = np.full(k, np.inf)
    I_updated = np.full(k, -1)

    valid_pairs = []
    
    for i, idx in enumerate(indices):
        if valid_structured_CNF(
            self.base_attributes[idx],
            filter
        ):
            valid_pairs.append((dist[i], idx))

    num_valid = min(len(valid_pairs), k)
    for i in range(num_valid):
        D_updated[i] = valid_pairs[i][0]
        I_updated[i] = valid_pairs[i][1]

    return D_updated, I_updated

@HNSWPostfilter.register_build("sparse")
def build_sparse(self, vectors, attributes, parameters):
    self.base_attributes = attributes
    self.base_attributes.sort_indices()
    self.build_parameters = parameters

    self.index = faiss.IndexHNSWFlat(self.dim, self.build_parameters.graph_degree, faiss.METRIC_L2)
    self.index.hnsw.efConstruction = self.build_parameters.efConstruction
    self.index.add(vectors)
    return

@HNSWPostfilter.register_build_from_files("sparse")
def build_from_files_sparse(self, ds_file, index_file, attributes, parameters):
    self.base_attributes = attributes
    self.base_attributes.sort_indices()
    self.build_parameters = parameters

    self.index = faiss.IndexHNSWFlat(index_file, ds_file)
    return

@HNSWPostfilter.register_init_query("sparse", "conjunction")
def init_query_sparse_conjunction(self, vectors, filters, k, parameters):
    self.query_parameters = parameters
    self.index.hnsw.efSearch = self.query_parameters.efSearch
    return

@HNSWPostfilter.register_query("sparse", "conjunction")
def query_structured_CNF(self, vector, filter, k):
        
    D, I = self.index.search(vector.reshape(1, -1), self.query_parameters.initial_k)
    dist = D[0]
    indices = I[0]

    D_updated = np.full(k, np.inf)
    I_updated = np.full(k, -1)

    valid_pairs = []
    filter_indices = filter.indices

    for i, idx in enumerate(indices):

        if valid_csr_conjunction(
            self.base_attributes,
            idx,
            filter_indices
        ):
            valid_pairs.append((dist[i], idx))

    num_valid = min(len(valid_pairs), k)
    for i in range(num_valid):
        D_updated[i] = valid_pairs[i][0]
        I_updated[i] = valid_pairs[i][1]

    return D_updated, I_updated


