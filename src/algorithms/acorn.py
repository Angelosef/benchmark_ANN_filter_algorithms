from ACORN.acorn_class import IndexACORNFlat
import numpy as np
from src.algorithms.baseIndex import BaseANNIndex
from src.algorithms.utils import AttributeIndex, TagAssigner, TagEncoder

class AcornBuildParameters:
    def __init__(self, M=32, gamma=12, M_beta=32, num_bins=None):
        self.M = M
        self.gamma = gamma
        self.M_beta = M_beta
        self.num_bins = num_bins

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
    self.attribute_index = AttributeIndex(attributes)
    self.build_params = parameters
    self.index = IndexACORNFlat(
        self.dim,
        self.build_params.M,
        self.build_params.gamma,
        attributes,
        self.build_params.M_beta
    )
    self.index.add(vectors)

# --- QUERY STRATEGIES ---
@Acorn.register_init_query("structured", "conjunction")
def init_query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_params = parameters
    self.index.efSearch = self.query_params.efSearch
    return


@Acorn.register_query("structured", "conjunction")
def query_structured_conjunction(self, vector, filter, k):
    nb = self.base_vectors.shape[0]
    filter_map = np.zeros((1, nb), dtype='int8')

    valid_ids = self.attribute_index.get_valid_ids_conj(filter)

    filter_map[0][valid_ids] = 1

    D, I = self.index.search(vector.reshape(1, -1), k, filter_map)
    
    return D[0], I[0]


@Acorn.register_init_query("structured", "CNF")
def init_query_structured_CNF(self, vectors, filters, k, parameters):
    self.query_params = parameters
    self.index.efSearch = self.query_params.efSearch
    return

@Acorn.register_query("structured", "CNF")
def query_structured_CNF(self, vector, filter, k):
    nb = self.base_vectors.shape[0]
    
    filter_map = np.zeros((1, nb), dtype='int8')

    valid_ids = self.attribute_index.get_valid_ids_cnf(filter)
    filter_map[0][valid_ids] = 1

    D, I = self.index.search(vector.reshape(1, -1), k, filter_map)
    
    return D[0], I[0]

@Acorn.register_build("sparse")
def build_sparse_translator(self, vectors, attributes, parameters):
    
    tag_assigner = TagAssigner(attributes, parameters.num_bins)
    assignment = tag_assigner.get_assignment()
    self.attribute_encoder = TagEncoder(assignment, parameters.num_bins)
    encoded_attrs = self.attribute_encoder.get_encoded_data(attributes)
    
    return build_structured(self, vectors, encoded_attrs, parameters)

@Acorn.register_init_query("sparse", "conjunction")
def init_query_sparse_conjunction_translator(self, vectors, filters, k, parameters):
    self.query_params = parameters
    self.index.efSearch = self.query_params.efSearch
    return

@Acorn.register_query("sparse", "conjunction")
def query_sparse_conjunction_translator(self, vector, filter, k):
    encoded_filters = self.attribute_encoder.get_encoded_queries(filter)

    return query_structured_conjunction(self, vector, encoded_filters[0], k)
