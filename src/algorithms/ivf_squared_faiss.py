from src.algorithms.baseIndex import BaseANNIndex
import faiss
import numpy as np

class IVFSquaredFaissBuildParameters:
    def __init__(self, cut_off=1000, cluster_size=16, cut_off_tiny=100, cut_off_bitvector=10000, efConstruction=128, M=16):
        self.cut_off = cut_off
        self.cluster_size = cluster_size
        self.cut_off_tiny = cut_off_tiny
        self.cut_off_bitvector = cut_off_bitvector
        self.efConstruction = efConstruction
        self.M = M
        
        
class IVFSquaredFaissQueryParameters:
    def __init__(self, efSearch=16, target_points=1000):
        self.efSearch = efSearch
        self.target_points = target_points
        
class IVFSquaredFaiss(BaseANNIndex):
    def __init__(self, dim, metric):
        super().__init__(dim, metric)
        self.algo_name = "IVFSquaredFaiss"

    def name(self):
        return self.algo_name

@IVFSquaredFaiss.register_build("sparse")
def build_sparse_ivf_squared(self, vectors, attributes, parameters):    
    self.build_parameters = parameters
    

    self.index = faiss.IndexIVFSquared(
        vectors.shape[1], 
        cut_off=parameters.cut_off, 
        cluster_size=parameters.cluster_size, 
        cut_off_tiny=parameters.cut_off_tiny, 
        cut_off_bitvector=parameters.cut_off_bitvector,
        efConstruction=parameters.efConstruction,
        M=parameters.M
    )

    attributes_csc = attributes.tocsc()

    self.index.add_with_tags(vectors, attributes_csc.indices, attributes_csc.indptr)    
    return

@IVFSquaredFaiss.register_init_query("sparse", "conjunction")
def init_query_sparse_ivf_squared(self, vectors, filters, k, parameters):
    self.query_params = parameters
    
@IVFSquaredFaiss.register_query("sparse", "conjunction")
def query_sparse_ivf_squared(self, vector, filter, k):
    required_tags = filter.indices
    if len(required_tags) > 2:
        raise ValueError(f"Query  has {len(required_tags)} tags. ParlayANN supports 1 or 2.")
    query_tags = -1 * np.ones(2, dtype=np.int64)
    for i, tag in enumerate(required_tags):
        query_tags[i] = tag
    search_params = faiss.SearchParametersIVFSquared()
    search_params.efSearch = self.query_params.efSearch
    search_params.target_points = self.query_params.target_points
    search_params.set_query_tags(query_tags)

    distances, labels = self.index.search(vector.reshape(1, -1), k=k, params=search_params)
    return distances[0], labels[0]
