from src.algorithms.baseIndex import BaseANNIndex
from datetime import datetime

import ParlayANN.python.wrapper as wp
import os
from src.algorithms.utils import save_fbin, save_csr_to_spmat
import shutil

class IVFSquaredBuildParameters:
    def __init__(self, weight_classes=(100_000, 400_000), max_degrees=(8, 10, 12), 
                 cutoff=20_000, cluster_size=2500, alpha=1.175, beam_widths=(85, 85, 85)):
        self.weight_classes = weight_classes
        self.max_degrees = max_degrees
        self.cutoff = cutoff
        self.cluster_size = cluster_size
        self.alpha = alpha
        self.beam_widths = beam_widths
        
        
class IVFSquaredQueryParameters:
    def __init__(self, tiny_cutoff=500, target_points=15_000):
        
        self.tiny_cutoff = tiny_cutoff
        self.target_points = target_points
        
        
class IVFSquared(BaseANNIndex):
    def __init__(self, dim, metric):
        super().__init__(dim, metric)
        self.algo_name = "IVFSquared"
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        self.index_path = "my_indexes/index_cache/"
        self.temp_data_path = os.path.join("temp_data", f"data_{timestamp}")
        if os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
        if os.path.exists(self.temp_data_path):
            shutil.rmtree(self.temp_data_path)
    
        os.makedirs(self.index_path)
        os.makedirs(self.temp_data_path)
    
    def name(self):
        return self.algo_name

@IVFSquared.register_build("sparse")
def build_sparse_ivf_squared(self, vectors, attributes, parameters):

    self.base_file = os.path.join(self.temp_data_path, "base.fbin")
    self.meta_file = os.path.join(self.temp_data_path, "metadata.spmat")
    
    self.build_parameters = parameters
    
    
    # Save the actual data passed in
    save_fbin(vectors, self.base_file)
    save_csr_to_spmat(attributes, self.meta_file)

    self.index = wp.init_squared_ivf_index("Euclidian", "float")

    for i in range(3):
        # Using parameters.max_degrees, alpha, and beam_widths from the config object
        b_param = wp.BuildParams(
            parameters.max_degrees[i], 
            500, 
            parameters.alpha
        )
        q_param = wp.QueryParams(
            10, 
            parameters.beam_widths[i], 
            1.35, 
            1000, 
            parameters.max_degrees[i]
        )
        
        self.index.set_build_params(b_param, i)
        self.index.set_query_params(q_param, i)

    self.index.fit_from_filename(
        self.base_file,
        self.meta_file, 
        parameters.cutoff, 
        parameters.cluster_size, 
        self.index_path, 
        parameters.weight_classes,
        False
    )
    
    return

@IVFSquared.register_init_query("sparse", "conjunction")
def init_query_sparse_ivf_squared(self, vectors, filters, k, parameters):
    self.query_parametrs = parameters
    self.index.set_target_points(parameters.target_points)
    self.index.set_tiny_cutoff(parameters.tiny_cutoff)    
    
@IVFSquared.register_query("sparse", "conjunction")
def query_sparse_ivf_squared(self, vector, filter, k):
    required_tags = filter.indices
    if len(required_tags) == 1:
        q_filter = wp.QueryFilter(required_tags[0])
    elif len(required_tags) == 2:
        q_filter = wp.QueryFilter(required_tags[0], required_tags[1])
    else:
        raise ValueError(f"Query  has {len(required_tags)} tags. ParlayANN supports 1 or 2.")
    
    neighbors, distances = self.index.batch_filter_search(vector.reshape(1, -1), [q_filter], 1, k)
    
    return distances[0], neighbors[0]
