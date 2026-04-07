from constrainedANN.caps import CapsIndex

import numpy as np
from src.algorithms.baseIndex import BaseANNIndex
from src.algorithms.utils import HybridBloomEncoder
from datetime import datetime
import os


class CAPSBuildParameters:
    def __init__(self, nc=100, m_bits=None, n_head=None, k_tail=None):
        self.nc = nc
        # used for sparse build
        self.m_bits = m_bits
        self.n_head = n_head
        self.k_tail = k_tail

class CAPSQueryParameters:
    def __init__(self, nprobe=10):
        self.nprobe = nprobe

class CAPS(BaseANNIndex):
    def __init__(self, dim, metric):
        super().__init__(dim, metric)
        self.algo_name = "CAPS"
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")

        folder_name = f"index_{timestamp}"
        self.index_path = os.path.join("my_indexes", folder_name)
    
    def name(self):
        return self.algo_name

@CAPS.register_build("structured")
def build_structured(self, vectors, attributes, parameters):
    self.build_params = parameters
    # convert attributes from int to strings
    num_cols = attributes.shape[1]
    prefixes = np.char.add(np.arange(num_cols).astype(str), "_")
    attr_str = attributes.astype(str)
    props = np.char.add(prefixes, attr_str)
    self.index = CapsIndex(base_vecs=vectors, props=props, nc=self.build_params.nc)
    
    self.index.save_index(self.index_path)
    self.index.load_index()

@CAPS.register_query("structured", "conjunction")
def query_structured_conjunction(self, vectors, filters, k, parameters):
    self.query_params = parameters
    
    num_cols = filters.shape[1]
    prefixes = np.char.add(np.arange(num_cols).astype(str), "_")

    q_labels = np.char.add(prefixes, filters.astype(str))

    mask = (filters == -1)
    q_labels[mask] = "X"

    D, I = self.index.query(vectors, q_labels, k, n_probe=self.query_params.nprobe)

    return D, I

@CAPS.register_build("sparse")
def build_sparse_translator(self, vectors, attributes, parameters):
    self.attribute_encoder = HybridBloomEncoder(parameters.m_bits, parameters.n_head, parameters.k_tail)
    encoded_attrs = self.attribute_encoder.encode_csr(attributes)
    build_fn = self.build_strategies["structured"] 
    return build_fn(self, vectors, encoded_attrs, parameters)

@CAPS.register_query("sparse", "conjunction")
def query_sparse_conjunction_translator(self, vectors, filters, k, parameters):
    nq = vectors.shape[0]
    encoded_filters = np.full((nq, self.attribute_encoder.m_bits), -1, dtype=np.int8)

    for q_idx in range(nq):
        f_start = filters.indptr[q_idx]
        f_end = filters.indptr[q_idx+1]
        required_tags = filters.indices[f_start:f_end]
        target_bits = self.attribute_encoder.create_query_indices(required_tags)
        encoded_filters[q_idx][target_bits] = 1

    query_fn = self.query_strategies[("structured", "conjunction")] 
    return query_fn(self, vectors, encoded_filters, k, parameters)
