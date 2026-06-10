from src.algorithms.baseIndex import BaseANNIndex
from src.datasets.base_dataset import Dataset
import time
from typing import Type
import os, psutil
import resource

def get_rss():
    return psutil.Process(os.getpid()).memory_info().rss

def get_peak_memory():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss * 1024

class BenchmarkRunner:
    def __init__(self, dataset: Dataset, ds_query_param, index_class: Type[BaseANNIndex], build_params, query_params):
        self.dataset = dataset
        self.ds_query_param = ds_query_param
        self.index_class = index_class
        self.build_params = build_params
        self.query_params = query_params
        self.dim = self.dataset.get_dim()
        self.metric = self.dataset.get_metric()
        self.config = self.dataset.get_config()
        
        self.index = self.index_class(dim=self.dim, metric=self.metric)
        
    def run(self):
        baseline_mem = get_rss()
        initial_peak = get_peak_memory()

        print("start building index")
        start_build = time.time()
        self.index.build(
            self.dataset.get_base_vectors(), 
            self.dataset.get_base_attributes(), 
            self.build_params, 
            self.config
        )
        build_time = time.time() - start_build

        after_build_mem = get_rss()
        index_memory = after_build_mem - baseline_mem
        print(f"Index Memory: {index_memory / 1e6:.2f} MB")

        print("start querying")
        start_query = time.time()
        D, I, L = self.index.query(
            self.dataset.get_query_vectors(),
            self.dataset.get_query_filters(self.ds_query_param),
            self.dataset.get_neighbors_retrieved(),
            self.query_params,
            self.config
        )
        query_time = time.time() - start_query
        final_peak = get_peak_memory()
        peak_memory_overhead = final_peak - baseline_mem

        metadata = {
            "index_name": self.index.name(),
            "dataset_name": self.dataset.get_name(),
            "build_time": build_time,
            "query_time": query_time,
            "index_memory": index_memory,
            "initial_peak": initial_peak,
            "peak_memory": final_peak,
            "peak_memory_overhead": peak_memory_overhead,
            "build_params": vars(self.build_params),
            "query_params": vars(self.query_params),
            "base_count": self.dataset.get_base_count(),
            "query_count": self.dataset.get_query_count(),
            "subset_size": self.dataset.get_subset_size(),
            "neighbors_retrieved": self.dataset.get_neighbors_retrieved(),
            "ds_query_param": self.ds_query_param,
            "dim": self.dim,
            "metric": self.metric
        }
        return D, I, L, metadata
    