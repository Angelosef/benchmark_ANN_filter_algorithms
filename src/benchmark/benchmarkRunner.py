from src.algorithms.baseIndex import BaseANNIndex
from src.datasets.base_dataset import Dataset
import time
from typing import Type
import os, psutil
import resource
from src.benchmark.save_manager import SaveManager

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
        self.save_manager = SaveManager()
        
        self.index = self.index_class(dim=self.dim, metric=self.metric)
        
    def run(self):
        baseline_mem = get_rss()

        ds_file, index_file = self.save_manager.find_saved_files(
            self.index.name(),
            self.build_params,
            self.dataset.get_name(),
            self.dataset.get_subset_size()
        )
        built_from_file = False
        build_time = None
        index_memory = None
        build_memory_peak = None

        if (not (ds_file is None or index_file is None)):
            print("index is already built - use saved version")
            built_from_file = True
            self.index.build_from_files(
                ds_file,
                index_file,
                self.dataset.get_base_attributes(),
                self.build_params,
                self.config
            )
        else:
            print("building from scratch")
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
            build_peak = get_peak_memory()
            build_memory_peak = build_peak - baseline_mem
            print(f"Index Memory: {index_memory / 1e6:.2f} MB")
            ds_file, index_file = self.save_manager.prepare_folders(
                self.index.name(),
                self.build_params,
                self.dataset.get_name(),
                self.dataset.get_subset_size()
            )
            print("saving index to: ", ds_file, " - ", index_file)
            self.index.save_to_files(ds_file, index_file)

        print("start querying")
        start_query = time.time()
        D, I, L, total_query_time = self.index.query(
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
            "built_from_file": built_from_file,
            "build_time": build_time,
            "query_time": query_time,
            "total_query_time": total_query_time,
            "baseline_memory": baseline_mem,
            "index_memory": index_memory,
            "build_memory_peak": build_memory_peak,
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
    