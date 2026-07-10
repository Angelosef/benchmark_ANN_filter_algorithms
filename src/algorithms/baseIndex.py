from abc import ABC, abstractmethod
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
import time

class BaseANNIndex(ABC):
    def __init__(self, dim: int, metric: str):
        self.dim = dim
        self.metric = metric
        self.rng = np.random.default_rng(seed=42)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.build_strategies = {}
        cls.query_strategies = {}
        cls.init_query_strategies = {}

    @classmethod
    def register_build(cls, attribute_type):
        def decorator(func):
            cls.build_strategies[attribute_type] = func
            return func
        return decorator
    
    @classmethod
    def register_query(cls, attribute_type, query_type):
        def decorator(func):
            cls.query_strategies[(attribute_type, query_type)] = func
            return func
        return decorator
    
    @classmethod
    def register_init_query(cls, attribute_type, query_type):
        def decorator(func):
            cls.init_query_strategies[(attribute_type, query_type)] = func
            return func
        return decorator

    def build(self, vectors, attributes, parameters, config):
        self.attribute_type = config.attribute_type
        if self.attribute_type not in self.build_strategies:
            raise ValueError(f"Unsupported attribute type: {self.attribute_type}")
        
        fn = self.build_strategies[self.attribute_type]
        return fn(self, vectors, attributes, parameters)
    
    def timed_single_query(self, vector, filter, k, single_query_fn):
        start_time = time.perf_counter()
        D, I = single_query_fn(self, vector, filter, k)

        latency = time.perf_counter() - start_time

        return D, I, latency

    def query(self, vectors, filters, k, parameters, config):
        if config.attribute_type != self.attribute_type:
            raise ValueError("Mismatch between build and query attribute type")

        key = (config.attribute_type, config.query_type)
        if key not in self.query_strategies:
            raise ValueError(f"Unsupported query type: {key}")

        init_query_fn = self.init_query_strategies[key]
        init_query_fn(self, vectors, filters, k, parameters)
        
        single_query_fn = self.query_strategies[key]
        D = np.full((len(vectors), k), np.inf)
        I = np.full((len(vectors), k), -1)
        L = np.zeros(len(vectors))
        
        num_threads = os.cpu_count() or 1

        num_warm_up = min(100, len(vectors))
        warm_up_indices = self.rng.choice(np.arange(len(vectors)), size=num_warm_up, replace=False)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            unused_futures = [
                executor.submit(self.timed_single_query, vectors[i], filters[i], k, single_query_fn) 
                for i in warm_up_indices
            ]
        
        start_time = time.perf_counter()

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(self.timed_single_query, vectors[i], filters[i], k, single_query_fn) 
                for i in range(len(vectors))
            ]
            
            results = [future.result() for future in futures]

        for q_idx, (dist, indices, latency) in enumerate(results):
            D[q_idx] = dist
            I[q_idx] = indices
            L[q_idx] = latency
        
        total_query_time = time.perf_counter() - start_time

        return D, I, L, total_query_time

    @abstractmethod
    def name(self) -> str:
        pass