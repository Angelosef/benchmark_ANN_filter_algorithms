from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset
from src.algorithms.hnsw_postfilter import HNSWPostfilter, HNSWPostfilterBuildParameters, HNSWPostfilterQueryParameters
from src.benchmark.benchmark_helper import runFullBenchmark, generateLogGrid
import multiprocessing

def bench_sift(nr):
    ds = siftDataset(subset_size=1.0, neighbors_retrieved=10)

    initial_k = generateLogGrid(50, 1000, 3)
    efSearch = generateLogGrid(10, 200, 3)

    build_params = [HNSWPostfilterBuildParameters(graph_degree=32, efConstruction=200)]
    query_params = [
        HNSWPostfilterQueryParameters(efSearch=ef, initial_k=ik)
        for ef in efSearch
        for ik in initial_k
    ]

    runFullBenchmark(ds, nr, HNSWPostfilter, build_params, query_params)

def bench_glove():
    ds = GloVeDataset(subset_size=1.0, neighbors_retrieved=10)

    initial_k = generateLogGrid(50, 1000, 3)
    efSearch = generateLogGrid(10, 200, 3)

    build_params = [HNSWPostfilterBuildParameters(graph_degree=32, efConstruction=200)]
    query_params = [
        HNSWPostfilterQueryParameters(efSearch=ef, initial_k=ik)
        for ef in efSearch
        for ik in initial_k
    ]
    
    runFullBenchmark(ds, None, HNSWPostfilter, build_params, query_params)

def bench_yfcc():
    ds = yfccDataset(subset_size=0.1, neighbors_retrieved=10)

    initial_k = generateLogGrid(50, 1000, 3)
    efSearch = generateLogGrid(10, 200, 3)

    build_params = [HNSWPostfilterBuildParameters(graph_degree=32, efConstruction=200)]
    query_params = [
        HNSWPostfilterQueryParameters(efSearch=ef, initial_k=ik)
        for ef in efSearch
        for ik in initial_k
    ]
    
    runFullBenchmark(ds, None, HNSWPostfilter, build_params, query_params)

if __name__=="__main__":
    test_sift = True
    test_glove = True
    test_yfcc = True

    if test_sift:
        for nr in range(1, 4):
            p = multiprocessing.Process(
                target=bench_sift,
                args=(nr,)
            )
            
            p.start()
            p.join()
    
    if test_glove:
        p = multiprocessing.Process(
            target=bench_glove
        )
            
        p.start()
        p.join()
    
    if test_yfcc:
        p = multiprocessing.Process(
            target=bench_yfcc
        )
        
        p.start()
        p.join()
