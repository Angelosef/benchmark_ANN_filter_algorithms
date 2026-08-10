
from src.datasets.yfcc import yfccDataset
from src.datasets.gist import gistDataset
from src.algorithms.ivf_squared_faiss import IVFSquaredFaiss, IVFSquaredFaissBuildParameters, IVFSquaredFaissQueryParameters

from src.benchmark.benchmark_helper import runFullBenchmark, generateLogGrid
import multiprocessing

def bench_yfcc():
    ds = yfccDataset(subset_size=0.1, neighbors_retrieved=10)
    
    target_points = generateLogGrid(5000, 10000, 2)
    efSearch = generateLogGrid(16, 64, 2)
    cut_off = generateLogGrid(10000, 20000, 2)

    build_params = [
        IVFSquaredFaissBuildParameters(cut_off=co, cluster_size=1000, cut_off_tiny=300, cut_off_bitvector=40000, efConstruction=128, M=16)
        for co in cut_off
    ]
    query_params = [
        IVFSquaredFaissQueryParameters(efSearch=ef, target_points=tp)
        for ef in efSearch
        for tp in target_points
    ]
    runFullBenchmark(ds, None, IVFSquaredFaiss, build_params, query_params)


def bench_gist():
    ds = gistDataset(subset_size=1.0, neighbors_retrieved=10)
    
    target_points = generateLogGrid(5000, 10000, 2)
    efSearch = generateLogGrid(16, 64, 2)
    cut_off = generateLogGrid(30000, 30000, 1)

    build_params = [
        IVFSquaredFaissBuildParameters(cut_off=co, cluster_size=1000, cut_off_tiny=300, cut_off_bitvector=40000, efConstruction=128, M=16)
        for co in cut_off
    ]
    query_params = [
        IVFSquaredFaissQueryParameters(efSearch=ef, target_points=tp)
        for ef in efSearch
        for tp in target_points
    ]
    runFullBenchmark(ds, None, IVFSquaredFaiss, build_params, query_params)

if __name__=="__main__":
    test_yfcc = False
    test_gist = True
    
    if test_yfcc:
        p = multiprocessing.Process(
            target=bench_yfcc
        )
        
        p.start()
        p.join()
    
    
    if test_gist:
        p = multiprocessing.Process(
            target=bench_gist
        )
        
        p.start()
        p.join()