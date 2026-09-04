
from src.datasets.yfcc import yfccDataset
from src.datasets.gist import gistDataset
from src.algorithms.ivf_squared_faiss import IVFSquaredFaiss, IVFSquaredFaissBuildParameters, IVFSquaredFaissQueryParameters

from src.benchmark.benchmark_helper import runFullBenchmark, generateLogGrid
import multiprocessing

def bench_yfcc():
    ds = yfccDataset(subset_size=0.1, neighbors_retrieved=10)

    cut_off = generateLogGrid(5_000, 20_000, 2)
    cut_off_bitvector = generateLogGrid(30_000, 100_000, 2)

    tiny_cutoff = generateLogGrid(1_000, 10_000, 2)
    target_points = generateLogGrid(1_000, 50_000, 3)
    efSearch = generateLogGrid(16, 64, 3)
    
    build_params = [
        IVFSquaredFaissBuildParameters(cut_off=co, cluster_size=1000, cut_off_bitvector=cob, efConstruction=128, M=32)
        for co in cut_off
        for cob in cut_off_bitvector
    ]
    query_params = [
        IVFSquaredFaissQueryParameters(cut_off_tiny=tc, efSearch=ef, target_points=tp)
        for tc in tiny_cutoff
        for ef in efSearch
        for tp in target_points
    ]
    runFullBenchmark(ds, None, IVFSquaredFaiss, build_params, query_params)


def bench_gist():
    ds = gistDataset(subset_size=1.0, neighbors_retrieved=10)
    
    cut_off = generateLogGrid(5_000, 20_000, 2)
    cut_off_bitvector = generateLogGrid(30_000, 100_000, 2)

    tiny_cutoff = generateLogGrid(1_000, 10_000, 2)
    target_points = generateLogGrid(1_000, 50_000, 3)
    efSearch = generateLogGrid(16, 64, 3)
    
    build_params = [
        IVFSquaredFaissBuildParameters(cut_off=co, cluster_size=1000, cut_off_bitvector=cob, efConstruction=128, M=32)
        for co in cut_off
        for cob in cut_off_bitvector
    ]
    query_params = [
        IVFSquaredFaissQueryParameters(cut_off_tiny=tc, efSearch=ef, target_points=tp)
        for tc in tiny_cutoff
        for ef in efSearch
        for tp in target_points
    ]
    runFullBenchmark(ds, None, IVFSquaredFaiss, build_params, query_params)

if __name__=="__main__":
    test_yfcc = True
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