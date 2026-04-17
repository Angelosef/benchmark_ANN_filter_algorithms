from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset
from src.algorithms.acorn import Acorn, AcornBuildParameters, AcornQueryParameters

from src.benchmark.benchmark_helper import runFullBenchmark, generateLogGrid
import multiprocessing

def bench_sift(nr):
    ds = siftDataset(subset_size=0.1, neighbors_retrieved=10)

    build_params = [AcornBuildParameters(M=32, gamma=24, M_beta=32)]
    efSearch = generateLogGrid(10, 500, 6)

    query_params = [
        AcornQueryParameters(efSearch=ef)
        for ef in efSearch
    ]

    runFullBenchmark(ds, nr, Acorn, build_params, query_params)

def bench_glove():
    ds = GloVeDataset(subset_size=0.1, neighbors_retrieved=10)
    
    build_params = [AcornBuildParameters(M=32, gamma=24, M_beta=32)]
    efSearch = generateLogGrid(10, 500, 6)

    query_params = [
        AcornQueryParameters(efSearch=ef)
        for ef in efSearch
    ]
    runFullBenchmark(ds, None, Acorn, build_params, query_params)
    
def bench_yfcc():
    ds = yfccDataset(subset_size=0.1, neighbors_retrieved=10)
        
    build_params = [AcornBuildParameters(M=32, gamma=24, M_beta=32)]
    efSearch = generateLogGrid(10, 500, 6)

    query_params = [
        AcornQueryParameters(efSearch=ef)
        for ef in efSearch
    ]
    runFullBenchmark(ds, None, Acorn, build_params, query_params)


if __name__=="__main__":
    test_sift = True
    test_glove = True
    test_yfcc = False

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
