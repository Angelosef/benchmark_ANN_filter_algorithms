from src.datasets.all_datasets import *
from src.algorithms.acorn import Acorn, AcornBuildParameters, AcornQueryParameters

from src.benchmark.benchmark_helper import runFullBenchmark, generateLogGrid
import multiprocessing

def bench_sift(nr):
    ds = siftDataset(subset_size=1.0, neighbors_retrieved=10)

    build_params = [AcornBuildParameters(M=16, gamma=12, M_beta=32, efConstruction=16)]
    gamma = generateLogGrid(36, 36, 1)
    efSearch = generateLogGrid(12, 64, 3)
    if nr==1:
        gamma = generateLogGrid(12, 12, 1)
    if nr==3:
        efSearch = generateLogGrid(10, 10, 1)

    build_params = [
        AcornBuildParameters(M=16, gamma=g, M_beta=32, efConstruction=200)
        for g in gamma
    ]
    query_params = [
        AcornQueryParameters(efSearch=ef)
        for ef in efSearch
    ]

    runFullBenchmark(ds, nr, Acorn, build_params, query_params)

def bench_glove():
    ds = GloVeDataset(subset_size=1.0, neighbors_retrieved=10)
    
    build_params = [AcornBuildParameters(M=16, gamma=12, M_beta=32, efConstruction=16)]
    gamma = generateLogGrid(32, 32, 1)
    efSearch = generateLogGrid(12, 64, 3)

    build_params = [
        AcornBuildParameters(M=16, gamma=g, M_beta=32, efConstruction=200)
        for g in gamma
    ]
    query_params = [
        AcornQueryParameters(efSearch=ef)
        for ef in efSearch
    ]
    runFullBenchmark(ds, None, Acorn, build_params, query_params)
    
def bench_yfcc():
    ds = yfccDataset(subset_size=0.1, neighbors_retrieved=10)
        
    build_params = [AcornBuildParameters(M=16, gamma=12, M_beta=32, efConstruction=16)]
    gamma = generateLogGrid(32, 32, 1)
    efSearch = generateLogGrid(12, 64, 3)

    build_params = [
        AcornBuildParameters(M=16, gamma=g, M_beta=32, efConstruction=200)
        for g in gamma
    ]
    query_params = [
        AcornQueryParameters(efSearch=ef)
        for ef in efSearch
    ]
    runFullBenchmark(ds, None, Acorn, build_params, query_params)

def bench_gist():
    ds = gistDataset(subset_size=1.0, neighbors_retrieved=10)
        
    build_params = [AcornBuildParameters(M=16, gamma=12, M_beta=32, efConstruction=16)]
    gamma = generateLogGrid(12, 32, 2)
    efSearch = generateLogGrid(12, 64, 3)

    build_params = [
        AcornBuildParameters(M=16, gamma=g, M_beta=32, efConstruction=200)
        for g in gamma
    ]
    query_params = [
        AcornQueryParameters(efSearch=ef)
        for ef in efSearch
    ]
    runFullBenchmark(ds, None, Acorn, build_params, query_params)
    


if __name__=="__main__":
    test_sift = False
    test_glove = False
    test_yfcc = True
    test_gist = False

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

    if test_gist:
        p = multiprocessing.Process(
            target=bench_gist
        )
        
        p.start()
        p.join()

