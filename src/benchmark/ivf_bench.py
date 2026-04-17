from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset
from src.algorithms.ivf_id_filter import IVFIdFilter, IVFIdFilterBuildParameters, IVFIdFilterQueryParameters

from src.benchmark.benchmark_helper import runFullBenchmark, generateLogGrid
import multiprocessing

def bench_sift(nr):
    ds = siftDataset(subset_size=0.1, neighbors_retrieved=10)
    nlist = generateLogGrid(200, 300, 2)
    nprobe = generateLogGrid(10, 100, 4)

    build_params = [
        IVFIdFilterBuildParameters(nlist=nl)
        for nl in nlist
    ]
    query_params = [
        IVFIdFilterQueryParameters(nprobe=np)
        for np in nprobe
    ]

    runFullBenchmark(ds, nr, IVFIdFilter, build_params, query_params)

def bench_glove():
    ds = GloVeDataset(subset_size=0.1, neighbors_retrieved=10)
        
    nlist = generateLogGrid(200, 300, 2)
    nprobe = generateLogGrid(10, 100, 4)

    build_params = [
        IVFIdFilterBuildParameters(nlist=nl)
        for nl in nlist
    ]
    query_params = [
        IVFIdFilterQueryParameters(nprobe=np)
        for np in nprobe
    ]
    runFullBenchmark(ds, None, IVFIdFilter, build_params, query_params)

def bench_yfcc():
    ds = yfccDataset(subset_size=0.1, neighbors_retrieved=10)
        
    nlist = generateLogGrid(200, 300, 2)
    nprobe = generateLogGrid(10, 100, 4)

    build_params = [
        IVFIdFilterBuildParameters(nlist=nl)
        for nl in nlist
    ]
    query_params = [
        IVFIdFilterQueryParameters(nprobe=np)
        for np in nprobe
    ]
    runFullBenchmark(ds, None, IVFIdFilter, build_params, query_params)

if __name__=="__main__":
    test_sift = False
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
