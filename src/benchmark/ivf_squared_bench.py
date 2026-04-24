
from src.datasets.yfcc import yfccDataset
from src.datasets.gist import gistDataset
from src.algorithms.ivf_squared import IVFSquared, IVFSquaredBuildParameters, IVFSquaredQueryParameters

from src.benchmark.benchmark_helper import runFullBenchmark, generateLogGrid
import multiprocessing

def bench_yfcc():
    ds = yfccDataset(subset_size=0.1, neighbors_retrieved=10)
    
    target_points = generateLogGrid(5000, 10000, 2)
    tiny_cutoff = generateLogGrid(500, 1000, 2)
    beam_width = generateLogGrid(40, 80, 2)

    build_params = [
        IVFSquaredBuildParameters(beam_widths=(bw, bw, bw))
        for bw in beam_width
    ]
    query_params = [
        IVFSquaredQueryParameters(tiny_cutoff=tc, target_points=tp)
        for tc in tiny_cutoff
        for tp in target_points
    ]
    runFullBenchmark(ds, None, IVFSquared, build_params, query_params)


def bench_gist():
    ds = gistDataset(subset_size=1.0, neighbors_retrieved=10)
    
    target_points = generateLogGrid(5000, 10000, 2)
    tiny_cutoff = generateLogGrid(500, 1000, 2)
    beam_width = generateLogGrid(40, 80, 2)

    build_params = [
        IVFSquaredBuildParameters(beam_widths=(bw, bw, bw))
        for bw in beam_width
    ]
    query_params = [
        IVFSquaredQueryParameters(tiny_cutoff=tc, target_points=tp)
        for tc in tiny_cutoff
        for tp in target_points
    ]
    runFullBenchmark(ds, None, IVFSquared, build_params, query_params)

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