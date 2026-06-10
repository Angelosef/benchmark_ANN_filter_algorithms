import multiprocessing
import gc
from src.benchmark.benchmarkRunner import BenchmarkRunner
from src.logger import BenchmarkLogger
import numpy as np

def _benchmark_worker(dataset, ds_query_param, algo_class, b_params, q_params):
    """
    This function runs in a completely separate memory space.
    When it finishes, all RAM it consumed is reclaimed by the OS.
    """
    logger = BenchmarkLogger()
    
    # Initialize the runner and execute
    runner = BenchmarkRunner(dataset, ds_query_param, algo_class, b_params, q_params)
    D, I, L, metadata = runner.run()
    
    # Log results
    run_path = logger.log_benchmark(metadata, D, I, L)
    recall = logger.log_recall(run_path)
    
    print(f"Algorithm: {algo_class.__name__} | "
          f"Recall: {recall:.4f} | Time: {metadata['query_time']:.4f}s")

def runFullBenchmark(dataset, ds_query_param, algo_class, build_params_list, query_params_list):
    for b_params in build_params_list:
        for q_params in query_params_list:
            
            p = multiprocessing.Process(
                target=_benchmark_worker,
                args=(dataset, ds_query_param, algo_class, b_params, q_params)
            )
            
            p.start()
            p.join()
            
            gc.collect()

def generateLogGrid(start, end, num_samples):
        """Generates a unique, sorted list of native Python integers."""
        if start == end or num_samples <= 1:
            return [int(start)]
        
        values = np.geomspace(start, end, num=num_samples).astype(int).tolist()
        return sorted(list(set(values)))