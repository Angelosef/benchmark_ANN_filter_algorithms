from src.benchmark.benchmarkRunner import BenchmarkRunner

from src.datasets.yfcc import yfccDataset
from src.datasets.gist import gistDataset
from src.algorithms.ivf_squared_faiss import IVFSquaredFaiss, IVFSquaredFaissBuildParameters, IVFSquaredFaissQueryParameters
from src.logger import BenchmarkLogger

if __name__=="__main__":
    test_yfcc = True
    test_gist = False

    if test_yfcc:
            
        print("running ivf-squared on yfcc dataset")
        
        subset_size = 0.1
        k = 10
    
        dataset = yfccDataset(subset_size, k)
        print("initializing benchmark")
        build_params = IVFSquaredFaissBuildParameters(cut_off=10000, cluster_size=1000, cut_off_tiny=300, cut_off_bitvector=40000, efConstruction=128, M=16)
        query_params = IVFSquaredFaissQueryParameters(efSearch=16, target_points=5000)
        runner = BenchmarkRunner(dataset, None, IVFSquaredFaiss, build_params, query_params)
        print("running benchmark")
        D, I, L, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I, L)
        recall = logger.log_recall(run_path)

        print("recall = ", recall)

    if test_gist:
            
        print("running ivf-squared on gist dataset")
        
        subset_size = 1.0
        k = 10
    
        dataset = gistDataset(subset_size, k)
        print("initializing benchmark")
        build_params = IVFSquaredFaissBuildParameters()
        query_params = IVFSquaredFaissQueryParameters()
        runner = BenchmarkRunner(dataset, None, IVFSquaredFaiss, build_params, query_params)
        print("running benchmark")
        D, I, L, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I, L)
        recall = logger.log_recall(run_path)

        print("recall = ", recall)
        
        