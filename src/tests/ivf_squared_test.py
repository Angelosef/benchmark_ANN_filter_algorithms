from src.benchmark.benchmarkRunner import BenchmarkRunner

from src.datasets.yfcc import yfccDataset
from src.algorithms.ivf_squared import IVFSquared, IVFSquaredBuildParameters, IVFSquaredQueryParameters
from src.logger import BenchmarkLogger

if __name__=="__main__":
    test_yfcc = True

    if test_yfcc:
            
            print("running ivf-squared on yfcc dataset")
            
            subset_size = 0.1
            k = 10
        
            dataset = yfccDataset(subset_size, k)
            print("initializing benchmark")
            build_params = IVFSquaredBuildParameters()
            query_params = IVFSquaredQueryParameters()
            runner = BenchmarkRunner(dataset, None, IVFSquared, build_params, query_params)
            print("running benchmark")
            D, I, metadata = runner.run()

            print("logging results")
            logger = BenchmarkLogger()
            run_path = logger.log_benchmark(metadata, D, I)
            recall = logger.log_recall(run_path)

            print("recall = ", recall)