from src.benchmark.benchmarkRunner import BenchmarkRunner
from src.datasets.sift import siftDataset
from src.algorithms.acorn import Acorn, AcornBuildParameters, AcornQueryParameters
from src.logger import BenchmarkLogger


if __name__=="__main__":
    print("setting parameters")
    subset_size = 0.1
    k = 10
    num_restrictions = 2

    build_params = AcornBuildParameters(M=32, gamma=12, M_beta=32)
    query_params = AcornQueryParameters(efSearch=10)

    dataset = siftDataset(subset_size, k)
    print("initializing benchmark")
    runner = BenchmarkRunner(dataset, num_restrictions, Acorn, build_params, query_params)
    print("running benchmark")
    D, I, metadata = runner.run()

    print("logging results")
    logger = BenchmarkLogger()
    run_path = logger.log_benchmark(metadata, D, I)
    recall = logger.log_recall(run_path)
    print("recall = ", recall)

