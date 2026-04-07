from src.benchmark.benchmarkRunner import BenchmarkRunner
from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.algorithms.acorn import Acorn, AcornBuildParameters, AcornQueryParameters
from src.logger import BenchmarkLogger

"""
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

"""


if __name__=="__main__":
    test_sift = True
    test_glove = True

    if test_sift:
        subset_size = 0.1
        k = 10
        
        dataset = siftDataset(subset_size, k)
        print("running IVF on sift dataset")
        for num_restrictions in range(1, 4):
            print("num_restr = ", num_restrictions)
            
            build_params = AcornBuildParameters(M=32, gamma=12, M_beta=32)
            query_params = AcornQueryParameters(efSearch=10)
    
            print("initializing benchmark")
            runner = BenchmarkRunner(dataset, num_restrictions, Acorn, build_params, query_params)
            print("running benchmark")
            D, I, metadata = runner.run()

            print("logging results")
            logger = BenchmarkLogger()
            run_path = logger.log_benchmark(metadata, D, I)
            recall = logger.log_recall(run_path)
            print("recall = ", recall)

    if test_glove:
        subset_size = 0.1
        k = 10

        dataset = GloVeDataset(subset_size, k)
        print("running IVF on glove dataset")
        build_params = AcornBuildParameters(M=32, gamma=12, M_beta=32)
        query_params = AcornQueryParameters(efSearch=10)

        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, num_restrictions, Acorn, build_params, query_params)
        print("running benchmark")
        D, I, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I)
        recall = logger.log_recall(run_path)
        print("recall = ", recall)
        