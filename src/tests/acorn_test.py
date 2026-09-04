from src.benchmark.benchmarkRunner import BenchmarkRunner
from src.datasets.all_datasets import *
from src.algorithms.acorn import Acorn, AcornBuildParameters, AcornQueryParameters
from src.logger import BenchmarkLogger


if __name__=="__main__":
    test_sift = False
    test_glove = True
    test_yfcc = False
    test_gist = False

    if test_sift:
        subset_size = 0.1
        k = 10
        
        dataset = siftDataset(subset_size, k)
        print("running Acorn on sift dataset")
        for num_restrictions in range(1, 4):
            print("num_restr = ", num_restrictions)
            
            build_params = AcornBuildParameters(M=32, gamma=36, M_beta=32*4, efConstruction=256)
            query_params = AcornQueryParameters(efSearch=32)
    
            print("initializing benchmark")
            runner = BenchmarkRunner(dataset, num_restrictions, Acorn, build_params, query_params)
            print("running benchmark")
            D, I, L, metadata = runner.run()

            print("logging results")
            logger = BenchmarkLogger()
            run_path = logger.log_benchmark(metadata, D, I, L)
            recall = logger.log_recall(run_path)
            print("recall = ", recall)

    if test_glove:
        subset_size = 0.1
        k = 10

        dataset = GloVeDataset(subset_size, k)
        print("running Acorn on glove dataset")
        build_params = AcornBuildParameters(M=32, gamma=28, M_beta=200, efConstruction=300)
        query_params = AcornQueryParameters(efSearch=64)

        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, None, Acorn, build_params, query_params)
        print("running benchmark")
        D, I, L, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I, L)
        recall = logger.log_recall(run_path)
        print("recall = ", recall)
        
    if test_yfcc:
        subset_size = 0.01
        k = 10

        dataset = yfccDataset(subset_size, k)
        dataset.prepare()
        print("running Acorn on yfcc dataset")
        build_params = AcornBuildParameters(M=32, gamma=12, M_beta=32, efConstruction=128)
        query_params = AcornQueryParameters(efSearch=46)

        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, None, Acorn, build_params, query_params)
        print("running benchmark")
        D, I, L, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I, L)
        recall = logger.log_recall(run_path)
        print("recall = ", recall)

    if test_gist:
            subset_size = 1.0
            k = 10
    
            dataset = gistDataset(subset_size, k)
            print("running Acorn on gist dataset")
            build_params = AcornBuildParameters(M=16, gamma=12, M_beta=32, efConstruction=16)
            query_params = AcornQueryParameters(efSearch=32)
    
            print("initializing benchmark")
            runner = BenchmarkRunner(dataset, None, Acorn, build_params, query_params)
            print("running benchmark")
            D, I, L, metadata = runner.run()
    
            print("logging results")
            logger = BenchmarkLogger()
            run_path = logger.log_benchmark(metadata, D, I, L)
            recall = logger.log_recall(run_path)
            print("recall = ", recall)