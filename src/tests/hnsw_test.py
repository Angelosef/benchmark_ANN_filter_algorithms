from src.benchmark.benchmarkRunner import BenchmarkRunner
from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset
from src.algorithms.hnsw_postfilter import HNSWPostfilter, HNSWPostfilterBuildParameters, HNSWPostfilterQueryParameters
from src.logger import BenchmarkLogger

if __name__=="__main__":
    test_sift = True
    test_glove = True
    test_yfcc = True

    if test_sift:
        subset_size = 0.1
        k = 10
        
        dataset = siftDataset(subset_size, k)
        print("running IVF on sift dataset")
        for num_restrictions in range(1, 4):
            print("num_restr = ", num_restrictions)
            
            build_params = HNSWPostfilterBuildParameters(graph_degree=32, efConstruction=200)
            query_params = HNSWPostfilterQueryParameters(efSearch=100, initial_k=200)
    
            print("initializing benchmark")
            runner = BenchmarkRunner(dataset, num_restrictions, HNSWPostfilter, build_params, query_params)
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
        build_params = HNSWPostfilterBuildParameters(graph_degree=32, efConstruction=200)
        query_params = HNSWPostfilterQueryParameters(efSearch=100, initial_k=200)

        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, num_restrictions, HNSWPostfilter, build_params, query_params)
        print("running benchmark")
        D, I, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I)
        recall = logger.log_recall(run_path)
        print("recall = ", recall)

    if test_yfcc:
        subset_size = 0.1
        k = 10

        dataset = yfccDataset(subset_size, k)
        print("running IVF on yfcc dataset")
        build_params = HNSWPostfilterBuildParameters(graph_degree=32, efConstruction=200)
        query_params = HNSWPostfilterQueryParameters(efSearch=100, initial_k=200)
        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, num_restrictions, HNSWPostfilter, build_params, query_params)
        print("running benchmark")
        D, I, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I)
        recall = logger.log_recall(run_path)
        print("recall = ", recall)
