from src.benchmark.benchmarkRunner import BenchmarkRunner
from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset
from src.datasets.gist import gistDataset
from src.algorithms.brute_force import BruteForceIdFilter
from src.logger import BenchmarkLogger

class dummy:
    def __init__(self):
        pass

if __name__=="__main__":
    test_sift = True
    test_glove = True
    test_yfcc = True
    test_gist = True

    if test_sift:
        print("running brute force on sift dataset")
        for num_restrictions in range(1, 4):
            print("num_restr = ", num_restrictions)
            subset_size = 0.1
            k = 10
        
            dataset = siftDataset(subset_size, k)
            print("initializing benchmark")
            runner = BenchmarkRunner(dataset, num_restrictions, BruteForceIdFilter, dummy(), dummy())
            print("running benchmark")
            D, I, L, metadata = runner.run()

            print("logging results")
            logger = BenchmarkLogger()
            run_path = logger.log_benchmark(metadata, D, I, L)
            recall = logger.log_recall(run_path)

            print("recall = ", recall)

    if test_glove:
            print("running brute force on glove dataset")
        
            subset_size = 0.1
            k = 10
        
            dataset = GloVeDataset(subset_size, k)
            print("initializing benchmark")
            runner = BenchmarkRunner(dataset, None, BruteForceIdFilter, dummy(), dummy())
            print("running benchmark")
            D, I, L, metadata = runner.run()

            print("logging results")
            logger = BenchmarkLogger()
            run_path = logger.log_benchmark(metadata, D, I, L)
            recall = logger.log_recall(run_path)

            print("recall = ", recall)

    if test_yfcc:
        print("running brute force on yfcc dataset")
        
        subset_size = 0.1
        k = 10
    
        dataset = yfccDataset(subset_size, k)
        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, None, BruteForceIdFilter, dummy(), dummy())
        print("running benchmark")
        D, I, L, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I, L)
        recall = logger.log_recall(run_path)

        print("recall = ", recall)
    
    if test_gist:
        print("running brute force on gist dataset")
        
        subset_size = 0.1
        k = 10
    
        dataset = gistDataset(subset_size, k)
        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, None, BruteForceIdFilter, dummy(), dummy())
        print("running benchmark")
        D, I, L, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I, L)
        recall = logger.log_recall(run_path)

        print("recall = ", recall)