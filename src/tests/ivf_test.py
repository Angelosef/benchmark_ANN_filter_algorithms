from src.benchmark.benchmarkRunner import BenchmarkRunner
from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset
from src.datasets.gist import gistDataset
from src.algorithms.ivf_id_filter import IVFIdFilter, IVFIdFilterBuildParameters, IVFIdFilterQueryParameters
from src.logger import BenchmarkLogger


if __name__=="__main__":
    test_sift = False
    test_glove = False
    test_yfcc = False
    test_gist = True

    if test_sift:
        subset_size = 0.1
        k = 10
        
        dataset = siftDataset(subset_size, k)
        print("running IVF on sift dataset")
        for num_restrictions in range(1, 4):
            print("num_restr = ", num_restrictions)
            
            build_params = IVFIdFilterBuildParameters(nlist=100)
            query_params = IVFIdFilterQueryParameters(nprobe=20)
    
            print("initializing benchmark")
            runner = BenchmarkRunner(dataset, num_restrictions, IVFIdFilter, build_params, query_params)
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
        print("running IVF on glove dataset")
        build_params = IVFIdFilterBuildParameters(nlist=100)
        query_params = IVFIdFilterQueryParameters(nprobe=20)

        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, None, IVFIdFilter, build_params, query_params)
        print("running benchmark")
        D, I, L, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I, L)
        recall = logger.log_recall(run_path)
        print("recall = ", recall)

    if test_yfcc:
        subset_size = 0.1
        k = 10

        dataset = yfccDataset(subset_size, k)
        print("running IVF on yfcc dataset")
        build_params = IVFIdFilterBuildParameters(nlist=200)
        query_params = IVFIdFilterQueryParameters(nprobe=50)

        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, None, IVFIdFilter, build_params, query_params)
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
        print("running IVF on gist dataset")
        build_params = IVFIdFilterBuildParameters(nlist=200)
        query_params = IVFIdFilterQueryParameters(nprobe=50)

        print("initializing benchmark")
        runner = BenchmarkRunner(dataset, None, IVFIdFilter, build_params, query_params)
        print("running benchmark")
        D, I, L, metadata = runner.run()

        print("logging results")
        logger = BenchmarkLogger()
        run_path = logger.log_benchmark(metadata, D, I, L)
        recall = logger.log_recall(run_path)
        print("recall = ", recall)



