from src.plotter import ANNBenchmarkPlotter

if __name__ == '__main__':
    plotter = ANNBenchmarkPlotter(log_root_dir="logs", output_dir="benchmark_plots")
    
    test_sift = False
    test_glove = False
    test_yfcc = True
    test_gist = False

    if test_sift:
        for num_restrictions in range(1, 4):
            plotter.plot(dataset_name="SIFT", subset_size=0.1, ds_query_param=num_restrictions)

    if test_glove:
        plotter.plot(dataset_name="GLOVE", subset_size=0.1, ds_query_param=None)

    if test_yfcc:
        plotter.plot(dataset_name="YFCC", subset_size=0.1, ds_query_param=None)
    
    if test_gist:
        plotter.plot(dataset_name="GIST", subset_size=0.1, ds_query_param=None)
