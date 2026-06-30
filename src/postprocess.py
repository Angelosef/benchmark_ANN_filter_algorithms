import os
from src.logger import BenchmarkLogger
from src.plotter import ANNBenchmarkPlotter

import src.datasets.all_datasets
from src.datasets.base_dataset import Dataset
import json

def find_selectivity_path(ds_name, subset_size, neighbors_retrieved, query_param=None):
    cls_name = Dataset.get_dataset_class(ds_name)
    ds = cls_name(subset_size, neighbors_retrieved)
    return ds.get_selectivity_path(query_param)

if __name__ == "__main__":
    logger = BenchmarkLogger()
    plotter = ANNBenchmarkPlotter(log_root_dir="logs", output_dir="benchmark_plots")

    target_dir = logger.get_log_dir()
    run_dirs = [
        os.path.join(target_dir, name) 
        for name in os.listdir(target_dir) 
            if os.path.isdir(os.path.join(target_dir, name))
    ]

    # run_dirs = ['logs/HNSWPostfilter_YFCC_20260610-232410']

    for run_directory in run_dirs:
        logger.log_recall(run_directory)

        """
        logger.log_latency_stats(run_directory)

        
        plotter.load_and_plot_recall(run_directory)
        plotter.load_and_plot_recall_latency(run_directory)

        with open(os.path.join(run_directory, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        sel_path = find_selectivity_path(
            metadata.get('dataset_name'),
            metadata.get('subset_size'),
            metadata.get('neighbors_retrieved'),
            metadata.get('ds_query_param')
        )

        plotter.load_and_plot_selectivity_avg_recall(run_directory, sel_path)
        plotter.load_and_plot_selectivity_avg_latency(run_directory, sel_path)

        """
        