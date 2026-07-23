import os
from src.logger import BenchmarkLogger
from src.plotter import ANNBenchmarkPlotter

from src.utils import find_selectivity_path
import json


if __name__ == "__main__":
    logger = BenchmarkLogger()
    plotter = ANNBenchmarkPlotter(log_root_dir="logs", output_dir="benchmark_plots")

    target_dir = logger.get_log_dir()
    run_dirs = [
        os.path.join(target_dir, name) 
        for name in os.listdir(target_dir) 
            if os.path.isdir(os.path.join(target_dir, name))
    ]

    # run_dirs = ['logs/Acorn_YFCC_20260722-232759']

    for run_directory in run_dirs:
        logger.log_recall(run_directory)

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

        