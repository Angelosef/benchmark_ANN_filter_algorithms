import os
from src.logger import BenchmarkLogger
from src.plotter import ANNBenchmarkPlotter

if __name__ == "__main__":
    logger = BenchmarkLogger()
    plotter = ANNBenchmarkPlotter(log_root_dir="logs", output_dir="benchmark_plots")

    target_dir = logger.get_log_dir()
    run_dirs = [
        os.path.join(target_dir, name) 
        for name in os.listdir(target_dir) 
            if os.path.isdir(os.path.join(target_dir, name))
    ]

    for run_directory in run_dirs:
        # logger.log_recall(run_directory)
        plotter.load_and_plot_recall(run_directory)

