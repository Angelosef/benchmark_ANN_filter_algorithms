import os
from src.logger import BenchmarkLogger
from src.analysers.utils import get_run_dirs


if __name__ == "__main__":
    logger = BenchmarkLogger()
    run_dirs = get_run_dirs('AcornFlat')

    for run_directory in run_dirs:
        logger.log_recall(run_directory)
        logger.log_latency_stats(run_directory)
        logger.log_file_memory(run_directory)
        logger.copy_construction_metrics(run_directory)
