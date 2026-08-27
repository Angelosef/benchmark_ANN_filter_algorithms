import os
from src.logger import BenchmarkLogger

if __name__ == "__main__":
    logger = BenchmarkLogger()

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
