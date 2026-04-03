import os
import json
import time
import numpy as np
import pandas as pd
from src.datasets.base_dataset import Dataset
from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset


class BenchmarkLogger:
    def __init__(self, base_log_dir="logs"):
        self.base_log_dir = base_log_dir
        os.makedirs(self.base_log_dir, exist_ok=True)
        self.master_log_path = os.path.join(self.base_log_dir, "master_registry.csv")

    def log_benchmark(self, runner_results, D, I):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{runner_results['index_name']}_{runner_results['dataset_name']}_{timestamp}"
        run_dir = os.path.join(self.base_log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        np.save(os.path.join(run_dir, "indices.npy"), I)
        np.save(os.path.join(run_dir, "distances.npy"), D)

        metadata = {
            "run_id": run_name,
            "timestamp": timestamp,
            **runner_results
        }
        
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

        self._update_master_registry(metadata)
        
        print(f"Results saved to: {run_dir}")
        return run_dir

    def _update_master_registry(self, metadata):
        summary = {
            "run_id": metadata["run_id"],
            "index": metadata["index_name"],
            "dataset": metadata["dataset_name"],
            "subset_size": metadata["subset_size"],
            "build_params": str(metadata["build_params"]),
            "query_params": str(metadata["query_params"])
        }
        
        df = pd.DataFrame([summary])
        header = not os.path.exists(self.master_log_path)
        df.to_csv(self.master_log_path, mode='a', index=False, header=header)
    
    def log_recall(self, run_dir):
        with open(os.path.join(run_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        k = metadata.get("neighbors_retrieved")
        subset_size = metadata.get("subset_size")
        ds_name = metadata.get("dataset_name")
        ds_query_param = metadata.get("ds_query_param")

        indices_path = os.path.join(run_dir, "indices.npy")
        I = np.load(indices_path)
        ds_class = Dataset.get_dataset_class(ds_name)
        dataset = ds_class(subset_size, k)
        gt_ids = dataset.get_ground_truth_ids(ds_query_param)
        recall = calculate_recall(I, gt_ids, k)

        metadata["recall"] = recall
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        
        return recall

def calculate_recall(I, gt_ids, k):
    """
    I: [num_queries, k_retrieved]
    gt_ids: [num_queries, k_ground_truth]
    """
    retrieved = I[:, :k]
    truth = gt_ids[:, :k]
    
    count = 0
    for i in range(len(I)):
        count += np.isin(retrieved[i], truth[i]).sum()
        
    return count / (len(I) * k)
    
def find_benchmarks(dataset_name=None, index_name=None):
    df = pd.read_csv("logs/master_registry.csv")
    
    if dataset_name:
        df = df[df['dataset'] == dataset_name]
    if index_name:
        df = df[df['index'] == index_name]
        
    return df
