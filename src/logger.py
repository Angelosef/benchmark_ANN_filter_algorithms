import os
import json
import time
import numpy as np
import pandas as pd
from src.datasets.base_dataset import Dataset
import src.datasets.all_datasets


class BenchmarkLogger:
    def __init__(self, base_log_dir="logs"):
        self.base_log_dir = base_log_dir
        os.makedirs(self.base_log_dir, exist_ok=True)
        self.master_log_path = os.path.join(self.base_log_dir, "master_registry.csv")
    
    def get_log_dir(self):
        return self.base_log_dir

    def log_benchmark(self, runner_results, D, I, L):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"{runner_results['index_name']}_{runner_results['dataset_name']}_{timestamp}"
        run_dir = os.path.join(self.base_log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        np.save(os.path.join(run_dir, "indices.npy"), I)
        np.save(os.path.join(run_dir, "distances.npy"), D)
        np.save(os.path.join(run_dir, "latencies.npy"), L)
        

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
            "query_params": str(metadata["query_params"]),
            "ds_query_param": metadata["ds_query_param"]
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
        recalls = calculate_recalls(I, gt_ids, k)

        avg_recall = np.mean(recalls)
        metadata["avg_recall"] = avg_recall
        metadata["p2_recall"] = np.percentile(recalls, 2)
        metadata["p5_recall"] = np.percentile(recalls, 5)
        metadata["p25_recall"] = np.percentile(recalls, 25)
        metadata["p50_recall"] = np.percentile(recalls, 50)
        metadata["p75_recall"] = np.percentile(recalls, 75)
        metadata["p95_recall"] = np.percentile(recalls, 95)
        metadata["p98_recall"] = np.percentile(recalls, 98)

        np.save(os.path.join(run_dir, "recalls.npy"), recalls)
        
        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        
        return avg_recall
    
    def log_latency_stats(self, run_dir):
        latencies = np.load(os.path.join(run_dir, 'latencies.npy'))

        with open(os.path.join(run_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        metadata['avg_latency'] = np.mean(latencies)
        metadata["p2_latency"] = np.percentile(latencies, 2)
        metadata["p5_latency"] = np.percentile(latencies, 5)
        metadata["p25_latency"] = np.percentile(latencies, 25)
        metadata["p50_latency"] = np.percentile(latencies, 50)
        metadata["p75_latency"] = np.percentile(latencies, 75)
        metadata["p95_latency"] = np.percentile(latencies, 95)
        metadata["p98_latency"] = np.percentile(latencies, 98)

        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
        return

def calculate_avg_recall(I, gt_ids, k):
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

def calculate_recalls(I, gt_ids, k):
    """
    I: [num_queries, k_retrieved]
    gt_ids: [num_queries, k_ground_truth]
    """
    recalls = np.zeros(len(I))
    retrieved = I[:, :k]
    truth = gt_ids[:, :k]
    
    for i in range(len(I)):
        recalls[i] = np.isin(retrieved[i], truth[i]).sum()
    recalls = recalls / k
        
    return recalls
    
def find_benchmarks(dataset_name=None, index_name=None):
    df = pd.read_csv("logs/master_registry.csv")
    
    if dataset_name:
        df = df[df['dataset'] == dataset_name]
    if index_name:
        df = df[df['index'] == index_name]
        
    return df
