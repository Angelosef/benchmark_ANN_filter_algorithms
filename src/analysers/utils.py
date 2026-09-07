import os
import json
import pandas as pd
import numpy as np


import src.datasets.all_datasets
from src.datasets.base_dataset import Dataset
import os
from src.logger import BenchmarkLogger

def find_selectivity_path(ds_name, subset_size, neighbors_retrieved, query_param=None):
    cls_name = Dataset.get_dataset_class(ds_name)
    ds = cls_name(subset_size, neighbors_retrieved)
    
    return ds.get_selectivity_path(query_param)

def load_run_data(metadata_run_path):
    """Reads the individual JSON metadata for a specific run."""
    if not os.path.isfile(metadata_run_path):
        print(f"Warning: Metadata for {metadata_run_path} not found.")
        return None
    with open(metadata_run_path, 'r') as f:
        return json.load(f)

def get_run_dirs(codeword=None):
    logger = BenchmarkLogger()
    target_dir = logger.get_log_dir()
    
    run_dirs = [
        os.path.join(target_dir, name) 
        for name in os.listdir(target_dir) 
        if os.path.isdir(os.path.join(target_dir, name))
        and (codeword is None or codeword in name)
    ]
    
    return run_dirs