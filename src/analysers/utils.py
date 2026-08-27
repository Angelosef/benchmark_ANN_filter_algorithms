import os
import json
import pandas as pd
import numpy as np


import src.datasets.all_datasets
from src.datasets.base_dataset import Dataset
import os


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

