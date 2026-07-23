
import src.datasets.all_datasets
from src.datasets.base_dataset import Dataset


def find_selectivity_path(ds_name, subset_size, neighbors_retrieved, query_param=None):
    cls_name = Dataset.get_dataset_class(ds_name)
    ds = cls_name(subset_size, neighbors_retrieved)
    
    return ds.get_selectivity_path(query_param)