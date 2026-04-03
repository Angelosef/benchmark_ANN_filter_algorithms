import os
import numpy as np
from scipy.sparse import issparse
import json

class Config:
    def __init__(self, attribute_type, query_type):
        self.attribute_type = attribute_type
        self.query_type = query_type

BASE_DIRECTORY = 'data'

class Dataset:
    registry = {}
    
    def __init__(self, name, subset_size=1.0, neighbors_retrieved=10):
        
        self.name = name
        self.subset_size = subset_size
        self.neighbors_retrieved = neighbors_retrieved
        self.dataset_path = os.path.join(BASE_DIRECTORY, name)
        self.rng = np.random.default_rng(seed=42)

        os.makedirs(self.dataset_path, exist_ok=True)
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "get_name"):
            cls.registry[cls().get_name()] = cls 

    @classmethod
    def get_dataset_class(cls, name):
        if name not in cls.registry:
            raise ValueError(f"Dataset {name} not found in registry.")
        return cls.registry[name]

    def download(self):
        raise NotImplementedError

    def build_shared_files(self):
        raise NotImplementedError
    
    def create_subset(self):
        raise NotImplementedError
    
    def get_base_vectors(self):
        raise NotImplementedError

    def get_query_vectors(self):
        raise NotImplementedError
    
    def get_query_filters(self, query_type=None):
        raise NotImplementedError

    def get_ground_truth_ids(self, query_type=None):
        raise NotImplementedError
    
    def get_neighbors_retrieved(self):
        return self.neighbors_retrieved
    
    def get_name(self):
        return self.name
    
    def get_dim(self):
        subset_path = self.get_subset_path_or_fail()
        metadata_path = os.path.join(subset_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata["vector_dim"]
    
    def get_base_count(self):
        subset_path = self.get_subset_path_or_fail()
        metadata_path = os.path.join(subset_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata["base_count"]
    
    def get_query_count(self):
        subset_path = self.get_subset_path_or_fail()
        metadata_path = os.path.join(subset_path, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata["query_count"]
    
    def get_metric(self):
        return "L2"
    
    def get_subset_size(self):
        return self.subset_size
    
    def get_neighbors_retrieved(self):
        return self.neighbors_retrieved
    
    def get_config(self):
        raise NotImplementedError
    
    def prepare(self):
        self.download()
        self.build_shared_files()
        self.create_subset()
        return
    
    def find_subset_path(self):
        """
        Returns the absolute path to the matching subset folder if it exists, 
        otherwise returns None.
        """
        subsets_path = os.path.join(self.dataset_path, 'subsets')
        
        if not os.path.isdir(subsets_path):
            return None

        # Check numeric folders
        for folder_name in os.listdir(subsets_path):
            folder_path = os.path.join(subsets_path, folder_name)
            
            if os.path.isdir(folder_path) and folder_name.isdigit():
                metadata_path = os.path.join(folder_path, 'metadata.json')
                
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                            
                        # Validation logic
                        if (metadata.get("subset_size") == self.subset_size and 
                            metadata.get("neighbors_retrieved") == self.neighbors_retrieved):
                            return folder_path  # Return the full path to the match
                            
                    except (json.JSONDecodeError, IOError):
                        continue
                        
        return None
    
    def get_next_subset_id(self):
        subsets_path = os.path.join(self.dataset_path, 'subsets')
        
        if not os.path.exists(subsets_path):
            os.makedirs(subsets_path, exist_ok=True)
            return 1

        existing_items = os.listdir(subsets_path)
        
        ids = []
        for item in existing_items:
            if os.path.isdir(os.path.join(subsets_path, item)) and item.isdigit():
                ids.append(int(item))
        
        if not ids:
            return 1
            
        return max(ids) + 1

    def get_subset_path_or_fail(self):
        subset_path = self.find_subset_path()
        if subset_path is None:
            raise RuntimeError("Subset has not been created yet")
        return subset_path

    def inspect_data(self, vecs, attrs, q_vecs, q_filters):
        print(f"\n{'='*10} {self.name} Dataset {'='*10}")
        data_map = {
            "Base Vectors": vecs,
            "Base Attributes": attrs,
            "Query Vectors": q_vecs,
            "Query Filters": q_filters
        }
        
        for label, arr in data_map.items():
            if arr is not None:
                if issparse(arr):
                    # For CSR/Sparse: sum of data, indices, and indptr arrays
                    bytes_used = arr.data.nbytes + arr.indices.nbytes + arr.indptr.nbytes
                    dtype_str = f"sparse({arr.dtype})"
                    shape_str = str(arr.shape)
                else:
                    bytes_used = arr.nbytes
                    dtype_str = str(arr.dtype)
                    shape_str = str(arr.shape)

                mb = bytes_used / (1024**2)
                print(f"{label:16} | Shape: {shape_str:18} | Dtype: {dtype_str:12} | Size: {mb:7.2f} MB")
            else:
                print(f"{label:16} | NOT LOADED")

