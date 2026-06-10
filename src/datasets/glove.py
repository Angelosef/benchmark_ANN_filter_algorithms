from src.datasets.base_dataset import Dataset, Config
import os
import numpy as np
import zipfile
from src.datasets.utils import (download_file, load_vecs_from_txt, plot_selectivity, save_gt_isolated)
import json

class GloVeDataset(Dataset):
    def __init__(self, subset_size=1.0, neighbors_retrieved=10):
        super().__init__("GLOVE", subset_size, neighbors_retrieved)
        self.url = f"https://nlp.stanford.edu/data/glove.6B.zip"
        self.base_ratio = 0.99

        # for the synthetic data:
        self.number_of_attributes = 3
        self.value_cardinality = 6
        #for the query filters
        self.vals_per_attr = 2
        return
    
    
    def get_config(self):
        return Config("structured", "CNF")
    
    def download(self):
        filename = os.path.join(self.dataset_path, 'glove.6B.zip')
        download_file(self.url, filename)
        return

    def extract(self):
        filename = os.path.join(self.dataset_path, 'glove.6B.zip')
        extraction_dir = self.dataset_path 
        target_txt = "glove.6B.300d.txt"
        final_path = os.path.join(extraction_dir, target_txt)
        if os.path.exists(final_path):
            print("file already extracted")
            return

        print(f"Opening {filename}...")
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            # Get a list of all files inside the ZIP
            all_files = zip_ref.namelist()
            
            if target_txt in all_files:
                print(f"Extracting ONLY {target_txt}...")
                zip_ref.extract(target_txt, path=extraction_dir)
            else:
                print(f"Error: {target_txt} not found in the zip!")
                # Fallback: list what IS there so you can debug
                print("Available files:", all_files)
                return None

        print(f"GloVe 300d ready at: {final_path}")
        return final_path
    
    def build_shared_files(self):
        base_path = os.path.join(self.dataset_path, 'base')
        query_path = os.path.join(self.dataset_path, 'queries')

        if os.path.exists(base_path) and os.path.exists(query_path):
            print("shared files already built")
            return
        self.extract()
        
        ds_filename = os.path.join(self.dataset_path, 'glove.6B.300d.txt')
        loaded_vectors = load_vecs_from_txt(ds_filename)
        indices = self.rng.permutation(loaded_vectors.shape[0])
        all_vectors = loaded_vectors[indices]
        base_ids_end = int(self.base_ratio * len(indices))
        base_vecs = all_vectors[:base_ids_end]
        query_vecs = all_vectors[base_ids_end:]
        
        base_attributes = self.create_synthetic(base_vecs.shape[0], self.value_cardinality, self.number_of_attributes)
        filters = self.create_filters(query_vecs.shape[0], self.value_cardinality, self.number_of_attributes, self.vals_per_attr)

        os.makedirs(os.path.join(self.dataset_path, 'base'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, 'queries'), exist_ok=True)

        np.save(os.path.join(self.dataset_path, 'base', 'vectors.npy'), base_vecs)
        np.save(os.path.join(self.dataset_path, 'queries', 'vectors.npy'), query_vecs)
        np.save(os.path.join(self.dataset_path, 'base', 'attributes.npy'), base_attributes)

        np.save(os.path.join(self.dataset_path, 'queries', 'filters.npy'), filters)
        
        self.save_global_metadata(base_vecs, query_vecs)
        return
    
    def create_subset(self):
        if not (self.find_subset_path() is None):
            return
        subset_id = self.get_next_subset_id()
        subset_path = os.path.join(self.dataset_path, 'subsets', str(subset_id))

        os.makedirs(subset_path, exist_ok=True)

        full_base_count = self.get_full_base_count()
        full_query_count = self.get_full_query_count()

        base_size = int(full_base_count * self.subset_size)
        query_size = int(full_query_count * self.subset_size)

        base_ids = self.rng.choice(np.arange(full_base_count), size=base_size, replace=False)
        base_ids.sort()
        query_ids = self.rng.choice(np.arange(full_query_count), size=query_size, replace=False)
        query_ids.sort()

        os.makedirs(os.path.join(subset_path, 'base'), exist_ok=True)
        os.makedirs(os.path.join(subset_path, 'queries'), exist_ok=True)
        np.save(os.path.join(subset_path, 'base', 'ids.npy'), base_ids)
        np.save(os.path.join(subset_path, 'queries', 'ids.npy'), query_ids)

        metadata = {
            "subset_size": self.subset_size,
            "neighbors_retrieved": self.neighbors_retrieved,
            "base_count": len(base_ids),
            "query_count": len(query_ids)
        }

        with open(os.path.join(subset_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        gt_ids_path = os.path.join(subset_path, 'queries')
        gt_dst_path = os.path.join(subset_path, 'queries')
        save_gt_isolated(self, None, gt_dst_path, gt_ids_path)

        return
    
    def calculate_selectivity(self):
        base_attributes = self.get_base_attributes()
        query_filters = self.get_query_filters()
        counts = []
        for i in range(query_filters.shape[0]):
            q_filt = query_filters[i]
            
            combined_mask = np.ones(base_attributes.shape[0], dtype=bool)

            for attr_idx in range(self.number_of_attributes):
                attr_mask = np.isin(base_attributes[:, attr_idx], q_filt[attr_idx])
                
                combined_mask &= attr_mask
            
            valid_ids = np.where(combined_mask)[0].astype('int64')
            counts.append(len(valid_ids))
        
        counts = np.array(counts)
        subset_path = self.get_subset_path_or_fail()
        os.makedirs(os.path.join(subset_path, 'analysis'), exist_ok=True)
        np.save(os.path.join(subset_path, 'analysis', 'selectivity.npy'), counts)
        
        return counts
    
    def plot_selectivity(self, relative_counts=False):
        subset_path = self.get_subset_path_or_fail()
        counts = np.load(os.path.join(subset_path, 'analysis', 'selectivity.npy'))

        base_count = None
        if relative_counts:
            metadata_path = os.path.join(subset_path, 'metadata.json')
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                base_count = metadata.get("base_count")
        dst_path = os.path.join(subset_path, 'analysis')
        param_dict = None
        plot_selectivity(counts, self.name, dst_path, relative_counts, base_count, param_dict)
        return
    

    def get_base_vectors(self):
        subset_path = self.get_subset_path_or_fail()
        base_vecs = np.load(os.path.join(self.dataset_path, 'base', 'vectors.npy'))
        base_ids = np.load(os.path.join(subset_path, 'base', 'ids.npy'))
        
        return base_vecs[base_ids]
    
    def get_base_attributes(self):
        subset_path = self.get_subset_path_or_fail()
        base_attributes = np.load(os.path.join(self.dataset_path, 'base', 'attributes.npy'))
        base_ids = np.load(os.path.join(subset_path, 'base', 'ids.npy'))

        return base_attributes[base_ids]
    
    def get_query_vectors(self):
        subset_path = self.get_subset_path_or_fail()
        query_vecs = np.load(os.path.join(self.dataset_path, 'queries', 'vectors.npy'))
        query_ids = np.load(os.path.join(subset_path, 'queries', 'ids.npy'))

        return query_vecs[query_ids]
    
    def get_query_filters(self, query_type=None):
        subset_path = self.get_subset_path_or_fail()
        filter = np.load(os.path.join(self.dataset_path, 'queries', 'filters.npy'))
        query_ids = np.load(os.path.join(subset_path, 'queries', 'ids.npy'))
        
        return filter[query_ids]
    
    def get_ground_truth_ids(self, query_type=None):
        subset_path = self.get_subset_path_or_fail()
        gt_ids = np.load(os.path.join(subset_path, 'queries', 'ground_truth_ids.npy'))
        
        return gt_ids

    def create_synthetic(self, size, n_max, number_of_attributes):
        return self.rng.integers(0, n_max, size=(size, number_of_attributes), dtype='int32')
    
    def create_filters(self, size, n_max, num_attributes, values_per_attr):
        filters = np.empty((size, num_attributes, values_per_attr), dtype=np.int32)
        
        for q_idx in range(size):
            for attr_idx in range(num_attributes):
                filters[q_idx, attr_idx] = self.rng.choice(
                    n_max, 
                    size=values_per_attr, 
                    replace=False
                )
                
        return filters
    
    def get_selectivity_path(self, query_type=None):
        subset_path = self.get_subset_path_or_fail()
        path = os.path.join(subset_path, 'analysis', 'selectivity.npy')
        
        return path

