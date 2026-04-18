from src.datasets.base_dataset import Dataset, Config
import os
import numpy as np
import tarfile
from src.datasets.utils import (download_file, load_fvecs, plot_selectivity, save_gt_isolated)
import json

class siftDataset(Dataset):
    def __init__(self, subset_size=1.0, neighbors_retrieved=10):
        super().__init__("SIFT", subset_size, neighbors_retrieved)
        self.url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
        
        # for the synthetic data:
        self.number_of_attributes = 3
        self.value_cardinality = 6
    
    def get_config(self):
        return Config("structured", "conjunction")
    
    def download(self):
        filename = 'sift.tar.gz'
        path = os.path.join(self.dataset_path, filename)
        download_file(self.url, path)
        return path

    def extract(self):
        base_vec_fn = os.path.join(self.dataset_path, 'base', 'vectors.fvecs')
        query_vec_fn = os.path.join(self.dataset_path, 'queries', 'vectors.fvecs')
        if os.path.exists(base_vec_fn) and os.path.exists(query_vec_fn):
            print("files already extracted")
            return
        tar_filename = os.path.join(self.dataset_path, 'sift.tar.gz')
        with tarfile.open(tar_filename, "r:gz") as tar:
            files = {
                "sift/sift_base.fvecs": base_vec_fn,
                "sift/sift_query.fvecs": query_vec_fn
            }

            for internal_path, final_destination in files.items():
                os.makedirs(os.path.dirname(final_destination), exist_ok=True)
                
                tar.extract(internal_path, path=self.dataset_path)
                
                current_path = os.path.join(self.dataset_path, internal_path)
                os.rename(current_path, final_destination)
            
            
    def build_shared_files(self):
        base_path = os.path.join(self.dataset_path, 'base')
        query_path = os.path.join(self.dataset_path, 'queries')

        if os.path.exists(base_path) and os.path.exists(query_path):
            print("shared files already built")
            return
        self.extract()
        
        base_vecs_filename = os.path.join(self.dataset_path, 'base', 'vectors.fvecs')
        query_vecs_filename = os.path.join(self.dataset_path, 'queries', 'vectors.fvecs')

        base_vecs = load_fvecs(base_vecs_filename)
        query_vecs = load_fvecs(query_vecs_filename)

        os.makedirs(os.path.join(self.dataset_path, 'base'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, 'queries'), exist_ok=True)

        np.save(os.path.join(self.dataset_path, 'base', 'vectors.npy'), base_vecs)
        np.save(os.path.join(self.dataset_path, 'queries', 'vectors.npy'), query_vecs)

        base_attributes = self.create_synthetic(size=base_vecs.shape[0], n_max=self.value_cardinality, number_of_attributes=self.number_of_attributes)
        np.save(os.path.join(self.dataset_path, 'base', 'attributes.npy'), base_attributes)

        for i in range(1, self.number_of_attributes+1):
            query_filter = self.create_filters(size=query_vecs.shape[0], n_max=self.value_cardinality, number_of_attributes=self.number_of_attributes, number_of_restrictions=i)
            os.makedirs(os.path.join(self.dataset_path, 'queries', 'restriction_'+str(i)), exist_ok=True)
            np.save(os.path.join(self.dataset_path, 'queries', 'restriction_'+str(i), 'filters.npy'), query_filter)
    
        self.save_global_metadata(base_vecs, query_vecs)

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

        for num_restr in range(1, self.number_of_attributes+1):
            gt_dst_path = os.path.join(subset_path, 'queries', 'restriction_'+str(num_restr))
            gt_ids_path = os.path.join(subset_path, 'queries', 'restriction_'+str(num_restr))
            save_gt_isolated(self, num_restr, gt_dst_path, gt_ids_path)
        
        return

    
    def calculate_selectivity(self, number_of_restrictions):
        base_attributes = self.get_base_attributes()
        query_filters = self.get_query_filters(number_of_restrictions)
        counts = []
        for i in range(query_filters.shape[0]):
            q_filt = query_filters[i]

            mask = np.all((base_attributes == q_filt) | (q_filt == -1), axis=1)
            
            valid_ids = np.where(mask)[0].astype('int32')
            counts.append(len(valid_ids))
        
        counts = np.array(counts)
        subset_path = self.get_subset_path_or_fail()
        os.makedirs(os.path.join(subset_path, 'analysis', 'restriction_'+str(number_of_restrictions)), exist_ok=True)
        np.save(os.path.join(subset_path, 'analysis', 'restriction_'+str(number_of_restrictions), 'selectivity.npy'), counts)
        
        return counts
    
    def plot_selectivity(self, number_of_restrictions, relative_counts=False):
        subset_path = self.get_subset_path_or_fail()
        counts = np.load(os.path.join(subset_path, 'analysis', 'restriction_'+str(number_of_restrictions), 'selectivity.npy'))

        base_count = None
        if relative_counts:
            metadata_path = os.path.join(subset_path, 'metadata.json')
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                base_count = metadata.get("base_count")                
        dst_path = os.path.join(subset_path, 'analysis', 'restriction_'+str(number_of_restrictions))
        param_dict = {
            "number_of_restrictions": str(number_of_restrictions)
        }
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
    
    def get_query_filters(self, number_of_restrictions):
        subset_path = self.get_subset_path_or_fail()
        filter = np.load(os.path.join(self.dataset_path, 'queries', 'restriction_'+str(number_of_restrictions), 'filters.npy',))
        query_ids = np.load(os.path.join(subset_path, 'queries', 'ids.npy'))
        
        return filter[query_ids]
    
    def get_ground_truth_ids(self, number_of_restrictions):
        subset_path = self.get_subset_path_or_fail()
        gt_ids = np.load(os.path.join(subset_path, 'queries', 'restriction_'+str(number_of_restrictions), 'ground_truth_ids.npy'))
        
        return gt_ids

    def create_synthetic(self, size, n_max, number_of_attributes):
        return self.rng.integers(0, n_max, size=(size, number_of_attributes), dtype='int32')
    
    def create_filters(self, size, n_max, number_of_attributes, number_of_restrictions):
        filters = np.full((size, number_of_attributes), -1, dtype=np.int32)

        for i in range(number_of_restrictions):
            filters[:, i] = self.rng.integers(0, n_max, size)
        
        return filters
