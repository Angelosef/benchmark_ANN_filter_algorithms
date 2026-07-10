from src.datasets.base_dataset import Dataset, Config
import os
import numpy as np
import tarfile
from src.datasets.utils import (download_file, load_fvecs, save_gt_isolated, generate_zipfian_tags,
                                 generate_zipfian_fixed_tags, plot_selectivity, plot_tag_popularity)
import json
from scipy.sparse import save_npz, load_npz

class gistDataset(Dataset):
    def __init__(self, subset_size=1.0, neighbors_retrieved=10):
        super().__init__("GIST", subset_size, neighbors_retrieved)
        self.url = 'ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz'
        
        # for the synthetic data:
        # folows zipfian distribution
        self.a = 0.8
        self.max_prob = 0.5
        self.tags_count = 500
    
    def get_config(self):
        return Config("sparse", "conjunction")
    
    def download(self):
        filename = 'gist.tar.gz'
        path = os.path.join(self.dataset_path, filename)
        download_file(self.url, path)
        return path

    def extract(self):
        base_vec_fn = os.path.join(self.dataset_path, 'base', 'vectors.fvecs')
        query_vec_fn = os.path.join(self.dataset_path, 'queries', 'vectors.fvecs')
        if os.path.exists(base_vec_fn) and os.path.exists(query_vec_fn):
            print("files already extracted")
            return
        tar_filename = os.path.join(self.dataset_path, 'gist.tar.gz')
        with tarfile.open(tar_filename, "r:gz") as tar:
            files = {
                "gist/gist_base.fvecs": base_vec_fn,
                "gist/gist_query.fvecs": query_vec_fn
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

        base_attributes = generate_zipfian_tags(base_vecs.shape[0], self.tags_count, self.a, self.max_prob)
        filters = generate_zipfian_fixed_tags(query_vecs.shape[0], self.tags_count, self.a, 0.5)

        save_npz(os.path.join(self.dataset_path, 'base', 'attributes.npz'), base_attributes)
        save_npz(os.path.join(self.dataset_path, 'queries', 'filters.npz'), filters)
        
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
            "neighbors_retrieved": self.neighbors_retrieved
        }
        with open(os.path.join(subset_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        counts = self.calculate_selectivity()
        valid_query_indices = np.where(counts >= self.neighbors_retrieved)[0]
        query_ids = query_ids[valid_query_indices]
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
        docs_per_word = base_attributes.T.tocsr()
        for q in range(query_filters.shape[0]):
            q_tags = query_filters[q].indices
            
            if len(q_tags) == 0:
                continue

            valid_ids = docs_per_word[q_tags[0]].indices
            for tag in q_tags[1:]:
                valid_ids = np.intersect1d(valid_ids, docs_per_word[tag].indices)
            
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
    
    def calculate_tag_counts(self):
        base_attributes = self.get_base_attributes()
        docs_per_word = base_attributes.T.tocsr()
        tag_counts = np.diff(docs_per_word.indptr)

        subset_path = self.get_subset_path_or_fail()
        os.makedirs(os.path.join(subset_path, 'analysis'), exist_ok=True)
        np.save(os.path.join(subset_path, 'analysis', 'tag_counts.npy'), tag_counts)

        return tag_counts
    
    def plot_tag_popularity(self, relative_counts=False):
        subset_path = self.get_subset_path_or_fail()
        tag_counts = np.load(os.path.join(subset_path, 'analysis', 'tag_counts.npy'))

        base_count = None
        if relative_counts:
            metadata_path = os.path.join(subset_path, 'metadata.json')
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                base_count = metadata.get("base_count")
        dst_path = os.path.join(subset_path, 'analysis')
        param_dict = None
        plot_tag_popularity(tag_counts, self.name, dst_path, relative_counts, base_count, param_dict)
        return
    
    def calculate_query_tag_counts(self):
        filters = self.get_query_filters()
        docs_per_word = filters.T.tocsr()
        tag_counts = np.diff(docs_per_word.indptr)

        subset_path = self.get_subset_path_or_fail()
        os.makedirs(os.path.join(subset_path, 'analysis', 'query'), exist_ok=True)
        np.save(os.path.join(subset_path, 'analysis', 'query', 'tag_counts.npy'), tag_counts)

        return tag_counts
    
    def plot_query_tag_popularity(self, relative_counts=False):
        subset_path = self.get_subset_path_or_fail()
        query_tag_counts = np.load(os.path.join(subset_path, 'analysis', 'query', 'tag_counts.npy'))

        base_count = None
        if relative_counts:
            metadata_path = os.path.join(subset_path, 'metadata.json')
                
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                base_count = metadata.get("base_count")
        dst_path = os.path.join(subset_path, 'analysis', 'query')
        param_dict = None
        plot_tag_popularity(query_tag_counts, self.name, dst_path, relative_counts, base_count, param_dict)
        return

    def get_base_vectors(self):
        subset_path = self.get_subset_path_or_fail()
        base_vecs = np.load(os.path.join(self.dataset_path, 'base', 'vectors.npy'))
        base_ids = np.load(os.path.join(subset_path, 'base', 'ids.npy'))
        
        return base_vecs[base_ids]
    
    def get_base_attributes(self):
        subset_path = self.get_subset_path_or_fail()
        base_attributes = load_npz(os.path.join(self.dataset_path, 'base', 'attributes.npz'))
        base_ids = np.load(os.path.join(subset_path, 'base', 'ids.npy'))

        return base_attributes[base_ids]
    
    def get_query_vectors(self):
        subset_path = self.get_subset_path_or_fail()
        query_vecs = np.load(os.path.join(self.dataset_path, 'queries', 'vectors.npy'))
        query_ids = np.load(os.path.join(subset_path, 'queries', 'ids.npy'))

        return query_vecs[query_ids]
    
    def get_query_filters(self, query_type=None):
        subset_path = self.get_subset_path_or_fail()
        filter = load_npz(os.path.join(self.dataset_path, 'queries', 'filters.npz'))
        query_ids = np.load(os.path.join(subset_path, 'queries', 'ids.npy'))
        
        return filter[query_ids]
    
    def get_ground_truth_ids(self, query_type=None):
        subset_path = self.get_subset_path_or_fail()
        gt_ids = np.load(os.path.join(subset_path, 'queries', 'ground_truth_ids.npy'))
        
        return gt_ids
    
    def get_selectivity_path(self, query_type=None):
        subset_path = self.get_subset_path_or_fail()
        path = os.path.join(subset_path, 'analysis', 'selectivity.npy')
        
        return path
