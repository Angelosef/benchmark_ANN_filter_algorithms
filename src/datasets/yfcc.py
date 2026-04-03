from src.datasets.base_dataset import Dataset, Config
import os
import numpy as np
from src.datasets.utils import (download_file, load_vectors_from_u8bin, load_metadata, plot_selectivity, plot_tag_popularity)
import json
import faiss
from scipy.sparse import csr_matrix, save_npz, load_npz, issparse

class yfccDataset(Dataset):
    def __init__(self, subset_size=1.0, neighbors_retrieved=10):
        super().__init__("YFCC", subset_size, neighbors_retrieved)

        self.url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/"
        self.files_to_download = {
            "base.10M.u8bin": "base.10M.u8bin",
            "query.public.100K.u8bin": "query.public.100K.u8bin",
            "base.metadata.10M.spmat": "base.metadata.10M.spmat",
            "query.metadata.public.100K.spmat": "query.metadata.public.100K.spmat"
        }
        
        return
    
    
    def get_config(self):
        return Config("sparse", "conjunction")
    
    def download(self):
        for filename, remote_path in self.files_to_download.items():
            url = self.url + remote_path
            local_path = os.path.join(self.dataset_path, filename)

            download_file(url, local_path)

    def build_shared_files(self):
        base_path = os.path.join(self.dataset_path, 'base')
        query_path = os.path.join(self.dataset_path, 'queries')

        if os.path.exists(base_path) and os.path.exists(query_path):
            print("shared files already built")
            return
        
        base_vecs = load_vectors_from_u8bin(os.path.join(self.dataset_path, "base.10M.u8bin"))
        query_vecs = load_vectors_from_u8bin(os.path.join(self.dataset_path, "query.public.100K.u8bin"))
        
        # Load metadata (keeping them as CSR matrices for now)
        base_attributes = load_metadata(os.path.join(self.dataset_path, "base.metadata.10M.spmat"))
        filters = load_metadata(os.path.join(self.dataset_path, "query.metadata.public.100K.spmat"))

        os.makedirs(os.path.join(self.dataset_path, 'base'), exist_ok=True)
        os.makedirs(os.path.join(self.dataset_path, 'queries'), exist_ok=True)

        np.save(os.path.join(self.dataset_path, 'base', 'vectors.npy'), base_vecs)
        np.save(os.path.join(self.dataset_path, 'queries', 'vectors.npy'), query_vecs)
        save_npz(os.path.join(self.dataset_path, 'base', 'attributes.npz'), base_attributes)
        save_npz(os.path.join(self.dataset_path, 'queries', 'filters.npz'), filters)
        return
    
    def create_subset(self):
        if not (self.find_subset_path() is None):
            return
        subset_id = self.get_next_subset_id()
        subset_path = os.path.join(self.dataset_path, 'subsets', str(subset_id))

        os.makedirs(subset_path, exist_ok=True)

        base_vecs = np.load(os.path.join(self.dataset_path, 'base', 'vectors.npy'))
        query_vecs = np.load(os.path.join(self.dataset_path, 'queries', 'vectors.npy'))

        base_size = int(base_vecs.shape[0] * self.subset_size)
        query_size = int(query_vecs.shape[0] * self.subset_size)

        base_ids = self.rng.choice(np.arange(base_vecs.shape[0]), size=base_size, replace=False)
        base_ids.sort()
        query_ids = self.rng.choice(np.arange(query_vecs.shape[0]), size=query_size, replace=False)
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
            "vector_dim": base_vecs.shape[1],
            "vector_dtype": str(base_vecs.dtype),
            "base_count": len(base_ids),
            "query_count": len(query_ids)
        }
        with open(os.path.join(subset_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

        base_vecs = self.get_base_vectors()
        base_attributes = self.get_base_attributes()
        query_vecs = self.get_query_vectors()
        query_filters = self.get_query_filters()

        gt_ids, gt_dst = self.calculate_ground_truth(base_vecs, base_attributes, query_vecs, query_filters)
        np.save(os.path.join(subset_path, 'queries', 'ground_truth_ids.npy'), gt_ids)
        np.save(os.path.join(subset_path, 'queries', 'distances.npy'), gt_dst)

        return
    
    def calculate_ground_truth(self, base_vecs, base_attributes, query_vecs, query_filters):
        print("calculating ground truth")
        k = self.neighbors_retrieved
        num_queries = query_vecs.shape[0]
        
        gt_ids = np.full((num_queries, k), -1, dtype=np.int64)
        gt_dst = np.full((num_queries, k), np.inf, dtype=np.float32)

        d = base_vecs.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(base_vecs)
        
        docs_per_word = base_attributes.T.tocsr()

        for q in range(num_queries):
            q_tags = query_filters[q].indices
            
            if len(q_tags) == 0:
                continue

            valid_ids = docs_per_word[q_tags[0]].indices
            for tag in q_tags[1:]:
                valid_ids = np.intersect1d(valid_ids, docs_per_word[tag].indices)

            if len(valid_ids) == 0:
                print("no valid ids")
                continue

            selector = faiss.IDSelectorBatch(valid_ids.astype('int64'))
            params = faiss.SearchParameters(sel=selector)
            
            distances, indices = index.search(query_vecs[q:q+1], k, params=params)
            
            actual_k = min(k, len(valid_ids))
            gt_ids[q, :actual_k] = indices[0, :actual_k]
            gt_dst[q, :actual_k] = distances[0, :actual_k]

        return gt_ids, gt_dst
    
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

    def get_base_vectors(self):
        subset_path = self.get_subset_path_or_fail()
        base_vecs = np.load(os.path.join(self.dataset_path, 'base', 'vectors.npy'))
        base_ids = np.load(os.path.join(subset_path, 'base', 'ids.npy'))
        
        return base_vecs[base_ids].astype('float32')
    
    def get_base_attributes(self):
        subset_path = self.get_subset_path_or_fail()
        base_attributes = load_npz(os.path.join(self.dataset_path, 'base', 'attributes.npz'))
        base_ids = np.load(os.path.join(subset_path, 'base', 'ids.npy'))

        return base_attributes[base_ids]
    
    def get_query_vectors(self):
        subset_path = self.get_subset_path_or_fail()
        query_vecs = np.load(os.path.join(self.dataset_path, 'queries', 'vectors.npy'))
        query_ids = np.load(os.path.join(subset_path, 'queries', 'ids.npy'))

        return query_vecs[query_ids].astype('float32')
    
    def get_query_filters(self, query_type=None):
        subset_path = self.get_subset_path_or_fail()
        filter = load_npz(os.path.join(self.dataset_path, 'queries', 'filters.npz'))
        query_ids = np.load(os.path.join(subset_path, 'queries', 'ids.npy'))
        
        return filter[query_ids]
    
    def get_ground_truth_ids(self, query_type=None):
        subset_path = self.get_subset_path_or_fail()
        gt_ids = np.load(os.path.join(subset_path, 'queries', 'ground_truth_ids.npy'))
        
        return gt_ids

