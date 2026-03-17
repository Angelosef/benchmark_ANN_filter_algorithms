import os
import tarfile
import zipfile
import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import json

from utils import (download_file, load_fvecs, load_vecs_from_txt, load_vectors_from_u8bin, load_metadata)

BASE_DIRECTORY = 'datasets'

class Dataset:
    def __init__(self, name, subset_size=1.0, neighbors_retrieved=10):
        self.name = name
        self.subset_size = subset_size
        self.neighbors_retrieved = neighbors_retrieved
        self.dataset_path = os.path.join(BASE_DIRECTORY, name)

        os.makedirs(self.dataset_path, exist_ok=True)

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
    
class siftDataset(Dataset):
    def __init__(self, subset_size=1.0, neighbors_retrieved=10):
        super().__init__("SIFT", subset_size, neighbors_retrieved)
        self.url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
        
        self.rng = np.random.default_rng(seed=42)
        # for the synthetic data:
        self.number_of_attributes = 3
        self.value_cardinality = 4
    
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

        for i in range(self.number_of_attributes):
            query_filter = self.create_filters(size=query_vecs.shape[0], n_max=self.value_cardinality, number_of_attributes=self.number_of_attributes, number_of_restrictions=i)
            os.makedirs(os.path.join(self.dataset_path, 'queries', 'restriction_'+str(i)), exist_ok=True)
            np.save(os.path.join(self.dataset_path, 'queries', 'restriction_'+str(i), 'filters.npy'), query_filter)
    
    def create_subset(self):
        if not (self.find_subset_path() is None):
            return
        subset_id = self.get_next_subset_id()
        subset_path = os.path.join(self.dataset_path, 'subsets', str(subset_id))

        os.makedirs(subset_path, exist_ok=True)

        base_vecs = np.load(os.path.join(self.dataset_path, 'base', 'vectors.npy'))
        base_attributes = np.load(os.path.join(self.dataset_path, 'base', 'attributes.npy'))
        query_vecs = np.load(os.path.join(self.dataset_path, 'queries', 'vectors.npy'))

        base_size = int(base_vecs.shape[0] * self.subset_size)
        query_size = int(query_vecs.shape[0] * self.subset_size)

        base_ids = np.arange(base_size)
        query_ids = np.arange(query_size)

        os.makedirs(os.path.join(subset_path, 'base'), exist_ok=True)
        os.makedirs(os.path.join(subset_path, 'queries'), exist_ok=True)
        np.save(os.path.join(subset_path, 'base', 'ids.npy'), base_ids)
        np.save(os.path.join(subset_path, 'queries', 'ids.npy'), query_ids)

        # ground truth not implemented yet
        """
        for i in range(self.number_of_attributes):
            filter = np.load(os.path.join(self.dataset_path, 'queries', 'restriction_'+str(i), 'filter.npy))
            gt_ids = self.calculate_ground_truth(base_vecs[base_ids], base_attributes[base_ids], query_vecs[query_ids], filter[query_ids])
            np.save(os.path.join(subset_path, 'queries', 'restriction_'+str(i), 'ids.npy'), gt_ids)

        """

        metadata = {
            "subset_size": self.subset_size,
            "neighbors_retrieved": self.neighbors_retrieved
        }

        with open(os.path.join(subset_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
        return

    # not implemented yet    
    def calculate_ground_truth(self, base_vecs, base_attributes, query_vecs, filter):
        ground_truth_ids = None
        return ground_truth_ids
    
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
        gt_ids = np.load(os.path.join(subset_path, 'queries', 'restriction_'+str(number_of_restrictions), 'ground_truth_ids'))
        
        return gt_ids

    def create_synthetic(self, size, n_max, number_of_attributes):
        return self.rng.integers(0, n_max, size=(size, number_of_attributes))
    
    def create_filters(self, size, n_max, number_of_attributes, number_of_restrictions):
        filters = np.full((size, number_of_attributes), -1, dtype=np.int32)

        for i in range(number_of_restrictions):
            filters[:, i] = self.rng.integers(0, n_max, size)
        
        return filters

class GloVeDataset(Dataset):
    def __init__(self, subset_size=1.0, neighbors_retrieved=10):
        super().__init__("GLOVE", subset_size, neighbors_retrieved)
        self.url = f"https://nlp.stanford.edu/data/glove.6B.zip"
        self.base_ratio = 0.99

        self.rng = np.random.default_rng(seed=42)
        # for the synthetic data:
        self.number_of_attributes = 3
        self.value_cardinality = 4
        #for the query filters
        self.vals_per_attr = 2
        return
    
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
        return
    
    def create_subset(self):
        if not (self.find_subset_path() is None):
            return
        subset_id = self.get_next_subset_id()
        subset_path = os.path.join(self.dataset_path, 'subsets', str(subset_id))

        os.makedirs(subset_path, exist_ok=True)

        base_vecs = np.load(os.path.join(self.dataset_path, 'base', 'vectors.npy'))
        base_attributes = np.load(os.path.join(self.dataset_path, 'base', 'attributes.npy'))
        query_vecs = np.load(os.path.join(self.dataset_path, 'queries', 'vectors.npy'))

        base_size = int(base_vecs.shape[0] * self.subset_size)
        query_size = int(query_vecs.shape[0] * self.subset_size)

        base_ids = np.arange(base_size)
        query_ids = np.arange(query_size)

        os.makedirs(os.path.join(subset_path, 'base'), exist_ok=True)
        os.makedirs(os.path.join(subset_path, 'queries'), exist_ok=True)
        np.save(os.path.join(subset_path, 'base', 'ids.npy'), base_ids)
        np.save(os.path.join(subset_path, 'queries', 'ids.npy'), query_ids)

        # ground truth not implemented yet

        metadata = {
            "subset_size": self.subset_size,
            "neighbors_retrieved": self.neighbors_retrieved
        }

        with open(os.path.join(subset_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
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
        gt_ids = np.load(os.path.join(subset_path, 'queries', 'ground_truth_ids'))
        
        return gt_ids

    def create_synthetic(self, size, n_max, number_of_attributes):
        return self.rng.integers(0, n_max, size=(size, number_of_attributes))
    
    def create_filters(self, size, n_max, num_attributes, values_per_attr):
        return self.rng.integers(
            0,
            n_max,
            size=(size, num_attributes, values_per_attr),
            dtype=np.int32
        )
    
    
    
class yfccDataset(Dataset):
    def __init__(self, name="YFCC", subset_size=1.0, neighbors_retrieved=10):
        super().__init__(name, subset_size, neighbors_retrieved)

        self.url = "https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/yfcc100M/"
        self.files_to_download = {
            "base.10M.u8bin": "base.10M.u8bin",
            "query.public.100K.u8bin": "query.public.100K.u8bin",
            "base.metadata.10M.spmat": "base.metadata.10M.spmat",
            "query.metadata.public.100K.spmat": "query.metadata.public.100K.spmat"
        }
        
        return
    
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
        base_attributes = load_npz(os.path.join(self.dataset_path, 'base', 'attributes.npz'))
        query_vecs = np.load(os.path.join(self.dataset_path, 'queries', 'vectors.npy'))

        base_size = int(base_vecs.shape[0] * self.subset_size)
        query_size = int(query_vecs.shape[0] * self.subset_size)

        base_ids = np.arange(base_size)
        query_ids = np.arange(query_size)

        os.makedirs(os.path.join(subset_path, 'base'), exist_ok=True)
        os.makedirs(os.path.join(subset_path, 'queries'), exist_ok=True)
        np.save(os.path.join(subset_path, 'base', 'ids.npy'), base_ids)
        np.save(os.path.join(subset_path, 'queries', 'ids.npy'), query_ids)

        # ground truth not implemented yet

        metadata = {
            "subset_size": self.subset_size,
            "neighbors_retrieved": self.neighbors_retrieved
        }

        with open(os.path.join(subset_path, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)
        
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
        gt_ids = np.load(os.path.join(subset_path, 'queries', 'ground_truth_ids'))
        
        return gt_ids

    
    
if __name__ == '__main__':
    """
    dataset = siftDataset()
    dataset.prepare()
    restr_count = 2
    base_vecs = dataset.get_base_vectors()
    base_attributes = dataset.get_base_attributes()
    query_vecs = dataset.get_query_vectors()
    query_filters = dataset.get_query_filters(restr_count)
    
    print("base_vecs-shape: ", base_vecs.shape)
    print("base_attrs-shape: ", base_attributes.shape)
    print("query_vecs-shape: ", query_vecs.shape)
    print("query_filters-shape: ", query_filters.shape)
    
    """
    

    
    dataset = GloVeDataset(subset_size=0.8)
    dataset.prepare()
    base_vecs = dataset.get_base_vectors()
    base_attributes = dataset.get_base_attributes()
    query_vecs = dataset.get_query_vectors()
    query_filters = dataset.get_query_filters()
    
    print("base_vecs-shape: ", base_vecs.shape)
    print("base_attrs-shape: ", base_attributes.shape)
    print("query_vecs-shape: ", query_vecs.shape)
    print("query_filters-shape: ", query_filters.shape)
    
    
    
    

    """
    dataset = yfccDataset()
    dataset.prepare()
    base_vecs = dataset.get_base_vectors()
    base_attributes = dataset.get_base_attributes()
    query_vecs = dataset.get_query_vectors()
    query_filters = dataset.get_query_filters()
    
    print("base_vecs-shape: ", base_vecs.shape)
    print("base_attrs-shape: ", base_attributes.shape)
    print("query_vecs-shape: ", query_vecs.shape)
    print("query_filters-shape: ", query_filters.shape)
    
    """
    
    
    
    