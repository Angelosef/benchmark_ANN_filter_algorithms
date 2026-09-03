import os
import time
import json

class SaveManager:
    def __init__(self, save_dir='saved_indexes'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def prepare_folders(self, index_name, build_params, ds_name, ds_subset_size):
        ds_folder = f'{ds_name}_{ds_subset_size:.4f}'
        full_ds_path = os.path.join(self.save_dir, index_name, ds_folder)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        index_dir = os.path.join(full_ds_path, timestamp)
        ds_file = os.path.join(full_ds_path, 'dataset.faiss')
        index_file = os.path.join(index_dir, 'index.faiss')

        os.makedirs(full_ds_path, exist_ok=True)
        ds_metadata = {
            "timestamp": timestamp,
            "index_name": index_name,
            "dataset_name": ds_name,
            "subset_size": ds_subset_size,
            "ds_file": ds_file
        }
        with open(os.path.join(full_ds_path, "metadata.json"), "w") as f:
            json.dump(ds_metadata, f, indent=4)
        
        os.makedirs(index_dir, exist_ok=True)
        
        # Handle build_params whether it's a dict or an object with __dict__
        params_dict = build_params.__dict__ if hasattr(build_params, '__dict__') else build_params

        index_metadata = {
            "timestamp": timestamp,
            "index_name": index_name,
            "dataset_name": ds_name,
            "subset_size": ds_subset_size,
            "build_params": params_dict,
            "index_file": index_file
        }

        with open(os.path.join(index_dir, "metadata.json"), "w") as f:
            json.dump(index_metadata, f, indent=4)

        return ds_file, index_file

    def find_saved_files(self, index_name, build_params, ds_name, ds_subset_size):
        """
        Searches self.save_dir for matching ds_file and index_file.
        Returns tuple (ds_file, index_file) if found, otherwise (None, None).
        """
        ds_folder = f'{ds_name}_{ds_subset_size:.4f}'
        full_ds_path = os.path.join(self.save_dir, index_name, ds_folder)
        ds_file = os.path.join(full_ds_path, 'dataset.faiss')

        # Check if the dataset folder and the dataset file exist
        if not os.path.exists(full_ds_path) or not os.path.exists(ds_file):
            return None, None

        # Convert target build_params to a dict for reliable comparison
        target_params = build_params.__dict__ if hasattr(build_params, '__dict__') else build_params

        # Search inside timestamp folders for matching build_params
        for item in os.listdir(full_ds_path):
            item_path = os.path.join(full_ds_path, item)
            
            # Check if it's a directory (timestamp folder)
            if os.path.isdir(item_path):
                metadata_path = os.path.join(item_path, "metadata.json")
                index_file = os.path.join(item_path, "index.faiss")

                if os.path.exists(metadata_path) and os.path.exists(index_file):
                    try:
                        with open(metadata_path, "r") as f:
                            metadata = json.load(f)
                            
                        # Compare the saved parameters with target parameters
                        if metadata.get("build_params") == target_params:
                            return ds_file, index_file
                    except (json.JSONDecodeError, OSError):
                        continue  # Skip unreadable or corrupted metadata files

        return ds_file, None