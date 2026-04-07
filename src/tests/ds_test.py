from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset

import numpy as np

if __name__ == '__main__':
    test_sift = False
    test_glove = False
    test_yfcc = True
    
    if test_sift:
        # --- SIFT ---
        dataset = siftDataset(subset_size=0.1)
        dataset.prepare()
        restr_count = 2
        
        dataset.inspect_data(
            dataset.get_base_vectors(), 
            dataset.get_base_attributes(), 
            dataset.get_query_vectors(), 
            dataset.get_query_filters(restr_count)
        )
        gt_ids = dataset.get_ground_truth_ids(restr_count)
        print("gt_shape: ", gt_ids.shape)

        for restr_count in range(1, 4):
            dataset.calculate_selectivity(restr_count)
            dataset.plot_selectivity(restr_count, relative_counts=False)
            dataset.plot_selectivity(restr_count, relative_counts=True)

    if test_glove:
        # --- GloVe ---
        dataset = GloVeDataset(subset_size=0.1)
        dataset.prepare()
        dataset.inspect_data(
            dataset.get_base_vectors(), 
            dataset.get_base_attributes(), 
            dataset.get_query_vectors(), 
            dataset.get_query_filters()
        )
        gt_ids = dataset.get_ground_truth_ids()
        print("gt_shape: ", gt_ids.shape)

        dataset.calculate_selectivity()
        dataset.plot_selectivity(relative_counts=False)
        dataset.plot_selectivity(relative_counts=True)

    if test_yfcc:    
        # --- YFCC ---
        dataset = yfccDataset(subset_size=0.1)
        dataset.prepare()
        
        dataset.inspect_data(
            dataset.get_base_vectors(), 
            dataset.get_base_attributes(), 
            dataset.get_query_vectors(), 
            dataset.get_query_filters()
        )
        gt_ids = dataset.get_ground_truth_ids()
        print("gt_shape: ", gt_ids.shape)
        
        counts = dataset.calculate_selectivity()
        k = dataset.neighbors_retrieved

        # Find queries that return enough results
        sufficient_indices = np.where(counts >= k)[0]
        insufficient_indices = np.where(counts < k)[0]

        # Calculate statistics
        num_bad = len(insufficient_indices)
        percent_bad = (num_bad / len(counts)) * 100

        print(f"Total Queries: {len(counts)}")
        print(f"Queries with counts >= {k}: {len(sufficient_indices)}")
        print(f"Queries with counts < {k} ('Bad' queries): {num_bad} ({percent_bad:.2f}%)")
        tag_counts = dataset.calculate_tag_counts()
        dataset.calculate_query_tag_counts()

        dataset.plot_selectivity(relative_counts=False)
        dataset.plot_selectivity(relative_counts=True)
        dataset.plot_tag_popularity(relative_counts=False)
        dataset.plot_tag_popularity(relative_counts=True)
        dataset.plot_query_tag_popularity(relative_counts=False)
        dataset.plot_query_tag_popularity(relative_counts=True)
        