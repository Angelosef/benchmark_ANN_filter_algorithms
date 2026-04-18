from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset

if __name__ == '__main__':
    test_sift = True
    test_glove = True
    test_yfcc = True
    
    if test_sift:
        # --- SIFT ---
        dataset = siftDataset(subset_size=1.0)
        dataset.prepare()

    if test_glove:
        # --- GloVe ---
        dataset = GloVeDataset(subset_size=1.0)
        dataset.prepare()

    if test_yfcc:    
        # --- YFCC ---
        dataset = yfccDataset(subset_size=0.1)
        dataset.prepare()
        