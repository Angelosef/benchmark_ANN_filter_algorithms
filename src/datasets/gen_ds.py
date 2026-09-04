from src.datasets.sift import siftDataset
from src.datasets.glove import GloVeDataset
from src.datasets.yfcc import yfccDataset
from src.datasets.gist import gistDataset

if __name__ == '__main__':
    test_sift = False
    test_glove = False
    test_yfcc = False
    test_gist = True
    
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
    
    if test_gist:
        # --- GIST ---
        dataset = gistDataset(subset_size=1.0)
        dataset.prepare()
        