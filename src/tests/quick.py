import numpy as np

import os

sel = np.load("data/GIST/subsets/1/analysis/selectivity.npy")

for i in range(21):
    perc = i*5

    print("sel perc ", perc, " ", np.percentile(sel, perc))
