import numpy as np
import ParlayANN.python.wrapper as wp
import os
import time



# --- CONFIGURATION ---
def foo():
    X = np.zeros((5000, 128))
    # ... save files ...

foo()

# --- 3. INDEX BUILDING ---
print("Building Squared IVF Index (UInt8)...")
index = wp.init_squared_ivf_index("Euclidian", "uint8")

# 2. Configure Parameters (Tiny values for our tiny dataset)
CUTOFF = 50
CLUSTER_SIZE = 50
WEIGHT_CLASSES = (200, 400)
MAX_DEGREES = (4, 8, 16)

for i in range(3):
    index.set_build_params(wp.BuildParams(MAX_DEGREES[i], 100, 1.1), i)
    index.set_query_params(wp.QueryParams(10, 50, 1.2, 1000, MAX_DEGREES[i]), i)

# 3. Build the Index
os.makedirs("my_indexes/index_cache/", exist_ok=True)

print("Building index...")
index.fit_from_filename(
    "toy_base.u8bin", 
    "toy_metadata.spmat", 
    CUTOFF, 
    CLUSTER_SIZE, 
    "my_indexes/index_cache/", 
    WEIGHT_CLASSES
)

