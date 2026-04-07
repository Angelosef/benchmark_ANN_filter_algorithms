
import ParlayANN.python.wrapper as wp
import numpy as np
import os

# 1. Initialize
# Using "Euclidian" and "uint8" to match our generated data
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
    "data/YFCC/base.10M.u8bin", 
    "data/YFCC/base.metadata.10M.spmat", 
    CUTOFF, 
    CLUSTER_SIZE, 
    "my_indexes/index_cache/", 
    WEIGHT_CLASSES
)

# 4. Search
# Let's search for vectors that have Tag 10 AND Tag 20
# (Note: In a random 1000-vec dataset, this might return 0 results; 
# try a single tag if that happens)
q_filter = wp.QueryFilter(0) 
X_query = np.fromfile("toy_query.u8bin", dtype=np.uint8)[8:].reshape((1, 128))

print("Searching...")
neighbors, distances = index.batch_filter_search(X_query, [q_filter], 1, 10)

print("Neighbors:", neighbors)
print("Distances:", distances)

index.print_stats()
