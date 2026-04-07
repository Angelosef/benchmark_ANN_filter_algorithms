import numpy as np
import ParlayANN.python.wrapper as wp
import os
import time
import multiprocessing
import shutil

N = 5000
D = 128
TOTAL_TAGS = 500
K = 10
QUERY_TAG = 0  # The tag we will filter by

# ---------------------------
# 1. Helper: Load .fbin (FLOAT32)
# ---------------------------
def load_fbin(filename):
    with open(filename, 'rb') as f:
        # Read 8-byte header (two int32 for N, D)
        header = np.fromfile(f, dtype='int32', count=2)
        n, d = header
        # Read as float32
        data = np.fromfile(f, dtype='float32').reshape((n, d))
        return data

# ---------------------------
# 2. Helper: Load .spmat for Ground Truth
# ---------------------------
def load_spmat_metadata(filename):
    with open(filename, 'rb') as f:
        nrow, ncol, nnz = np.fromfile(f, dtype='int64', count=3)
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        indices = np.fromfile(f, dtype='int32', count=nnz)
        
        metadata_list = []
        for i in range(nrow):
            tags = indices[indptr[i]:indptr[i+1]]
            metadata_list.append(set(tags))
        return metadata_list


# --- 2. THE MAIN CONTROLLER ---
def run_full_experiment():
    cutoff = 50
    cluster_size = 50
    weight_classes = (200, 400)
    MAX_DEGREES = (4, 8, 16)

    build_params = []
    query_params = []
    for i in range(3):
        build_params.append(wp.BuildParams(MAX_DEGREES[i], 100, 1.1))
        query_params.append(wp.QueryParams(10, 50, 1.2, 1000, MAX_DEGREES[i]))
        
    cache_path = "my_indexes/index_cache/"
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    
    os.makedirs(cache_path, exist_ok=True)
    
    
    print("Main Process: Loading index for Search...")
    # CRITICAL: Changed "uint8" to "float"
    index = wp.init_squared_ivf_index("Euclidian", "float")
    for i in range(3):
        index.set_build_params(build_params[i], i)
        index.set_query_params(query_params[i], i)
        
    index.fit_from_filename("toy_base.fbin", "toy_metadata.spmat", cutoff, cluster_size, cache_path, weight_classes)
    return index

# --- 3. DATA LOADING ---
print("Loading float files for Ground Truth...")
X_base = load_fbin("toy_base.fbin")
X_query = load_fbin("toy_query.fbin")
meta_sets = load_spmat_metadata("toy_metadata.spmat")

# --- 4. CALCULATE GROUND TRUTH ---
print(f"Calculating Ground Truth for Tag {QUERY_TAG}...")
valid_indices = [i for i, tags in enumerate(meta_sets) if QUERY_TAG in tags]

if not valid_indices:
    raise Exception(f"No vectors found with Tag {QUERY_TAG}.")

X_filtered = X_base[valid_indices] # Already float32
q_vec = X_query # Already float32

# Standard Euclidean Squared Distance
diff = X_filtered - q_vec
dist_sq = np.sum(diff**2, axis=1)

sorted_local_idx = np.argsort(dist_sq)[:K]
true_neighbors = set([valid_indices[i] for i in sorted_local_idx])

www = np.zeros((1000, 100), dtype=np.float32)
# --- 5. RUN INDEX ---
index = run_full_experiment()
www = np.zeros((1000, 100), dtype=np.float32)

q_filter = wp.QueryFilter(QUERY_TAG)
neighbors, distances = index.batch_filter_search(X_query, [q_filter], 1, K)

# --- 6. RECALL CALCULATION ---
found_neighbors = set(neighbors[0])
found_neighbors.discard(2147483647) 

intersection = true_neighbors.intersection(found_neighbors)
recall = len(intersection) / K

print("\n" + "="*30)
print(f"FLOAT RESULTS FOR TAG {QUERY_TAG}")
print(f"Valid vectors in base: {len(valid_indices)}")
print(f"True Neighbors:  {sorted(list(true_neighbors))}")
print(f"Index Neighbors: {sorted(list(found_neighbors))}")
print(f"RECALL@{K}: {recall:.2%}")
print("="*30)