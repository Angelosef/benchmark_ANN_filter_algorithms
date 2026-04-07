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
# 1. Helper: Load .u8bin 
# ---------------------------
def load_u8bin(filename):
    with open(filename, 'rb') as f:
        # Read 8-byte header (two int32 for N, D)
        header = np.fromfile(f, dtype='int32', count=2)
        n, d = header
        data = np.fromfile(f, dtype='uint8').reshape((n, d))
        return data

# ---------------------------
# 2. Helper: Load .spmat for Ground Truth
# ---------------------------
def load_spmat_metadata(filename):
    with open(filename, 'rb') as f:
        # Header: nrow, ncol, nnz (int64)
        nrow, ncol, nnz = np.fromfile(f, dtype='int64', count=3)
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        indices = np.fromfile(f, dtype='int32', count=nnz)
        # We don't strictly need 'data' for the filter logic, but we skip it
        # _ = np.fromfile(f, dtype='float32', count=nnz)
        
        # Convert to a list of sets for fast lookup
        metadata_list = []
        for i in range(nrow):
            tags = indices[indptr[i]:indptr[i+1]]
            metadata_list.append(set(tags))
        return metadata_list

# --- 1. THE ISOLATED BUILDER ---
def isolated_build_task(base_file, meta_file, cache_dir, build_params, query_params, cutoff, cluster_size, weight_classes):
    """This runs in a clean, fresh memory environment"""
    print("[Worker] Initializing Index for Build...")
    index = wp.init_squared_ivf_index("Euclidian", "uint8")
    
    # Set params for build
    for i in range(3):
        index.set_build_params(build_params[i], i)
        index.set_query_params(query_params[i], i)
    
    print("[Worker] Starting heavy fit_from_filename...")
    print("Building index...")
    index.fit_from_filename(
        base_file, 
        meta_file, 
        cutoff, 
        cluster_size, 
        cache_dir, 
        weight_classes
    )

    print("[Worker] Build Complete. Process exiting to release all memory.")

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
        
    # 3. Build the Index
    
    cache_path = "my_indexes/index_cache/"
    if os.path.exists(cache_path):
        shutil.rmtree(cache_path)
    
    # STEP A: Build in isolation if cache is empty
    if not os.path.exists(cache_path):
        os.makedirs(cache_path, exist_ok=True)
        p = multiprocessing.Process(
            target=isolated_build_task, 
            args=("toy_base.u8bin", "toy_metadata.spmat", cache_path, build_params, query_params
                  , cutoff, cluster_size, weight_classes)
        )
        p.start()
        p.join() # Wait for the 'dirty' memory process to die completely
    else:
        print("Cache detected. Skipping isolated build.")

    # STEP B: Use the index normally in the main process
    print("Main Process: Loading index for Search...")
    index = wp.init_squared_ivf_index("Euclidian", "uint8")
    for i in range(3):
        index.set_build_params(build_params[i], i)
        index.set_query_params(query_params[i], i)
    # This call is now lightning fast because it just reads the cache
    index.fit_from_filename("toy_base.u8bin", "toy_metadata.spmat", cutoff, cluster_size, cache_path, weight_classes)

    # Now you can use GT calculation and batch_filter_search normally
    # ... your Ground Truth logic here ...
    print("Ready for search!")
    return index

# --- 3. DATA LOADING ---
print("Loading files for Ground Truth...")
X_base = load_u8bin("toy_base.u8bin")
X_query = load_u8bin("toy_query.u8bin")
meta_sets = load_spmat_metadata("toy_metadata.spmat")

# --- 4. CALCULATE GROUND TRUTH (Brute Force) ---
print(f"Calculating Ground Truth for Tag {QUERY_TAG}...")
# Find all vectors that actually contain the QUERY_TAG
valid_indices = [i for i, tags in enumerate(meta_sets) if QUERY_TAG in tags]

if not valid_indices:
    raise Exception(f"No vectors found with Tag {QUERY_TAG}. Change QUERY_TAG.")

# Calculate Euclidean distances only for those valid vectors
# (a-b)^2 -> use float32 to avoid uint8 overflow during subtraction
X_filtered = X_base[valid_indices].astype('float32')
q_vec = X_query.astype('float32')

diff = X_filtered - q_vec
dist_sq = np.sum(diff**2, axis=1)

# Get top K indices
sorted_local_idx = np.argsort(dist_sq)[:K]
true_neighbors = set([valid_indices[i] for i in sorted_local_idx])

# 2. Configure Parameters (Tiny values for our tiny dataset)
index = run_full_experiment()


q_filter = wp.QueryFilter(QUERY_TAG)
neighbors, distances = index.batch_filter_search(X_query, [q_filter], 1, K)

# --- 6. RECALL CALCULATION ---
found_neighbors = set(neighbors[0])
found_neighbors.discard(2147483647) # Remove padding if index found < K

intersection = true_neighbors.intersection(found_neighbors)
recall = len(intersection) / K

print("\n" + "="*30)
print(f"RESULTS FOR TAG {QUERY_TAG}")
print(f"Valid vectors in base: {len(valid_indices)}")
print(f"True Neighbors:  {sorted(list(true_neighbors))}")
print(f"Index Neighbors: {sorted(list(found_neighbors))}")
print(f"RECALL@{K}: {recall:.2%}")
print("="*30)
