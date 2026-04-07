import os
import numpy as np

from constrainedANN import filter_index_py

def get_recall(true_neighbors, found_neighbors):
    """Calculates how many of the true top-K were actually found."""
    if len(true_neighbors) == 0: return 1.0  # Avoid div by zero if no points match filter
    
    # Intersection of the two sets
    found_set = set(found_neighbors)
    correct = sum(1 for idx in true_neighbors if idx in found_set)
    return correct / len(true_neighbors)

def brute_force_filter(queries, q_labels, base_data, base_labels, k):
    """
    Computes exact nearest neighbors for queries, 
    only considering points that match the filters.
    """
    gt_results = []
    for i in range(len(queries)):
        q_vec = queries[i]
        q_l = q_labels[i]
        
        # 1. Filter the base indices based on attributes (handling 'X' as wildcard)
        valid_indices = []
        for idx, b_l in enumerate(base_labels):
            match = True
            for attr_idx in range(len(q_l)):
                if q_l[attr_idx] != "X" and q_l[attr_idx] != b_l[attr_idx]:
                    match = False
                    break
            if match:
                valid_indices.append(idx)
        
        if not valid_indices:
            gt_results.append([])
            continue

        # 2. Calculate distances only for valid points
        valid_data = base_data[valid_indices]
        # L2 Distance: ||a-b||^2
        diff = valid_data - q_vec
        dist_sq = np.sum(diff**2, axis=1)
        
        # 3. Sort and pick top K
        sorted_indices = np.argsort(dist_sq)
        top_k_indices = [valid_indices[idx] for idx in sorted_indices[:k]]
        gt_results.append(top_k_indices)
        
    return gt_results

# --- 1. Prepare data ---
N_BASE = 5000
DIM = 64
K = 10
data = np.random.random((N_BASE, DIM)).astype('float32')

# Create some variety in labels so filters actually do something
labels = []
for i in range(N_BASE):
    cat = str(np.random.randint(1, 4))
    loc = "indoor" if i % 2 == 0 else "outdoor"
    time = "day" if i % 3 == 0 else "night"
    labels.append([cat, loc, time])

# --- 2. Initialize and Build Index ---
index = filter_index_py.FilterIndex(data, 100, labels, "kmeans", 1)
os.makedirs("indices/test", exist_ok=True)
index.get_index("L2", "indices/test", 1) 
index.loadIndex("indices/test")           

# --- 3. Prepare Queries ---
N_QUERY = 10
queries = np.random.random((N_QUERY, DIM)).astype('float32')
# Query for Category '1', wildcard Location, and 'day' time
q_labels = [["1", "X", "day"] for _ in range(N_QUERY)]

# --- 4. Run C++ ANN Search ---
# Note: max_num_distances (300) controls the search budget/speed
results = index.query(queries, q_labels, K, 300)

# --- 5. Run Brute Force for Ground Truth ---
print("Calculating Ground Truth via Brute Force...")
gt_results = brute_force_filter(queries, q_labels, data, labels, K)

# --- 6. Calculate and Print Recall ---
recalls = []
for i in range(N_QUERY):
    r = get_recall(gt_results[i], results[i])
    recalls.append(r)
    print(f"Query {i}: Recall = {r:.2f}")

print(f"\nAverage Recall: {np.mean(recalls):.4f}")
