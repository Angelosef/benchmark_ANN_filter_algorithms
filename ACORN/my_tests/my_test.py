import acorn
import numpy as np
import time

# 1. Setup Parameters
d = 128
nb = 20000
nq = 20
k = 10
num_categories = 500  # 500 categories means only 0.2% of data passes each filter (Hard!)

# Generate base data and queries
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# True Metadata: Categorical attributes
true_metadata = np.random.randint(0, num_categories, size=nb).astype('int32')
# Dummy Metadata: All zeros (The index thinks everything is the same category)
dummy_metadata = np.zeros(nb, dtype='int32')

# Target categories for queries
target_cats = np.random.randint(0, num_categories, size=nq).astype('int32')

# 2. Build the Indices
print("Building True Metadata Index...")
idx_true = acorn.ACORNIndex(d, 32, 12, list(true_metadata), 32)
idx_true.add(xb)

print("Building Dummy Metadata Index...")
idx_dummy = acorn.ACORNIndex(d, 32, 12, list(dummy_metadata), 32)
idx_dummy.add(xb)

# 3. Define the Search Filter
# Important: We use the SAME filter map for both. 
# This filter map is based on the REAL categories.
filter_mask = (target_cats[:, None] == true_metadata[None, :])
filter_flat = filter_mask.astype('int8').ravel()

# 4. Calculate Ground Truth (Brute Force)
print("Calculating Ground Truth...")
gt_labels = []
for i in range(nq):
    # Only consider vectors that pass the filter
    allowed_indices = np.where(filter_mask[i])[0]
    if len(allowed_indices) == 0:
        gt_labels.append([])
        continue
        
    # Manual distance calculation for the allowed subset
    sub_xb = xb[allowed_indices]
    dist = np.linalg.norm(sub_xb - xq[i], axis=1)
    # Get top-k nearest among the allowed ones
    nearest = np.argsort(dist)[:k]
    gt_labels.append(allowed_indices[nearest])

# 5. Run Recall Comparison
def evaluate(index, name):
    index.efSearch = 64
    start = time.time()
    dist, labels = index.search(xq, k, filter_flat)
    end = time.time()
    
    recall_scores = []
    for i in range(nq):
        if len(gt_labels[i]) == 0: continue
        intersection = np.intersect1d(labels[i], gt_labels[i])
        recall_scores.append(len(intersection) / len(gt_labels[i]))
    
    print(f"\nResults for {name}:")
    print(f"  Avg Recall: {np.mean(recall_scores):.4f}")
    print(f"  Search Time: {end - start:.4f}s")



evaluate(idx_true, "ACORN (True Metadata)")
evaluate(idx_dummy, "ACORN (Dummy Metadata)")
