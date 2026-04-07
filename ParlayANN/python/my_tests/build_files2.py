import numpy as np
import os

# ---------------------------
# Save .fbin (FLOAT32 FORMAT)
# ---------------------------
def save_fbin(data, filename):
    """
    Saves float32 data. 
    Header: [N, D] as int32 (Standard for most ANN benchmarks)
    Data: float32
    """
    n, d = data.shape
    with open(filename, 'wb') as f:
        # Standard fbin header is N and D as 4-byte integers
        np.array([n, d], dtype='int32').tofile(f)
        # Ensure data is strictly float32 (4 bytes per element)
        data.astype('float32').tofile(f)
    print(f"[fbin] Saved {n}x{d} floats to {filename}")

# ---------------------------
# Save .spmat (STAYS THE SAME)
# ---------------------------
def save_spmat(indices_list, num_total_tags, filename):
    nrow = len(indices_list)
    ncol = num_total_tags
    flat_indices = []
    indptr = [0]

    for tags in indices_list:
        flat_indices.extend(sorted(tags))
        indptr.append(len(flat_indices))

    nnz = len(flat_indices)
    indptr = np.array(indptr, dtype='int64')
    indices = np.array(flat_indices, dtype='int32')
    data = np.ones(nnz, dtype='float32')

    with open(filename, 'wb') as f:
        np.array([nrow, ncol, nnz], dtype='int64').tofile(f)
        indptr.tofile(f)
        indices.tofile(f)
        data.tofile(f)

    print(f"[spmat] nrow={nrow}, ncol={ncol}, nnz={nnz}")

# ---------------------------
# CONFIG
# ---------------------------
N = 5000
D = 128
TOTAL_TAGS = 4000

np.random.seed(42)

# ---------------------------
# 1. Create float vectors
# ---------------------------
# Generating random floats between 0.0 and 1.0
# For many ANN benchmarks, floats are normalized or range from -1 to 1.
vectors = np.random.random((N, D)).astype('float32')
save_fbin(vectors, "toy_base.fbin")

# ---------------------------
# 2. Create realistic metadata
# ---------------------------
metadata = []
tag_used = set()

for i in range(N):
    num_tags = np.random.randint(5, 15)
    tags = np.random.choice(TOTAL_TAGS, size=num_tags, replace=False).tolist()
    metadata.append(tags)
    tag_used.update(tags)

# Ensure ALL tags appear at least once
missing_tags = list(set(range(TOTAL_TAGS)) - tag_used)
if missing_tags:
    print("missing tags")

save_spmat(metadata, TOTAL_TAGS, "toy_metadata.spmat")

# ---------------------------
# 3. Create float query
# ---------------------------
query = np.random.random((1, D)).astype('float32')
save_fbin(query, "toy_query.fbin")