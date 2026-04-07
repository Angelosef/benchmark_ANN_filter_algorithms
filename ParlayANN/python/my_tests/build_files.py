import numpy as np
import os

# ---------------------------
# Save .u8bin (CONFIRMED OK)
# ---------------------------
def save_u8bin(data, filename):
    n, d = data.shape
    with open(filename, 'wb') as f:
        np.array([n, d], dtype='int32').tofile(f)   # correct
        data.astype('uint8').tofile(f)


# ---------------------------
# Save .spmat (CORRECT FORMAT)
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

    indptr = np.array(indptr, dtype='int64')     # REQUIRED
    indices = np.array(flat_indices, dtype='int32')
    data = np.ones(nnz, dtype='float32')

    with open(filename, 'wb') as f:
        np.array([nrow, ncol, nnz], dtype='int64').tofile(f)
        indptr.tofile(f)
        indices.tofile(f)
        data.tofile(f)

    # --- sanity prints ---
    print(f"[spmat] nrow={nrow}, ncol={ncol}, nnz={nnz}")
    print(f"[spmat] avg tags/vector = {nnz / nrow:.2f}")
    print(f"[spmat] max index = {indices.max()}")


# ---------------------------
# CONFIG
# ---------------------------
N = 5000
D = 128
TOTAL_TAGS = 500

np.random.seed(42)  # reproducibility


# ---------------------------
# 1. Create vectors
# ---------------------------
vectors = np.random.randint(0, 256, size=(N, D), dtype='uint8')
save_u8bin(vectors, "toy_base.u8bin")


# ---------------------------
# 2. Create realistic metadata
# ---------------------------

metadata = []
tag_used = set()

for i in range(N):
    # realistic number of tags per vector (IMPORTANT)
    num_tags = np.random.randint(5, 15)

    tags = np.random.choice(
        TOTAL_TAGS,
        size=num_tags,
        replace=False
    ).tolist()

    metadata.append(tags)
    tag_used.update(tags)


# ---------------------------
# Ensure ALL tags appear at least once
# ---------------------------
missing_tags = list(set(range(TOTAL_TAGS)) - tag_used)

if missing_tags:
    print(f"[fix] Adding missing tags: {len(missing_tags)}")

    for i, tag in enumerate(missing_tags):
        metadata[i % N].append(tag)


# ---------------------------
# Save metadata
# ---------------------------
save_spmat(metadata, TOTAL_TAGS, "toy_metadata.spmat")


# ---------------------------
# 3. Create query
# ---------------------------
query = np.random.randint(0, 256, size=(1, D), dtype='uint8')
save_u8bin(query, "toy_query.u8bin")