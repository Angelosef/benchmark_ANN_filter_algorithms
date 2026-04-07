
# doesnt work - need to load from disk

import sys

sys.path.append('python')

import numpy as np
import wrapper as wp
import _ParlayANNpy as pann
from scipy.sparse import csr_matrix

def load_u8bin_to_numpy(filename):
    """Loads .u8bin file into a NumPy array, skipping the 8-byte (int32 x 2) header"""
    with open(filename, "rb") as f:
        # Read header: N and D as int32 (matching your save_u8bin logic)
        header = np.fromfile(f, dtype='int32', count=2)
        n, d = header
        # Read the actual vector data
        data = np.fromfile(f, dtype='uint8')
        return data.reshape((n, d))

def load_spmat_to_csr(filename):
    """Loads .spmat file into a SciPy CSR matrix"""
    with open(filename, "rb") as f:
        # Header: nrow, ncol, nnz (int64)
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        # CSR Fields
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        indices = np.fromfile(f, dtype='int32', count=nnz)
        data = np.fromfile(f, dtype='float32', count=nnz)
    
    return csr_matrix((data, indices, indptr), shape=(nrow, ncol))

# --- 1. Load Data into Memory ---
print("Loading data into NumPy/SciPy...")
X = load_u8bin_to_numpy("toy_base.u8bin")
# Note: The C++ index needs a 'csr_filters' object, not a raw SciPy matrix.
# However, the bindings currently only allow initializing csr_filters from a FILE.
filters_obj = pann.csr_filters("toy_metadata.spmat")

# --- 2. Initialize the Index ---
# We use the wrapper to get the correct class instance
index = wp.init_squared_ivf_index("Euclidian", "uint8")

# --- 3. Configuration ---
CUTOFF = 50
CLUSTER_SIZE = 50
WEIGHT_CLASSES = (200, 400)
MAX_DEGREES = (4, 8, 16)

# Set build parameters (required before fit)
for i in range(3):
    index.set_build_params(wp.BuildParams(MAX_DEGREES[i], 500, 1.175), i)

# --- 4. Fit using the 'fit' method (Memory pointers) ---
print("Building index from memory objects...")
# The .fit method takes (numpy_array, csr_filters_object, cutoff, cluster_size)
index.fit(X, filters_obj, CUTOFF, CLUSTER_SIZE)

print("Index built successfully from memory!")