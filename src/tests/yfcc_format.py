

import numpy as np

def inspect_u8bin(filename):
    print(f"\n--- Inspecting {filename} ---")

    with open(filename, "rb") as f:
        h32 = np.fromfile(f, dtype=np.int32, count=2)
        f.seek(0)
        h64 = np.fromfile(f, dtype=np.int64, count=2)

    print("int32 header:", h32)
    print("int64 header:", h64)

    # Try interpreting as int32 header (most likely correct)
    n, d = h32
    print(f"[Assuming int32] n={n}, d={d}")

    # Read some data
    with open(filename, "rb") as f:
        f.seek(8)  # 2 * int32
        data_sample = np.fromfile(f, dtype=np.uint8, count=10)

    print("First 10 data values:", data_sample)

def inspect_spmat(filename):
    print(f"\n--- Inspecting {filename} ---")

    with open(filename, "rb") as f:
        # Read both possible headers
        h32 = np.fromfile(f, dtype=np.int32, count=3)
        f.seek(0)
        h64 = np.fromfile(f, dtype=np.int64, count=3)

    print("int32 header:", h32)
    print("int64 header:", h64)

    # Use int64 (based on original writer)
    nrow, ncol, nnz = h64
    print(f"[Using int64] nrow={nrow}, ncol={ncol}, nnz={nnz}")

    with open(filename, "rb") as f:
        # Skip header
        f.seek(8 * 3)

        # Read indptr
        indptr = np.fromfile(f, dtype=np.int64, count=nrow + 1)
        print("indptr[:10]:", indptr[:10])
        print("indptr[-1]:", indptr[-1])

        # Read indices
        indices = np.fromfile(f, dtype=np.int32, count=nnz)
        print("indices[:10]:", indices[:10])
        print("indices max:", indices.max())

        # Read data
        data = np.fromfile(f, dtype=np.float32, count=nnz)
        print("data[:10]:", data[:10])

bvecs_fn = "data/YFCC/base.10M.u8bin"
bmeta_fn = "data/YFCC/base.metadata.10M.spmat"

inspect_u8bin(bvecs_fn)
inspect_spmat(bmeta_fn)

