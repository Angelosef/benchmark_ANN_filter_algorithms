import os
import urllib.request
import numpy as np
import struct
from scipy.sparse import csr_matrix

def download_file(url, dst_path, overwrite=False):
    if os.path.exists(dst_path) and not overwrite:
        print(f"File already exists: {dst_path}. Skipping...")
        return dst_path

    print(f"Downloading from {url}...")
    
    try:
        # urlopen handles both HTTP and FTP automatically
        with urllib.request.urlopen(url) as response:
            # Get file size from headers (works for most HTTP and FTP servers)
            meta = response.info()
            total_size = int(meta.get("Content-Length", 0))
            
            with open(dst_path, 'wb') as f:
                downloaded = 0
                block_size = 8192
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    
                    f.write(buffer)
                    downloaded += len(buffer)
                    
                    # Progress Bar
                    if total_size > 0:
                        done = int(50 * downloaded / total_size)
                        percent = (downloaded / total_size) * 100
                        print(f"\r[{'=' * done}{' ' * (50-done)}] {percent:.1f}%", end="")
                    else:
                        print(f"\rDownloaded: {downloaded/(1024*1024):.1f} MB", end="")

        print(f"\nDownload complete: {dst_path}")
        return dst_path

    except Exception as e:
        print(f"\nFailed to download: {e}")
        if os.path.exists(dst_path):
            os.remove(dst_path)
        return None

def load_fvecs(filename):
        data = np.memmap(filename, dtype=np.float32, mode='r')
        dim = data.view(np.int32)[0]
        vectors = data.reshape(-1, dim + 1)[:, 1:]
        return vectors

def load_vecs_from_txt(filename):
        vectors = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.split()
                # The first token is the word, the rest are the vector components
                vector = np.array(tokens[1:], dtype=np.float32)
                
                vectors.append(vector)
                
        # Convert list of vectors to a single 2D NumPy array
        vectors = np.array(vectors)
        print(f"Done. Loaded {len(vectors)} vectors of dimension {vectors.shape[1]}.")
        
        return vectors

def load_vectors_from_u8bin(filename):
        """
        Reads .u8bin files. 
        Format: 4 bytes (int) for num_vectors, 4 bytes (int) for dims, 
        followed by raw uint8 data.
        """
        
        
        with open(filename, "rb") as f:
            # Read num_vectors (n) and dimensions (d)
            n, d = struct.unpack('ii', f.read(8))
            # Load the rest of the file as uint8
            data = np.fromfile(f, dtype=np.uint8)
            
        return data.reshape(n, d)

def load_metadata(filename):
    """
    Reads the YFCC metadata using the official competition format:
    Header: 3 x int64 (rows, cols, non-zero-elements)
    Pointers: (rows + 1) x int64
    Indices: nnz x int32
    Data: nnz x float32
    """
    
    with open(filename, "rb") as f:
        # 1. Read the 3-element header (24 bytes total)
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        print(f"Metadata Matrix: {nrow} rows, {ncol} columns, {nnz} non-zero entries")

        # 2. Read the row pointers (indptr) as 64-bit
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        
        # Safety check: the last pointer must equal total non-zero elements
        assert nnz == indptr[-1], f"File mismatch: header says {nnz} elements, but indptr ends at {indptr[-1]}"

        # 3. Read the column indices as 32-bit
        indices = np.fromfile(f, dtype='int32', count=nnz)

        # 4. Read the data values as 32-bit floats
        data = np.fromfile(f, dtype='float32', count=nnz)

    # Reconstruct the sparse matrix
    return csr_matrix((data, indices, indptr), shape=(nrow, ncol))