import os
import urllib.request
import numpy as np
import struct
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from src.benchmark.benchmarkRunner import BenchmarkRunner
from src.algorithms.brute_force import BruteForceIdFilter
import multiprocessing
from scipy import sparse

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

def plot_selectivity(counts, ds_name, dst_path, relative_counts=False, base_count=None, param_dict=None):
    """
    Plots the distribution of query selectivity sorted from most to least restrictive.
    """
    # Sort counts descending (highest selectivity/most results first)
    sorted_counts = np.sort(counts)[::-1]
    
    # Calculate Y-axis values
    y_values = sorted_counts
    y_label = "Valid Vector Count"
    
    if relative_counts and base_count:
        y_values = (sorted_counts / base_count) * 100
        y_label = "Selectivity (% of Base Dataset)"

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_values)), y_values, linewidth=2, color='#1f77b4')
    plt.fill_between(range(len(y_values)), y_values, alpha=0.3, color='#1f77b4')

    # Add Dataset info to Title
    title = f"Query Selectivity Profile: {ds_name}"
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Sorted Query Index", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add parameter info as a text box if provided
    if param_dict:
        param_text = "\n".join([f"{k}: {v}" for k, v in param_dict.items()])
        plt.gca().text(0.95, 0.95, param_text, transform=plt.gca().transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    prefix = "absolute"
    if relative_counts:
        prefix = 'relative'
    plt.savefig(os.path.join(dst_path, f"{prefix}_selectivity.png"), dpi=300)
    plt.close()

def plot_tag_popularity(counts, ds_name, dst_path, relative_counts=False, base_count=None, param_dict=None):
    """
    Plots the distribution of tag frequency (Power Law / Long Tail).
    Note: For YFCC, use a Log-Log scale to visualize the distribution properly.
    """
    sorted_counts = np.sort(counts)[::-1]
    
    y_values = sorted_counts
    y_label = "Frequency (Number of Documents)"
    
    if relative_counts and base_count:
        y_values = (sorted_counts / base_count) * 100
        y_label = "Tag Frequency (% of Database)"

    plt.figure(figsize=(10, 6))
    
    # Using log-log scale is standard for popularity/zipfian distributions
    plt.loglog(range(1, len(y_values) + 1), y_values, linewidth=2, color='#d62728')
    
    plt.title(f"Tag Popularity Distribution (Log-Log): {ds_name}", fontsize=14)
    plt.xlabel("Tag Rank (Log Scale)", fontsize=12)
    plt.ylabel(y_label + " (Log Scale)", fontsize=12)
    plt.grid(True, which="both", linestyle='--', alpha=0.5)

    if param_dict:
        param_text = "\n".join([f"{k}: {v}" for k, v in param_dict.items()])
        plt.gca().text(0.05, 0.05, param_text, transform=plt.gca().transAxes,
                       verticalalignment='bottom', horizontalalignment='left',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    prefix = "absolute"
    if relative_counts:
        prefix = 'relative'
    plt.savefig(os.path.join(dst_path, f"{prefix}_tag_popularity.png"), dpi=300)
    plt.close()

class dummy:
    def __init__(self):
        pass

def save_ground_truth(dataset, ds_query_param, gt_dst_path, gt_ids_path):
    br = BenchmarkRunner(dataset, ds_query_param, BruteForceIdFilter, dummy(), dummy())
    D, I, metadata = br.run()
    os.makedirs(gt_ids_path, exist_ok=True)
    os.makedirs(gt_dst_path, exist_ok=True)
    np.save(os.path.join(gt_ids_path, 'ground_truth_ids.npy'), I)
    np.save(os.path.join(gt_dst_path, 'distances.npy'), D)

def save_gt_isolated(dataset, ds_query_param, gt_dst_path, gt_ids_path):
    p = multiprocessing.Process(
        target=save_ground_truth,
        args=(dataset, ds_query_param, gt_dst_path, gt_ids_path)
    )
            
    p.start()
    p.join()


def generate_zipfian_tags(n_rows, n_cols, a=1.2, max_prop=0.5):
    """
    Generates a sparse matrix where tag occurrences are independent Binomial trials
    with probabilities following a Zipfian distribution.
    """
    ranks = np.arange(1, n_cols + 1)
    weights = 1.0 / (ranks ** a)
    probs = weights * (max_prop / weights[0])
    
    counts = np.random.binomial(n_rows, probs)
    
    all_rows = []
    all_cols = []
    
    for col_idx, count in enumerate(counts):
        if count > 0:
            selected_rows = np.random.choice(n_rows, size=count, replace=False)
            
            all_rows.append(selected_rows)
            all_cols.append(np.full(count, col_idx))
    
    row_indices = np.concatenate(all_rows)
    col_indices = np.concatenate(all_cols)
    data = np.ones(len(row_indices), dtype=np.int8)
    
    return sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols))

def generate_zipfian_fixed_tags(n_rows, n_cols, a=1.1, p_two_tags=0.5):
    """
    Generates a sparse matrix where tag popularity is Zipfian,
    but each row is strictly constrained to have either 1 or 2 tags.
    
    Args:
        n_rows (int): Number of rows.
        n_cols (int): Number of potential tags.
        a (float): Zipf parameter (higher = more skewed).
        p_two_tags (float): Probability (0 to 1) that a row has 2 tags instead of 1.
    """
    ranks = np.arange(1, n_cols + 1)
    weights = 1.0 / (ranks ** a)
    pmf = weights / weights.sum()
    
    n_tags_per_row = np.random.choice([1, 2], size=n_rows, p=[1 - p_two_tags, p_two_tags])
    
    tag1 = np.random.choice(n_cols, size=n_rows, p=pmf)
    
    mask_two = (n_tags_per_row == 2)
    n_two = mask_two.sum()
    
    tag2 = np.full(n_rows, -1) # Placeholder array
    
    if n_two > 0:
        sampled_tag2 = np.random.choice(n_cols, size=n_two, p=pmf)
        
        collisions = (sampled_tag2 == tag1[mask_two])
        while collisions.any():
            sampled_tag2[collisions] = np.random.choice(n_cols, size=collisions.sum(), p=pmf)
            collisions = (sampled_tag2 == tag1[mask_two])
            
        tag2[mask_two] = sampled_tag2

    rows_with_two = np.where(mask_two)[0]
    
    row_indices = np.concatenate([np.arange(n_rows), rows_with_two])
    col_indices = np.concatenate([tag1, tag2[mask_two]])
    
    data = np.ones(len(row_indices), dtype=np.int8)
    
    return sparse.csr_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols))
