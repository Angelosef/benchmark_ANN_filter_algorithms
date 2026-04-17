import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict
import mmh3

"""
class AttributeIndex:
    def __init__(self, attributes):
        
        attributes: np.ndarray (nb x num_dims) 
        
        self.nb = attributes.shape[0]
        self.num_dims = attributes.shape[1]
        self.inverted_index = self._build_index(attributes)

    def _build_index(self, attributes):
        inverted = defaultdict(set)
        for idx, attr_vec in enumerate(attributes):
            for dim, val in enumerate(attr_vec):
                inverted[(dim, val)].add(idx)
        return inverted

    def get_valid_ids_conj(self, filter):
        
        filter: 1D array/list [val_dim0, val_dim1, ...]
        Logic: AND between dimensions. -1 is a wildcard.
        
        valid_ids = None

        for dim_idx, required_val in enumerate(filter):
            if required_val == -1:
                continue
            
            key = (dim_idx, required_val)
            ids_for_this_dim = self.inverted_index.get(key, set())

            if valid_ids is None:
                valid_ids = ids_for_this_dim.copy()
            else:
                valid_ids.intersection_update(ids_for_this_dim)
            
            if not valid_ids:
                return np.array([], dtype='int64')

        if valid_ids is None:
            return np.arange(self.nb, dtype='int64')

        return np.array(list(valid_ids), dtype='int64')

    def get_valid_ids_cnf(self, filter):
        
        filter: List of lists [[vals_dim0], [vals_dim1], ...]
        Logic: OR between values in a dimension, AND between dimensions.
        
        valid_ids = None

        for dim_idx, valid_vals in enumerate(filter):
            dim_union = set()
            is_wildcard_dim = False
            
            for val in valid_vals:
                if val == -1: 
                    is_wildcard_dim = True
                    break
                key = (dim_idx, val)
                if key in self.inverted_index:
                    dim_union.update(self.inverted_index[key])
            
            if is_wildcard_dim:
                continue

            if valid_ids is None:
                valid_ids = dim_union
            else:
                valid_ids.intersection_update(dim_union)
            
            if not valid_ids:
                return np.array([], dtype='int64')

        if valid_ids is None:
            return np.arange(self.nb, dtype='int64')

        return np.array(list(valid_ids), dtype='int64')

"""

#-----------------

class AttributeIndex:
    def __init__(self, attributes):
        """
        attributes: np.ndarray (nb x num_dims)
        """
        self.nb, self.num_dims = attributes.shape
        self.index = self._build_index(attributes)

    def _build_index(self, attributes):
        index = [defaultdict(list) for _ in range(self.num_dims)]

        # collect indices
        for i in range(self.nb):
            for d in range(self.num_dims):
                val = attributes[i, d]
                index[d][val].append(i)

        # convert lists → compact numpy arrays
        for d in range(self.num_dims):
            for val in index[d]:
                index[d][val] = np.array(index[d][val], dtype=np.int64)

        return index
    
    def get_valid_ids_conj(self, flt):
        result = None

        for d, val in enumerate(flt):
            if val == -1:
                continue

            ids = self.index[d].get(val)
            if ids is None:
                return np.empty(0, dtype=np.int64)

            if result is None:
                result = ids
            else:
                result = np.intersect1d(result, ids, assume_unique=True)

            if result.size == 0:
                return result

        if result is None:
            return np.arange(self.nb, dtype=np.int64)

        return result
    
    def get_valid_ids_cnf(self, flt):
        result = None

        for d, vals in enumerate(flt):
            if -1 in vals:
                continue

            # union inside dimension
            arrays = []
            for v in vals:
                ids = self.index[d].get(v)
                if ids is not None:
                    arrays.append(ids)

            if not arrays:
                return np.empty(0, dtype=np.int64)

            dim_union = np.unique(np.concatenate(arrays))

            if result is None:
                result = dim_union
            else:
                result = np.intersect1d(result, dim_union, assume_unique=True)

            if result.size == 0:
                return result

        if result is None:
            return np.arange(self.nb, dtype=np.int64)

        return result

#----------------

class BitsetAttributeIndex:
    def __init__(self, attributes):
        """
        attributes: np.ndarray (nb x num_dims)
        """
        self.nb, self.num_dims = attributes.shape
        self.index = self._build_index(attributes)

    def _build_index(self, attributes):
        index = [defaultdict(lambda: np.zeros(self.nb, dtype=bool))
                 for _ in range(self.num_dims)]

        for i in range(self.nb):
            for d in range(self.num_dims):
                val = attributes[i, d]
                index[d][val][i] = True

        return index
    
    def get_valid_ids_conj(self, flt):
        mask = np.ones(self.nb, dtype=bool)

        for d, val in enumerate(flt):
            if val == -1:
                continue

            bitset = self.index[d].get(val)
            if bitset is None:
                return np.empty(0, dtype=np.int64)

            mask &= bitset

            if not mask.any():
                return np.empty(0, dtype=np.int64)

        return np.flatnonzero(mask).astype(np.int64)
    
    def get_valid_ids_cnf(self, flt):
        mask = np.ones(self.nb, dtype=bool)

        for d, vals in enumerate(flt):
            if -1 in vals:
                continue

            dim_mask = np.zeros(self.nb, dtype=bool)

            for v in vals:
                bitset = self.index[d].get(v)
                if bitset is not None:
                    dim_mask |= bitset

            if not dim_mask.any():
                return np.empty(0, dtype=np.int64)

            mask &= dim_mask

            if not mask.any():
                return np.empty(0, dtype=np.int64)

        return np.flatnonzero(mask).astype(np.int64)

# --------------
def save_fbin(data, filename):
    """Saves a numpy array (N, D) to .fbin format."""
    n, d = data.shape
    with open(filename, 'wb') as f:
        # Header: N, D as int32
        np.array([n, d], dtype='int32').tofile(f)
        # Data: float32
        data.astype('float32').tofile(f)

def save_csr_to_spmat(csr_matrix, filename):
    """
    Saves a scipy.sparse.csr_matrix to ParlayANN .spmat format.
    csr_matrix: The attributes matrix (Rows=Vectors, Cols=Tags)
    """
    nrow, ncol = csr_matrix.shape
    nnz = csr_matrix.nnz
    
    # Extract CSR components
    # ParlayANN expects:
    # indptr: int64
    # indices: int32
    # data: float32 (usually just ones for binary tags)
    indptr = csr_matrix.indptr.astype('int64')
    indices = csr_matrix.indices.astype('int32')
    data = csr_matrix.data.astype('float32') if csr_matrix.data.size > 0 else np.ones(nnz, dtype='float32')

    with open(filename, 'wb') as f:
        # Header: nrow, ncol, nnz as int64
        np.array([nrow, ncol, nnz], dtype='int64').tofile(f)
        indptr.tofile(f)
        indices.tofile(f)
        data.tofile(f)

def valid_structured_conjunction(attribute_vector, filter):
    for i in range(len(attribute_vector)):
        if filter[i] != -1 and attribute_vector[i] != filter[i]:
            return False

    return True

def valid_structured_CNF(attribute_vector, filter):
    for i in range(len(attribute_vector)):
        if attribute_vector[i] not in filter[i]:
            return False

    return True

def valid_csr_conjunction(base_attributes: csr_matrix, row_idx: int, filter_indices: np.ndarray):
    start = base_attributes.indptr[row_idx]
    end = base_attributes.indptr[row_idx + 1]
    
    row_tags = base_attributes.indices[start:end]
    
    positions = np.searchsorted(row_tags, filter_indices)
    
    if (positions < len(row_tags)).all() and np.all(row_tags[positions] == filter_indices):
        return True
    return False

def intersect_sorted_lists(lists):
    if not lists:
        return []

    # start from smallest list for efficiency
    lists = sorted(lists, key=len)
    result = set(lists[0])

    for lst in lists[1:]:
        result.intersection_update(lst)

        if not result:
            break

    return list(result)
 

class HybridBloomEncoder:
    def __init__(self, m_bits, n_head, k_tail):
        self.m_bits = m_bits
        self.n_head = n_head
        self.k_tail = k_tail
        
        self.head_tag_map = {}
        self.head_set = set()

    def _identify_head(self, csr_matrix):
        """Identifies the top N most frequent tags."""
        tag_counts = np.asarray(csr_matrix.sum(axis=0)).flatten()
        popular_indices = np.argsort(tag_counts)[-self.n_head:]
        
        for col_idx, tag_id in enumerate(popular_indices):
            self.head_tag_map[tag_id] = col_idx
            self.head_set.add(tag_id)

    def encode_csr(self, csr_matrix):
        num_rows = csr_matrix.shape[0]
        self._identify_head(csr_matrix)
        
        bloom_storage = np.zeros((num_rows, self.m_bits), dtype=np.int8)
        
        tail_start = self.n_head
        tail_width = self.m_bits - self.n_head
        
        for row_idx in range(num_rows):
            start, end = csr_matrix.indptr[row_idx], csr_matrix.indptr[row_idx+1]
            tags = csr_matrix.indices[start:end]
            
            for tag in tags:
                if tag in self.head_set:
                    col = self.head_tag_map[tag]
                    bloom_storage[row_idx, col] = 1
                else:
                    for i in range(self.k_tail):
                        idx = mmh3.hash(str(tag), i) % tail_width
                        bloom_storage[row_idx, tail_start + idx] = 1
                        
        return bloom_storage

    def create_query_indices(self, tag_ids):
        """Returns the specific bit-columns that must be 1 for these tags."""
        all_indices = set()
        tail_start = self.n_head
        tail_width = self.m_bits - self.n_head
        
        for tag in tag_ids:
            if tag in self.head_set:
                all_indices.add(self.head_tag_map[tag])
            else:
                for i in range(self.k_tail):
                    idx = mmh3.hash(str(tag), i) % tail_width
                    all_indices.add(tail_start + idx)
                    
        return list(all_indices)

class TagAssigner:
    def __init__(self, base_dataset, num_bins):
        self.num_bins = num_bins
        # Calculate empirical probabilities from the dataset
        self.probs = self.calculate_probs(base_dataset)
        # Perform the optimization
        self.assignment = self.assign_tags(self.probs, self.num_bins)

    @staticmethod
    def calculate_probs(base_dataset):
        num_elems = base_dataset.shape[0]
        # Sum of columns gives frequency of each tag
        tag_counts = np.asarray(base_dataset.sum(axis=0)).flatten()
        return tag_counts / num_elems

    @staticmethod
    def assign_tags(probs, num_bins):
        num_tags = len(probs)
        
        eps = 1e-12
        p_safe = np.clip(probs, eps, 1 - eps)
        
        log_not_p = np.log(1 - p_safe)
        p_ratio = p_safe / (1 - p_safe)
        
        bin_sum_log_not_p = np.zeros(num_bins)
        bin_sum_p_inv_p = np.zeros(num_bins)
        
        assignment = np.zeros(num_tags, dtype=np.int32)
        
        sorted_indices = np.argsort(probs)[::-1]
        
        for tag_idx in sorted_indices:
            l_np = log_not_p[tag_idx]
            p_r = p_ratio[tag_idx]
            
            future_scores = (bin_sum_log_not_p + l_np) + np.log(1 + bin_sum_p_inv_p + p_r)
            best_bin = np.argmax(future_scores)
            
            bin_sum_log_not_p[best_bin] += l_np
            bin_sum_p_inv_p[best_bin] += p_r
            assignment[tag_idx] = best_bin
            
        return assignment
    
    def get_assignment(self):
        return self.assignment
    
    def get_num_bins(self):
        return self.num_bins

class TagEncoder:
    def __init__(self, tag_assignment, num_bins):
        
        self.tag_assignment = tag_assignment
        self.num_bins = num_bins
        return
    
    def get_encoded_data(self, base_csr_matrix):
        num_elems = base_csr_matrix.shape[0]
        encoded_data = np.zeros((num_elems, self.num_bins), dtype=np.int64)
        
        coo = base_csr_matrix.tocoo()
        
        for i, tag_idx in zip(coo.row, coo.col):
            bin_idx = self.tag_assignment[tag_idx]
            encoded_data[i, bin_idx] = tag_idx + 1
        
        return encoded_data
    
    def get_encoded_queries(self, filters_csr):
        num_queries = filters_csr.shape[0]
        encoded_queries = np.full((num_queries, self.num_bins), -1, dtype=np.int64)
        for q_idx in range(num_queries):
            f_start = filters_csr.indptr[q_idx]
            f_end = filters_csr.indptr[q_idx+1]
            required_tags = filters_csr.indices[f_start:f_end]

            for tag_idx in required_tags:
                bin_idx = self.tag_assignment[tag_idx]
                encoded_queries[q_idx][bin_idx] = tag_idx + 1
        
        return encoded_queries

