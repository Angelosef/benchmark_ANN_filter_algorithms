import numpy as np
import acorn
import time

class IndexACORNFlat:
    def __init__(self, d, M, gamma, metadata_matrix, M_beta=None):
        self.d = d
        self.nb = metadata_matrix.shape[0]
        
        if M_beta is None:
            M_beta = 2 * M * gamma
            
        metadata_matrix = np.ascontiguousarray(metadata_matrix)
        
        dtype = np.dtype((np.void, metadata_matrix.dtype.itemsize * metadata_matrix.shape[1]))
        unique_rows_view = metadata_matrix.view(dtype).ravel()
        _, self.metadata_ids = np.unique(unique_rows_view, return_inverse=True)
        
        self.index = acorn.ACORNIndex(d, M, gamma, list(self.metadata_ids.astype('int32')), M_beta)

    def add(self, base_vectors):
        """Adds vectors to the index."""
        if not base_vectors.flags['C_CONTIGUOUS']:
            base_vectors = np.ascontiguousarray(base_vectors)
        self.index.add(base_vectors.astype('float32'))

    @property
    def efSearch(self):
        return self.index.efSearch

    @efSearch.setter
    def efSearch(self, value):
        self.index.efSearch = value

    def search(self, query_vectors, k, filter_map):
        """
        Generic filtered search.
        
        Args:
            query_vectors (np.ndarray): nq x d matrix.
            k (int): Number of neighbors to return.
            filter_map (np.ndarray): nq x nb boolean/int8 matrix.
            
        Returns:
            distances (nq x k), labels (nq x k)
        """
        nq = query_vectors.shape[0]
        
        if filter_map.shape != (nq, self.nb):
            raise ValueError(f"filter_map must be of shape (nq, nb): ({nq}, {self.nb})")

        # Ensure types and memory layout match C++ expectations
        # C++ expects a flat array of nq * nb chars (int8)
        if not query_vectors.flags['C_CONTIGUOUS']:
            query_vectors = np.ascontiguousarray(query_vectors)
            
        # Convert boolean to int8 (equivalent to char* in C++)
        filter_flat = filter_map.astype('int8').ravel()
        
        distances, labels = self.index.search(
            query_vectors.astype('float32'), 
            k, 
            filter_flat
        )
        
        return distances, labels
    
# --- Using your IndexACORNFlat class ---

if __name__ == '__main__':
    # 1. Setup Parameters (Strict Selectivity)
    nb = 20000
    d = 16
    attr_dim = 3
    attr_card = 30 # Increased cardinality to make filters "Harder"
    k = 10
    nq = 100

    # 2. Generate Data
    meta = np.random.randint(0, attr_card, size=(nb, attr_dim), dtype='int32')
    dummy_meta = np.zeros((nb, attr_dim), dtype='int32') 
    xb = np.random.random((nb, d)).astype('float32')
    xq = np.random.random((nq, d)).astype('float32')

    # 3. Define the Filter
    # Stricter Filter: Only ~0.25% of data will pass
    filter_mask = (meta[:, 0] == 5) & (meta[:, 1] == 10)
    pass_count = np.sum(filter_mask)
    print(f"Filter Selectivity: {pass_count}/{nb} pass ({ (pass_count/nb)*100 :.2f}%)")

    # 4. Ground Truth (Brute Force)
    valid_indices = np.where(filter_mask)[0]
    gt_labels = []
    actual_k = min(k, pass_count)
    print("actual_k: ", actual_k)

    for i in range(nq):
        if pass_count == 0:
            gt_labels.append(np.array([]))
            continue
        sub_xb = xb[valid_indices]
        dists = np.linalg.norm(sub_xb - xq[i], axis=1)
        nearest_in_sub = np.argsort(dists)[:actual_k]
        gt_labels.append(valid_indices[nearest_in_sub])

    # 5. Build and Test
    def run_detailed_test(name, metadata_to_use):
        print(f"\n--- Testing {name} ---")
        idx = IndexACORNFlat(d=d, M=32, gamma=12, metadata_matrix=metadata_to_use)
        idx.add(xb)
        idx.efSearch = 16
        
        batch_filter = np.tile(filter_mask, (nq, 1))

        start_search = time.time()
        distances, labels = idx.search(xq, k=k, filter_map=batch_filter)
        total_time = time.time() - start_search

        # CALCULATE VALIDITY & RECALL
        total_returned_indices = 0
        valid_returned_indices = 0
        recall_scores = []

        for i in range(nq):
            query_labels = labels[i]
            # Filter out padding (-1) if the index found fewer than k neighbors
            found_labels = query_labels[query_labels >= 0]
            
            total_returned_indices += len(found_labels)

            # Check how many actually satisfy the filter
            if len(found_labels) > 0:
                is_valid = filter_mask[found_labels]
                valid_returned_indices += np.sum(is_valid)
            
            # Recall
            if len(gt_labels[i]) > 0:
                intersection = np.intersect1d(found_labels, gt_labels[i])
                recall_scores.append(len(intersection) / len(gt_labels[i]))
            else:
                recall_scores.append(1.0)

        validity_rate = (valid_returned_indices / total_returned_indices * 100) if total_returned_indices > 0 else 0
        avg_recall = np.mean(recall_scores)

        print(f"  Validity Rate: {validity_rate:.2f}% (Should be 100%)")
        print(f"  Avg Recall:    {avg_recall:.4f}")
        print(f"  Avg Latency:   {total_time/nq:.6f}s")
        
        return avg_recall

    # Comparison
    run_detailed_test("ACORN (Real Metadata)", meta)

