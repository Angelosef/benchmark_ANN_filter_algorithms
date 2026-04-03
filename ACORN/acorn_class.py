import numpy as np
from ACORN import acorn

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
    
