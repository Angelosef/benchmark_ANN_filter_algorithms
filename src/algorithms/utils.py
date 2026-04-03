import numpy as np
from scipy.sparse import csr_matrix
from collections import defaultdict

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

def build_inverted_attribute_index(attributes):
    inverted = defaultdict(list)
    for idx, attr_vec in enumerate(attributes):
        for dim, val in enumerate(attr_vec):
            inverted[(dim, val)].append(idx)
    return inverted
