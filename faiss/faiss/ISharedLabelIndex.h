#ifndef I_SHARED_LABEL_INDEX_H
#define I_SHARED_LABEL_INDEX_H

#include <algorithm>
#include <memory>
#include <vector>

#include <faiss/Index.h>

namespace faiss {

// Abstract interface for both Small and Large Label Indexes
struct ISharedLabelIndex {
    virtual ~ISharedLabelIndex() = default;

    // Add new global vector IDs belonging to this tag
    virtual void add(const std::vector<idx_t>& new_ids) = 0;

    // Extract candidate global IDs for set-intersection during joins
    virtual std::vector<idx_t> get_candidates(const float* x, size_t n_target)
            const = 0;

    // Perform vector distance search on a single label
    virtual void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const = 0;

    virtual size_t size() const = 0;
};
} // namespace faiss
#endif
