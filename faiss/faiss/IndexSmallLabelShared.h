#ifndef INDEX_SMALL_LABEL_SHARED_H
#define INDEX_SMALL_LABEL_SHARED_H

#include <algorithm>
#include <memory>
#include <vector>

#include <faiss/ISharedLabelIndex.h>
#include <faiss/Index.h>
#include <faiss/impl/io.h>

namespace faiss {

class IndexSmallLabelShared : public ISharedLabelIndex {
   public:
    const Index* storage;
    std::vector<idx_t> ids; // Guaranteed to be kept sorted

   public:
    explicit IndexSmallLabelShared(const Index* storage) : storage(storage) {}

    IndexSmallLabelShared() = default;

    void add(const std::vector<idx_t>& new_ids) override;

    // For small labels, ALL IDs are returned (they fit well within n_target)
    std::vector<idx_t> get_candidates(const float* x, size_t n_target)
            const override {
        (void)x;        // Query vector ignored
        (void)n_target; // n_target ignored (returns all IDs)
        return ids;     // Pre-sorted!
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    size_t size() const override {
        return ids.size();
    }
    const std::vector<idx_t>& get_raw_ids() const {
        return ids;
    }
};

void write_small_label_shared(const IndexSmallLabelShared& idx, IOWriter* f);

void read_small_label_shared(
        IndexSmallLabelShared& idx,
        IOReader* f,
        const Index* storage);

} // namespace faiss

#endif