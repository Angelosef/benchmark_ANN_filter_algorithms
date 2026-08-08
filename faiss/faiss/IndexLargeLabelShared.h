#ifndef INDEX_LARGE_LABEL_SHARED_H
#define INDEX_LARGE_LABEL_SHARED_H

#include <algorithm>
#include <memory>
#include <vector>

#include <faiss/ISharedLabelIndex.h>
#include <faiss/Index.h>

#include "IndexHNSWShared.h"
#include "IndexIVFShared.h"

namespace faiss {

class IndexLargeLabelShared : public ISharedLabelIndex {
   private:
    const Index* storage;
    IndexHNSWShared hnsw_index;
    IndexIVFShared ivf_index;
    size_t ntotal = 0;

   public:
    IndexLargeLabelShared(
            const Index* storage,
            size_t nlist = 64,
            int efConstruction = 128,
            int M = 16);

    // Train the internal IVF quantizer before adding vectors
    void train_from_storage(const std::vector<idx_t>& sample_ids);

    void add(const std::vector<idx_t>& new_ids) override;

    // Delegates candidate harvesting to IVF centroids
    std::vector<idx_t> get_candidates(const float* x, size_t n_target)
            const override;

    // Delegates single-label fast spatial search to HNSW graph
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    size_t size() const override;
};

} // namespace faiss

#endif