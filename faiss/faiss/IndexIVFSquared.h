#ifndef FAISS_INDEX_IVFSQUARED_H
#define FAISS_INDEX_IVFSQUARED_H

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>

#ifndef SWIG
#include <faiss/ISharedLabelIndex.h>
#include <faiss/impl/HNSW.h>

#include <faiss/IndexHNSWShared.h>
#endif

#include <memory>
#include <unordered_map>
#include <vector>

namespace faiss {

constexpr idx_t NO_TAG = -1;

struct SearchParametersIVFSquared : public SearchParameters {
    const idx_t* query_tags = nullptr;
    size_t n_target = 1000;
    int efSearch = 32;

    ~SearchParametersIVFSquared() override = default;
};

#ifndef SWIG
// Hide internal helper structs from SWIG
struct LabelPairHash {
    size_t operator()(const std::pair<idx_t, idx_t>& p) const {
        idx_t low = std::min(p.first, p.second);
        idx_t high = std::max(p.first, p.second);
        return std::hash<idx_t>{}(low) ^
                (std::hash<idx_t>{}(high) + 0x9e3779b9 + (low << 6) +
                 (low >> 2));
    }
};

struct LabelPairEqual {
    bool operator()(
            const std::pair<idx_t, idx_t>& p1,
            const std::pair<idx_t, idx_t>& p2) const {
        return std::min(p1.first, p1.second) == std::min(p2.first, p2.second) &&
                std::max(p1.first, p1.second) == std::max(p2.first, p2.second);
    }
};
#endif

struct IndexIVFSquared : Index {
    IndexFlat storage;

#ifndef SWIG
    // Internal state - Python should not access these directly
    std::vector<std::unique_ptr<ISharedLabelIndex>> label_indexes;
    std::vector<std::vector<bool>> membership_bitvector;
    std::unordered_map<
            std::pair<idx_t, idx_t>,
            std::shared_ptr<IndexHNSWShared>,
            LabelPairHash,
            LabelPairEqual>
            two_label_indexes;
#endif

    int cut_off;
    int cluster_size;
    int cut_off_tiny;
    int cut_off_bitvector;
    int efConstruction;
    int M;

    explicit IndexIVFSquared(
            int dimensions,
            int cut_off,
            int cluster_size,
            int cut_off_tiny,
            int cut_off_bitvector,
            int efConstruction = 128,
            int M = 16,
            MetricType metric = METRIC_L2);

    ~IndexIVFSquared() override = default;

    void add(idx_t n, const float* x) override {
        throw std::runtime_error(
                "cannot use add directly since ivf squared requires tag data.");
    }

    void add_tags_c(
            idx_t n,
            const float* x,
            idx_t num_tags,
            idx_t* tag_flat_array,
            const size_t* tag_offsets);

    void reset() override;

    void reconstruct(idx_t key, float* recons) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

#ifndef SWIG
    // Internal routing methods that depend on HNSW parameters
    bool check_membership(idx_t tag, idx_t global_id, const float* query_vec)
            const;

    void rerank_candidates(
            const float* query_vec,
            const std::vector<idx_t>& cand_ids,
            idx_t k,
            float* simi,
            idx_t* idxi) const;

    void search_single_tag(
            const float* q,
            idx_t tag,
            idx_t k,
            const SearchParametersHNSW& hnsw_params,
            float* simi,
            idx_t* idxi) const;

    void search_dual_tag(
            const float* q,
            idx_t tag1,
            idx_t tag2,
            idx_t k,
            size_t n_target,
            const SearchParametersHNSW& hnsw_params,
            float* simi,
            idx_t* idxi) const;
#endif
};

} // namespace faiss

#endif