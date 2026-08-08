#ifndef INDEX_HNSW_SHARED_H
#define INDEX_HNSW_SHARED_H

#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/Panorama.h>
#include <faiss/impl/hnsw/LockVector.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/DistanceComputer.h>

namespace faiss {

struct SharedDistanceComputer : public DistanceComputer {
    DistanceComputer* base_dc; // non-owning
    const std::vector<idx_t>& ids;

    SharedDistanceComputer(DistanceComputer* dc, const std::vector<idx_t>& ids)
            : base_dc(dc), ids(ids) {}

    void set_query(const float* x) override {
        base_dc->set_query(x);
    }

    float operator()(idx_t i) override {
        assert(i >= 0);
        assert(i < ids.size());
        return (*base_dc)(ids[i]);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        assert(i >= 0 && i < ids.size());
        assert(j >= 0 && j < ids.size());
        return base_dc->symmetric_dis(ids[i], ids[j]);
    }
};

struct IndexHNSWShared {
    const Index* storage;
    HNSW graph;
    std::vector<idx_t> ids;

    int efContrsuction = 128;
    int M = 16;

    IndexHNSWShared(const Index* storage, int efConstruction, int M);
    ~IndexHNSWShared() = default;
    void add(const std::vector<idx_t>& new_ids);
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;
};
} // namespace faiss

#endif