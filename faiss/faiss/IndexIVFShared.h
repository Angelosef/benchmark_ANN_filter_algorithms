#ifndef INDEX_IVF_SHARED_H
#define INDEX_IVF_SHARED_H

#include <memory>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/Index.h>
#include <faiss/invlists/InvertedLists.h>

#include <faiss/impl/io.h>

namespace faiss {

struct IndexIVFShared {
    const Index* storage;             // Shared master vector storage
    std::unique_ptr<Index> quantizer; // Flat index holding coarse centroids
    std::unique_ptr<InvertedLists> invlists; // Inverted lists (code_size = 0)

    size_t d;     // Vector dimension
    size_t nlist; // Number of cluster centroids
    MetricType metric_type;
    bool is_trained = false;
    bool verbose = false;

    // Constructor with auto-created Flat quantizer
    IndexIVFShared(
            const Index* storage,
            size_t nlist,
            MetricType metric = METRIC_L2);

    IndexIVFShared() = default;

    // Constructor with custom quantizer
    IndexIVFShared(
            const Index* storage,
            Index* custom_quantizer,
            size_t nlist,
            MetricType metric = METRIC_L2);

    ~IndexIVFShared() = default;

    // 1. Train cluster centroids using raw float sample vectors
    void train(
            idx_t n,
            const float* x,
            const ClusteringParameters& cp = ClusteringParameters());

    // Helper: Train centroids by sampling vector IDs directly from master
    // storage
    void train_from_storage(
            const std::vector<idx_t>& sample_global_ids,
            const ClusteringParameters& cp = ClusteringParameters());

    // 2. Add global IDs to inverted lists (fetches raw vectors from storage
    // zero-copy)
    void add(const std::vector<idx_t>& global_ids);

    // 3. Harvest candidates for IVF^2 joins
    // Probes closest centroids until at least `n_target` vector IDs are
    // collected.
    std::vector<idx_t> get_candidates(const float* x, size_t n_target) const;

    void reset();
};

void write_ivf_shared(const IndexIVFShared& ivf_idx, IOWriter* f);
void read_ivf_shared(
        IndexIVFShared& ivf_idx,
        IOReader* f,
        const Index* storage);

} // namespace faiss

#endif // INDEX_IVF_SHARED_H