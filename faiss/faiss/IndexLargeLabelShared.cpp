#include <algorithm>
#include <memory>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexLargeLabelShared.h>

#include <faiss/IndexHNSWShared.h>
#include <faiss/IndexIVFShared.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

void faiss::write_large_label_shared(
        const faiss::IndexLargeLabelShared& idx,
        faiss::IOWriter* f) {
    WRITE1(idx.ntotal);
    faiss::write_hnsw_shared(idx.hnsw_index, f);
    faiss::write_ivf_shared(idx.ivf_index, f);
}

void faiss::read_large_label_shared(
        faiss::IndexLargeLabelShared& idx,
        faiss::IOReader* f,
        const faiss::Index* storage) {
    READ1(idx.ntotal);
    faiss::read_hnsw_shared(idx.hnsw_index, f, storage);
    faiss::read_ivf_shared(idx.ivf_index, f, storage);
}

namespace faiss {

IndexLargeLabelShared::IndexLargeLabelShared(
        const Index* storage,
        size_t nlist,
        int efConstruction,
        int M)
        : storage(storage),
          hnsw_index(storage, efConstruction, M),
          ivf_index(storage, nlist, storage->metric_type) {}

// Train the internal IVF quantizer before adding vectors
void IndexLargeLabelShared::train_from_storage(
        const std::vector<idx_t>& sample_ids) {
    ivf_index.train_from_storage(sample_ids);
}

void IndexLargeLabelShared::add(const std::vector<idx_t>& new_ids) {
    hnsw_index.add(new_ids);
    ivf_index.add(new_ids);
    ntotal += new_ids.size();
}

// Delegates candidate harvesting to IVF centroids
std::vector<idx_t> IndexLargeLabelShared::get_candidates(
        const float* x,
        size_t n_target) const {
    std::vector<idx_t> cands = ivf_index.get_candidates(x, n_target);
    std::sort(
            cands.begin(),
            cands.end()); // Ensure sorted output for join sweep
    return cands;
}

// Delegates single-label fast spatial search to HNSW graph
void IndexLargeLabelShared::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    hnsw_index.search(n, x, k, distances, labels, params);
}

size_t IndexLargeLabelShared::size() const {
    return ntotal;
}

} // namespace faiss
