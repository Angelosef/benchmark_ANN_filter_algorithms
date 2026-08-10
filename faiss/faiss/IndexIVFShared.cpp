#include "IndexIVFShared.h"

#include <algorithm>
#include <memory>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>

namespace faiss {

IndexIVFShared::IndexIVFShared(
        const Index* storage,
        size_t nlist,
        MetricType metric)
        : storage(storage), d(storage->d), nlist(nlist), metric_type(metric) {
    FAISS_THROW_IF_NOT_MSG(storage, "Master storage pointer cannot be null");

    if (metric == METRIC_L2) {
        quantizer = std::make_unique<IndexFlatL2>(d);
    } else if (metric == METRIC_INNER_PRODUCT) {
        quantizer = std::make_unique<IndexFlatIP>(d);
    } else {
        FAISS_THROW_MSG("Unsupported metric type for IndexIVFShared");
    }

    // code_size = 0 because we store ONLY vector IDs in the posting lists
    invlists = std::make_unique<ArrayInvertedLists>(nlist, 0);
}

IndexIVFShared::IndexIVFShared(
        const Index* storage,
        Index* custom_quantizer,
        size_t nlist,
        MetricType metric)
        : storage(storage),
          quantizer(custom_quantizer),
          d(storage->d),
          nlist(nlist),
          metric_type(metric) {
    FAISS_THROW_IF_NOT_MSG(storage, "Master storage pointer cannot be null");
    FAISS_THROW_IF_NOT_MSG(
            custom_quantizer, "Quantizer pointer cannot be null");

    invlists = std::make_unique<ArrayInvertedLists>(nlist, 0);
}

void IndexIVFShared::train(
        idx_t n,
        const float* x,
        const ClusteringParameters& cp) {
    FAISS_THROW_IF_NOT_MSG(n >= nlist, "Training points must be >= nlist");

    Clustering clus(d, nlist, cp);
    clus.verbose = verbose;

    // Run k-means clustering to compute the centroids and store them in
    // quantizer
    clus.train(n, x, *quantizer);
    is_trained = true;
}

void IndexIVFShared::train_from_storage(
        const std::vector<idx_t>& sample_global_ids,
        const ClusteringParameters& cp) {
    size_t n = sample_global_ids.size();
    FAISS_THROW_IF_NOT_MSG(n >= nlist, "Sample size must be >= nlist");

    std::vector<float> temp_vecs(n * d);
    for (size_t i = 0; i < n; ++i) {
        storage->reconstruct(sample_global_ids[i], temp_vecs.data() + i * d);
    }

    train(n, temp_vecs.data(), cp);
}

void IndexIVFShared::add(const std::vector<idx_t>& global_ids) {
    FAISS_THROW_IF_NOT_MSG(
            is_trained, "IndexIVFShared must be trained before adding IDs");
    size_t n = global_ids.size();
    if (n == 0) {
        return;
    }

    // Process in batches to keep buffer size under control
    constexpr size_t batch_size = 16384;
    std::vector<float> temp_vecs;
    std::vector<idx_t> assign(batch_size);

    for (size_t i0 = 0; i0 < n; i0 += batch_size) {
        size_t i1 = std::min(i0 + batch_size, n);
        size_t cur_n = i1 - i0;

        temp_vecs.resize(cur_n * d);

        // Reconstruct raw vectors from master storage
        for (size_t i = 0; i < cur_n; ++i) {
            storage->reconstruct(global_ids[i0 + i], temp_vecs.data() + i * d);
        }

        // Find nearest centroids via quantizer
        quantizer->assign(cur_n, temp_vecs.data(), assign.data());

        // Append global IDs into the inverted lists (code = nullptr since
        // code_size = 0)
        for (size_t i = 0; i < cur_n; ++i) {
            idx_t list_no = assign[i];
            idx_t global_id = global_ids[i0 + i];
            invlists->add_entries(list_no, 1, &global_id, nullptr);
        }
    }
}

std::vector<idx_t> IndexIVFShared::get_candidates(
        const float* x,
        size_t n_target) const {
    FAISS_THROW_IF_NOT_MSG(is_trained, "IndexIVFShared is not trained");

    std::vector<idx_t> candidates;

    // Probing all centroids when n_target == 0 (unlimited)
    size_t target =
            (n_target == 0) ? std::numeric_limits<size_t>::max() : n_target;
    candidates.reserve(
            target == std::numeric_limits<size_t>::max() ? 4096 : target * 2);

    // Rank centroids by distance to query x
    std::vector<float> centroid_dists(nlist);
    std::vector<idx_t> centroid_ids(nlist);

    quantizer->search(1, x, nlist, centroid_dists.data(), centroid_ids.data());

    // Iterate through closest centroids
    for (size_t i = 0; i < nlist; ++i) {
        idx_t list_no = centroid_ids[i];
        if (list_no < 0) {
            continue;
        }

        size_t list_size = invlists->list_size(list_no);
        if (list_size == 0) {
            continue;
        }

        const idx_t* ids = invlists->get_ids(list_no);
        candidates.insert(candidates.end(), ids, ids + list_size);
        invlists->release_ids(list_no, ids);

        if (candidates.size() >= target) {
            break;
        }
    }

    std::sort(candidates.begin(), candidates.end());
    return candidates;
}

void IndexIVFShared::reset() {
    invlists->reset();
}

} // namespace faiss