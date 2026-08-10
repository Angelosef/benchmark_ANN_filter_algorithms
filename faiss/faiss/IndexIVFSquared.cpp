#include <faiss/IndexIVFSquared.h>

#include <faiss/IndexLargeLabelShared.h>
#include <faiss/IndexSmallLabelShared.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>

#include <omp.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <iostream>

namespace faiss {

IndexIVFSquared::IndexIVFSquared(
        int dimensions,
        int cut_off,
        int cluster_size,
        int cut_off_tiny,
        int cut_off_bitvector,
        int efConstruction,
        int M,
        MetricType metric)
        : Index(dimensions, metric),
          storage(dimensions, metric),
          cut_off(cut_off),
          cluster_size(cluster_size),
          cut_off_tiny(cut_off_tiny),
          cut_off_bitvector(cut_off_bitvector),
          efConstruction(efConstruction),
          M(M) {}

void IndexIVFSquared::add_tags_c(
        idx_t n,
        const float* x,
        idx_t num_tags,
        idx_t* tag_flat_array,
        const size_t* tag_offsets) {
    this->storage.add(n, x);
    this->ntotal = this->storage.ntotal;
    this->label_indexes.resize(num_tags);
    this->membership_bitvector.resize(num_tags);

    std::vector<std::vector<idx_t>> sorted_label_ids(num_tags);

#pragma omp parallel for schedule(dynamic, 1)
    for (idx_t label = 0; label < num_tags; label++) {
        size_t start = tag_offsets[label];
        size_t end = tag_offsets[label + 1];
        size_t label_count = end - start;

        // Extract global point IDs belonging to this label
        std::vector<idx_t> label_ids(
                tag_flat_array + start, tag_flat_array + end);

        // Keep IDs sorted for set operations
        std::sort(label_ids.begin(), label_ids.end());
        sorted_label_ids[label] = label_ids;

        // Construct Index based on tag frequency
        if (label_count < static_cast<size_t>(this->cut_off)) {
            // Small label: Brute-force index with pre-sorted IDs
            auto small_idx =
                    std::make_unique<IndexSmallLabelShared>(&this->storage);
            small_idx->add(label_ids);
            this->label_indexes[label] = std::move(small_idx);
        } else {
            // Large label: HNSW graph + IVF shared centroids
            size_t nlist = std::max<size_t>(
                    1, label_count / static_cast<size_t>(this->cluster_size));

            auto large_idx = std::make_unique<IndexLargeLabelShared>(
                    &this->storage, nlist, this->efConstruction, this->M);

            large_idx->train_from_storage(label_ids);
            large_idx->add(label_ids);
            this->label_indexes[label] = std::move(large_idx);
        }

        // Build membership bitvector if label exceeds bitvector threshold
        if (label_count >= static_cast<size_t>(this->cut_off_bitvector)) {
            this->membership_bitvector[label].assign(this->ntotal, false);
            for (idx_t global_id : label_ids) {
                if (global_id < this->ntotal) {
                    this->membership_bitvector[label][global_id] = true;
                }
            }
        }
    }

    std::vector<idx_t> large_label_indices;
    large_label_indices.reserve(num_tags);

    for (idx_t label = 0; label < num_tags; label++) {
        if (sorted_label_ids[label].size() >=
            static_cast<size_t>(this->cut_off)) {
            large_label_indices.push_back(label);
        }
    }

    size_t num_large = large_label_indices.size();

    for (size_t i = 0; i < num_large; i++) {
        idx_t label1 = large_label_indices[i];

        for (size_t j = i + 1; j < num_large; j++) {
            idx_t label2 = large_label_indices[j];

            // Compute exact intersection set between label1 and label2
            std::vector<idx_t> intersection;

            std::set_intersection(
                    sorted_label_ids[label1].begin(),
                    sorted_label_ids[label1].end(),
                    sorted_label_ids[label2].begin(),
                    sorted_label_ids[label2].end(),
                    std::back_inserter(intersection));

            // Create dedicated index only if co-occurrence count meets cutoff
            if (intersection.size() >= static_cast<size_t>(this->cut_off)) {
                size_t nlist = std::max<size_t>(
                        1,
                        intersection.size() /
                                static_cast<size_t>(this->cluster_size));

                auto pair_idx = std::make_unique<IndexHNSWShared>(
                        &this->storage, this->efConstruction, this->M);

                pair_idx->add(intersection);
                this->two_label_indexes[{label1, label2}] = std::move(pair_idx);
            }
        }
    }
}

void IndexIVFSquared::reset() {
    this->storage.reset();
    this->label_indexes.clear();
    this->membership_bitvector.clear();
    this->two_label_indexes.clear();
    this->ntotal = 0;
}

void IndexIVFSquared::reconstruct(idx_t key, float* recons) const {
    this->storage.reconstruct(key, recons);
}

bool IndexIVFSquared::check_membership(
        idx_t tag,
        idx_t global_id,
        const float* query_vec) const {
    if (tag < 0 || tag >= static_cast<idx_t>(this->label_indexes.size()) ||
        !this->label_indexes[tag]) {
        return false;
    }

    // Fast O(1) path via bitvector if available
    if (tag < static_cast<idx_t>(this->membership_bitvector.size()) &&
        !this->membership_bitvector[tag].empty()) {
        return global_id <
                static_cast<idx_t>(this->membership_bitvector[tag].size()) &&
                this->membership_bitvector[tag][global_id];
    }

    // Fallback: Candidate fetch and binary search
    auto cands = this->label_indexes[tag]->get_candidates(query_vec, 0);

    return std::binary_search(cands.begin(), cands.end(), global_id);
}

// ============================================================================
// Helper 2: Heap Re-ranking
// ============================================================================
void IndexIVFSquared::rerank_candidates(
        const float* query_vec,
        const std::vector<idx_t>& cand_ids,
        idx_t k,
        float* simi,
        idx_t* idxi) const {
    bool is_sim = is_similarity_metric(this->metric_type);

    // Initialize result arrays with default fallback values
    std::fill_n(
            simi,
            k,
            is_sim ? -std::numeric_limits<float>::infinity()
                   : std::numeric_limits<float>::infinity());
    std::fill_n(idxi, k, -1);

    if (cand_ids.empty()) {
        return;
    }

    std::unique_ptr<DistanceComputer> dc(this->storage.get_distance_computer());
    dc->set_query(query_vec);

    if (is_sim) {
        minheap_heapify(k, simi, idxi);
    } else {
        maxheap_heapify(k, simi, idxi);
    }

    for (idx_t id : cand_ids) {
        if (id < 0 || id >= this->ntotal)
            continue;
        float dis = (*dc)(id);

        if (is_sim) {
            if (dis > simi[0]) {
                minheap_replace_top(k, simi, idxi, dis, id);
            }
        } else {
            if (dis < simi[0]) {
                maxheap_replace_top(k, simi, idxi, dis, id);
            }
        }
    }

    if (is_sim) {
        minheap_reorder(k, simi, idxi);
    } else {
        maxheap_reorder(k, simi, idxi);
    }
}

// ============================================================================
// Helper 3: Single-Tag Handler
// ============================================================================
void IndexIVFSquared::search_single_tag(
        const float* q,
        idx_t tag,
        idx_t k,
        const SearchParametersHNSW& hnsw_params,
        float* simi,
        idx_t* idxi) const {
    if (tag >= 0 && tag < static_cast<idx_t>(this->label_indexes.size()) &&
        this->label_indexes[tag]) {
        size_t label_sz = this->label_indexes[tag]->size();

        if (label_sz < static_cast<size_t>(this->cut_off)) {
            // Small / Tiny Label: Brute-force candidates scan
            std::vector<idx_t> cands =
                    this->label_indexes[tag]->get_candidates(q, 0);
            rerank_candidates(q, cands, k, simi, idxi);
        } else {
            // Large Label: Delegate directly to HNSW sub-index
            this->label_indexes[tag]->search(1, q, k, simi, idxi, &hnsw_params);
        }
    } else {
        std::fill_n(simi, k, std::numeric_limits<float>::infinity());
        std::fill_n(idxi, k, -1);
    }
}

// ============================================================================
// Helper 4: Dual-Tag Handler
// ============================================================================
void IndexIVFSquared::search_dual_tag(
        const float* q,
        idx_t tag1,
        idx_t tag2,
        idx_t k,
        size_t n_target,
        const SearchParametersHNSW& hnsw_params,
        float* simi,
        idx_t* idxi) const {
    // Bounds check tags
    if (tag1 < 0 || tag1 >= static_cast<idx_t>(this->label_indexes.size()) ||
        tag2 < 0 || tag2 >= static_cast<idx_t>(this->label_indexes.size()) ||
        !this->label_indexes[tag1] || !this->label_indexes[tag2]) {
        std::fill_n(simi, k, std::numeric_limits<float>::infinity());
        std::fill_n(idxi, k, -1);
        return;
    }

    size_t size1 = this->label_indexes[tag1]->size();
    size_t size2 = this->label_indexes[tag2]->size();

    // Subcase 2a: Tiny Label Intersection
    if (size1 < static_cast<size_t>(this->cut_off_tiny) ||
        size2 < static_cast<size_t>(this->cut_off_tiny)) {
        idx_t tiny_tag = (size1 <= size2) ? tag1 : tag2;
        idx_t other_tag = (size1 <= size2) ? tag2 : tag1;

        std::vector<idx_t> tiny_cands =
                this->label_indexes[tiny_tag]->get_candidates(q, 0);

        std::vector<idx_t> common_ids;
        common_ids.reserve(tiny_cands.size());

        for (idx_t id : tiny_cands) {
            if (check_membership(other_tag, id, q)) {
                common_ids.push_back(id);
            }
        }

        rerank_candidates(q, common_ids, k, simi, idxi);
        return;
    }

    // Subcase 2b: Dedicated Joint Pair Index
    auto pair_it = this->two_label_indexes.find({tag1, tag2});
    if (pair_it != this->two_label_indexes.end()) {
        pair_it->second->search(1, q, k, simi, idxi, &hnsw_params);
        return;
    }

    // Subcase 2c: Dynamic Candidate Intersection & Bitvector Masking
    std::vector<idx_t> common_ids;

    if (tag2 < static_cast<idx_t>(this->membership_bitvector.size()) &&
        !this->membership_bitvector[tag2].empty()) {
        std::vector<idx_t> cands1 =
                this->label_indexes[tag1]->get_candidates(q, n_target);

        for (idx_t id : cands1) {
            if (id < static_cast<idx_t>(
                             this->membership_bitvector[tag2].size()) &&
                this->membership_bitvector[tag2][id]) {
                common_ids.push_back(id);
            }
        }
    } else if (
            tag1 < static_cast<idx_t>(this->membership_bitvector.size()) &&
            !this->membership_bitvector[tag1].empty()) {
        std::vector<idx_t> cands2 =
                this->label_indexes[tag2]->get_candidates(q, n_target);

        for (idx_t id : cands2) {
            if (id < static_cast<idx_t>(
                             this->membership_bitvector[tag1].size()) &&
                this->membership_bitvector[tag1][id]) {
                common_ids.push_back(id);
            }
        }
    } else {
        std::vector<idx_t> cands1 =
                this->label_indexes[tag1]->get_candidates(q, n_target);
        std::vector<idx_t> cands2 =
                this->label_indexes[tag2]->get_candidates(q, n_target);

        std::set_intersection(
                cands1.begin(),
                cands1.end(),
                cands2.begin(),
                cands2.end(),
                std::back_inserter(common_ids));
    }

    rerank_candidates(q, common_ids, k, simi, idxi);
}

// ============================================================================
// Main Public Search Entrypoint
// ============================================================================
void IndexIVFSquared::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    const auto* ivf_params =
            dynamic_cast<const SearchParametersIVFSquared*>(params);

    size_t n_target = ivf_params ? ivf_params->n_target : 1000;
    int efSearch = ivf_params ? ivf_params->efSearch : 64;

    SearchParametersHNSW hnsw_params;
    hnsw_params.efSearch = efSearch;

#pragma omp parallel for schedule(guided)
    for (idx_t i = 0; i < n; ++i) {
        const float* q = x + i * this->d;
        float* simi = distances + i * k;
        idx_t* idxi = labels + i * k;

        idx_t tag1 = NO_TAG;
        idx_t tag2 = NO_TAG;

        if (ivf_params && ivf_params->query_tags) {
            tag1 = ivf_params->query_tags[2 * i];
            tag2 = ivf_params->query_tags[2 * i + 1];
        }

        // Unfiltered Search
        if (tag1 == NO_TAG && tag2 == NO_TAG) {
            this->storage.search(1, q, k, simi, idxi, nullptr);
            continue;
        }

        // Normalize duplicate tags to single tag search
        if (tag2 == tag1) {
            tag2 = NO_TAG;
        }

        // Single Tag Search
        if (tag2 == NO_TAG) {
            // std::cout << "single tag search\n";
            search_single_tag(q, tag1, k, hnsw_params, simi, idxi);
        }
        // Dual Tag Search
        else {
            // std::cout << "dual tag search\n";
            search_dual_tag(
                    q, tag1, tag2, k, n_target, hnsw_params, simi, idxi);
        }
    }
    // std::cout << "finished processing query\n\n";
}
} // namespace faiss
