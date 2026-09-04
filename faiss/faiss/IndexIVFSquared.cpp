#include <faiss/IndexIVFSquared.h>

#include <faiss/IndexLargeLabelShared.h>
#include <faiss/IndexSmallLabelShared.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
#include <faiss/index_io.h>
#include <faiss/utils/Heap.h>

#include <omp.h>
#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <iostream>

namespace faiss {

enum LabelIndexType : uint8_t { TYPE_LARGE_LABEL = 1, TYPE_SMALL_LABEL = 2 };

IndexIVFSquared::IndexIVFSquared(
        int dimensions,
        int cut_off,
        int cluster_size,
        int cut_off_bitvector,
        int efConstruction,
        int M,
        MetricType metric)
        : Index(dimensions, metric),
          storage(dimensions, metric),
          cut_off(cut_off),
          cluster_size(cluster_size),
          cut_off_bitvector(cut_off_bitvector),
          efConstruction(efConstruction),
          M(M) {}

void IndexIVFSquared::writeToFile(
        const std::string& index_file,
        const std::string& dataset_file) const {
    // Step A: Write Index state & sub-indexes to index_file
    {
        faiss::FileIOWriter writer(index_file.c_str());
        faiss::IOWriter* f = &writer;

        // 1. Base Index Metadata
        WRITE1(this->d);
        WRITE1(this->ntotal);
        WRITE1(this->verbose);
        WRITE1(this->is_trained);
        WRITE1(this->metric_type);

        // 2. IndexIVFSquared Hyperparameters
        WRITE1(this->cut_off);
        WRITE1(this->cluster_size);
        WRITE1(this->cut_off_bitvector);
        WRITE1(this->efConstruction);
        WRITE1(this->M);

        // 3. Serialize label_indexes (Polymorphic vector)
        size_t n_labels = this->label_indexes.size();
        WRITE1(n_labels);
        for (size_t i = 0; i < n_labels; ++i) {
            const auto& label_idx = this->label_indexes[i];
            if (auto* large = dynamic_cast<const IndexLargeLabelShared*>(
                        label_idx.get())) {
                uint8_t type = TYPE_LARGE_LABEL;
                WRITE1(type);
                write_large_label_shared(*large, f);
            } else if (
                    auto* small = dynamic_cast<const IndexSmallLabelShared*>(
                            label_idx.get())) {
                uint8_t type = TYPE_SMALL_LABEL;
                WRITE1(type);
                write_small_label_shared(*small, f);
            } else {
                FAISS_THROW_MSG(
                        "Unknown ISharedLabelIndex implementation type");
            }
        }

        // 4. Serialize membership_bitvector (vector<vector<bool>>)
        size_t n_bv = this->membership_bitvector.size();
        WRITE1(n_bv);
        for (size_t i = 0; i < n_bv; ++i) {
            // Convert std::vector<bool> to std::vector<uint8_t> for clean
            // binary output
            const auto& bv = this->membership_bitvector[i];
            size_t bv_sz = bv.size();
            WRITE1(bv_sz);
            std::vector<uint8_t> buf(bv.begin(), bv.end());
            WRITEVECTOR(buf);
        }

        // 5. Serialize two_label_indexes (std::unordered_map)
        size_t n_map = this->two_label_indexes.size();
        WRITE1(n_map);
        for (const auto& [pair, hnsw_ptr] : this->two_label_indexes) {
            WRITE1(pair.first);  // idx_t 1
            WRITE1(pair.second); // idx_t 2
            FAISS_THROW_IF_NOT_MSG(
                    hnsw_ptr, "IndexHNSWShared inside map cannot be null");
            write_hnsw_shared(*hnsw_ptr, f);
        }
    }

    // Step B: Write raw vectors (storage) to dataset_file
    if (!dataset_file.empty()) {
        faiss::FileIOWriter dataset_writer(dataset_file.c_str());
        faiss::write_index(&this->storage, &dataset_writer);
    }
}

IndexIVFSquared::IndexIVFSquared(
        const std::string& index_file,
        const std::string& dataset_file) {
    // Step A: Read raw vector dataset first into storage
    if (!dataset_file.empty()) {
        faiss::FileIOReader dataset_reader(dataset_file.c_str());
        std::unique_ptr<faiss::Index> loaded_storage(
                faiss::read_index(&dataset_reader));

        auto* loaded_flat =
                dynamic_cast<faiss::IndexFlat*>(loaded_storage.get());
        FAISS_THROW_IF_NOT_MSG(
                loaded_flat,
                "dataset_file does not contain a valid IndexFlat.");

        this->storage = std::move(*loaded_flat);
    }

    // Step B: Read index state and sub-indexes from index_file
    {
        faiss::FileIOReader reader(index_file.c_str());
        faiss::IOReader* f = &reader;

        // 1. Base Index Metadata
        READ1(this->d);
        READ1(this->ntotal);
        READ1(this->verbose);
        READ1(this->is_trained);
        READ1(this->metric_type);

        // 2. IndexIVFSquared Hyperparameters
        READ1(this->cut_off);
        READ1(this->cluster_size);
        READ1(this->cut_off_bitvector);
        READ1(this->efConstruction);
        READ1(this->M);

        // 3. Deserialize label_indexes
        size_t n_labels = 0;
        READ1(n_labels);
        this->label_indexes.resize(n_labels);

        for (size_t i = 0; i < n_labels; ++i) {
            uint8_t type = 0;
            READ1(type);

            if (type == TYPE_LARGE_LABEL) {
                auto large = std::make_unique<IndexLargeLabelShared>();
                read_large_label_shared(*large, f, &this->storage);
                this->label_indexes[i] = std::move(large);
            } else if (type == TYPE_SMALL_LABEL) {
                auto small = std::make_unique<IndexSmallLabelShared>();
                read_small_label_shared(*small, f, &this->storage);
                this->label_indexes[i] = std::move(small);
            } else {
                FAISS_THROW_MSG("Unknown label index type tag encountered.");
            }
        }

        // 4. Deserialize membership_bitvector
        size_t n_bv = 0;
        READ1(n_bv);
        this->membership_bitvector.resize(n_bv);

        for (size_t i = 0; i < n_bv; ++i) {
            size_t bv_sz = 0;
            READ1(bv_sz);
            std::vector<uint8_t> buf;
            READVECTOR(buf);
            this->membership_bitvector[i].assign(buf.begin(), buf.end());
        }

        // 5. Deserialize two_label_indexes
        size_t n_map = 0;
        READ1(n_map);
        this->two_label_indexes.clear();
        this->two_label_indexes.reserve(n_map);

        for (size_t i = 0; i < n_map; ++i) {
            std::pair<idx_t, idx_t> key;
            READ1(key.first);
            READ1(key.second);

            auto hnsw_ptr = std::make_shared<IndexHNSWShared>();
            read_hnsw_shared(*hnsw_ptr, f, &this->storage);

            this->two_label_indexes[key] = std::move(hnsw_ptr);
        }
    }
}

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
// Helper 4: Dual-Tag Handler (Simplified with check_membership)
// ============================================================================
void IndexIVFSquared::search_dual_tag(
        const float* q,
        idx_t tag1,
        idx_t tag2,
        idx_t k,
        size_t n_target,
        const SearchParametersHNSW& hnsw_params,
        int cut_off_tiny,
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

    // ------------------------------------------------------------------------
    // 1. Bitvector Join: Case where one label is especially small (< Ctiny)
    // ------------------------------------------------------------------------
    if (size1 < static_cast<size_t>(cut_off_tiny) ||
        size2 < static_cast<size_t>(cut_off_tiny)) {
        idx_t tiny_tag = (size1 <= size2) ? tag1 : tag2;
        idx_t large_tag = (size1 <= size2) ? tag2 : tag1;

        // Small Label Join Candidates: Get ALL points for small label (n_target
        // = 0)
        std::vector<idx_t> tiny_cands =
                this->label_indexes[tiny_tag]->get_candidates(q, 0);

        std::vector<idx_t> common_ids;
        common_ids.reserve(tiny_cands.size());

        // Delegate O(1) bitvector lookups (or binary search fallback) to
        // check_membership
        for (idx_t id : tiny_cands) {
            if (check_membership(large_tag, id, q)) {
                common_ids.push_back(id);
            }
        }

        rerank_candidates(q, common_ids, k, simi, idxi);
        return;
    }

    // ------------------------------------------------------------------------
    // 2. Dedicated Joint Pair Index
    // ------------------------------------------------------------------------
    auto pair_it = this->two_label_indexes.find({tag1, tag2});
    if (pair_it != this->two_label_indexes.end()) {
        pair_it->second->search(1, q, k, simi, idxi, &hnsw_params);
        return;
    }

    // ------------------------------------------------------------------------
    // 3. Large Label Candidate Intersection (General AND Filter Case)
    // ------------------------------------------------------------------------
    std::vector<idx_t> cands1 =
            this->label_indexes[tag1]->get_candidates(q, n_target);
    std::vector<idx_t> cands2 =
            this->label_indexes[tag2]->get_candidates(q, n_target);

    std::vector<idx_t> common_ids;
    std::set_intersection(
            cands1.begin(),
            cands1.end(),
            cands2.begin(),
            cands2.end(),
            std::back_inserter(common_ids));

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
    int cut_off_tiny = ivf_params ? ivf_params->cut_off_tiny : 200;

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
                    q,
                    tag1,
                    tag2,
                    k,
                    n_target,
                    hnsw_params,
                    cut_off_tiny,
                    simi,
                    idxi);
        }
    }
    // std::cout << "finished processing query\n\n";
}
} // namespace faiss
