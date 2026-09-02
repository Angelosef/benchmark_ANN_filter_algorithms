#include "IndexHNSWShared.h"

#include <omp.h>
#include <algorithm>
#include <atomic>
#include <memory>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
#include <faiss/index_io.h>

void faiss::write_hnsw_shared(
        const faiss::IndexHNSWShared& hnsw_idx,
        faiss::IOWriter* f) {
    // Write graph topology using standard Faiss HNSW serializer
    write_HNSW(&hnsw_idx.graph, f);

    // Write ids vector
    WRITEVECTOR(hnsw_idx.ids);

    // Write hyperparameters
    WRITE1(hnsw_idx.efContrsuction);
    WRITE1(hnsw_idx.M);
}

void faiss::read_hnsw_shared(
        faiss::IndexHNSWShared& hnsw_idx,
        faiss::IOReader* f,
        const faiss::Index* storage) {
    hnsw_idx.storage = storage;

    // Read graph topology using standard Faiss HNSW deserializer
    read_HNSW(hnsw_idx.graph, f);

    // Read ids vector
    READVECTOR(hnsw_idx.ids);

    // Read hyperparameters
    READ1(hnsw_idx.efContrsuction);
    READ1(hnsw_idx.M);
}

namespace faiss {

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
    return storage->get_distance_computer();
}

template <class BlockResultHandler>
void hnsw_shared_search(
        const IndexHNSWShared* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params) {
    FAISS_THROW_IF_NOT_MSG(index->storage, "No storage index provided.");
    const HNSW& graph = index->graph;
    size_t d = index->storage->d;
    size_t ntotal = index->ids.size();

    int efSearch = graph.efSearch;
    if (params) {
        if (const SearchParametersHNSW* hnsw_params =
                    dynamic_cast<const SearchParametersHNSW*>(params)) {
            efSearch = hnsw_params->efSearch;
        }
    }

    idx_t check_period =
            InterruptCallback::get_period_hint(graph.max_level * d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);
        std::exception_ptr ex;
        std::atomic<bool> interrupt{false};

#pragma omp parallel if (i1 - i0 > 1)
        {
            std::unique_ptr<VisitedTable> vt;
            std::unique_ptr<typename BlockResultHandler::SingleResultHandler>
                    res;
            std::unique_ptr<DistanceComputer> base_dis;
            std::unique_ptr<SharedDistanceComputer> dis;
            try {
                vt = VisitedTable::create(ntotal, graph.use_visited_hashset);
                res = std::make_unique<
                        typename BlockResultHandler::SingleResultHandler>(bres);
                base_dis.reset(storage_distance_computer(index->storage));
                dis = std::make_unique<SharedDistanceComputer>(
                        base_dis.get(), index->ids);
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }

#pragma omp for schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                if (interrupt.load(std::memory_order_relaxed)) {
                    continue;
                }
                try {
                    res->begin(i);
                    dis->set_query(x + i * d);

                    graph.search(*dis, nullptr, *res, *vt, params);
                    res->end();
                    vt->advance();
                } catch (...) {
                    omp_capture_exception(ex, [&] { interrupt = true; });
                }
            }
        }
        omp_rethrow_if_exception(ex);
        InterruptCallback::check();
    }
}

} // namespace

IndexHNSWShared::IndexHNSWShared(
        const Index* storage,
        int efConstruction,
        int M)
        : storage(storage), graph(M), efContrsuction(efConstruction), M(M) {
    FAISS_THROW_IF_NOT_MSG(storage, "Storage index cannot be null");
    graph.efConstruction = efConstruction;
    graph.is_similarity = is_similarity_metric(storage->metric_type);
}

void IndexHNSWShared::add(const std::vector<idx_t>& new_ids) {
    size_t n = new_ids.size();
    if (n == 0) {
        return;
    }

    size_t n0 = ids.size();
    size_t ntotal = n0 + n;
    size_t d = storage->d;

    ids.insert(ids.end(), new_ids.begin(), new_ids.end());

    // 1. Generate random levels for new points
    int max_level = graph.prepare_level_tab(n, false);

    // 2. CRITICAL FIX: Allocate offsets & neighbors storage in the HNSW
    // structure
    if (graph.offsets.empty()) {
        graph.offsets.push_back(0);
    }
    for (size_t i = 0; i < n; i++) {
        HNSW::storage_idx_t pt_id = static_cast<HNSW::storage_idx_t>(n0 + i);
        int pt_level = graph.levels[pt_id] - 1;
        graph.offsets.push_back(
                graph.offsets.back() + graph.cum_nb_neighbors(pt_level));
    }
    graph.neighbors.resize(graph.offsets.back());

    LockVector locks;
    locks.prepare(ntotal);

    std::vector<int> hist;
    std::vector<int> order(n);

    // Build level histogram & bucket sort new vertices by graph level
    {
        for (size_t i = 0; i < n; i++) {
            HNSW::storage_idx_t pt_id =
                    static_cast<HNSW::storage_idx_t>(i + n0);
            int pt_level = graph.levels[pt_id] - 1;
            while (pt_level >= static_cast<int>(hist.size())) {
                hist.push_back(0);
            }
            hist[pt_level]++;
        }

        std::vector<int> offsets(hist.size() + 1, 0);
        for (size_t i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        for (size_t i = 0; i < n; i++) {
            HNSW::storage_idx_t pt_id =
                    static_cast<HNSW::storage_idx_t>(i + n0);
            int pt_level = graph.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * d * graph.efConstruction);

    // Add vectors from highest to lowest level
    {
        RandomGenerator rng2(789);
        size_t i1 = n;

        for (int pt_level = static_cast<int>(hist.size()) - 1; pt_level >= 0;
             pt_level--) {
            size_t i0 = i1 - hist[pt_level];

            for (size_t j = i0; j < i1; j++) {
                std::swap(
                        order[j],
                        order[j + rng2.rand_int(static_cast<int>(i1 - j))]);
            }

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                std::unique_ptr<VisitedTable> vt =
                        VisitedTable::create(ntotal, graph.use_visited_hashset);
                std::unique_ptr<DistanceComputer> base_dis(
                        storage_distance_computer(storage));
                SharedDistanceComputer dis(base_dis.get(), ids);
                std::vector<float> vec_buffer(d);

                size_t counter = 0;

#pragma omp for schedule(static)
                for (int64_t i = i0; i < i1; i++) {
                    HNSW::storage_idx_t pt_id = order[i];
                    idx_t global_id = ids[pt_id];

                    // Reconstruct vector from master storage for distance
                    // computation
                    storage->reconstruct(global_id, vec_buffer.data());
                    dis.set_query(vec_buffer.data());

                    if (interrupt) {
                        continue;
                    }

                    graph.add_with_locks(
                            dis, pt_level, pt_id, locks, *vt, (pt_level == 0));

                    // Advance VisitedTable generation counter for thread reuse
                    vt->advance();

                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("HNSW add computation interrupted");
            }
            i1 = i0;
        }
    }
}

void IndexHNSWShared::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);

    if (ids.empty()) {
        std::fill_n(distances, n * k, std::numeric_limits<float>::max());
        std::fill_n(labels, n * k, -1);
        return;
    }

    if (is_similarity_metric(storage->metric_type)) {
        using RH = HeapBlockResultHandler<HNSW::C_similarity>;
        RH bres(n, distances, labels, k);
        hnsw_shared_search(this, n, x, bres, params);
    } else {
        using RH = HeapBlockResultHandler<HNSW::C_distance>;
        RH bres(n, distances, labels, k);
        hnsw_shared_search(this, n, x, bres, params);
    }

    // Remap graph internal vertex IDs (0 ... ntotal-1) back to global storage
    // IDs
    for (idx_t i = 0; i < n * k; ++i) {
        if (labels[i] >= 0 && static_cast<size_t>(labels[i]) < ids.size()) {
            labels[i] = ids[labels[i]];
        }
    }
}

} // namespace faiss