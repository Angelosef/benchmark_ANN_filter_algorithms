#include <faiss/IndexSmallLabelShared.h>

#include <faiss/impl/DistanceComputer.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <omp.h>
#include <limits>
#include <queue>

#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>

void faiss::write_small_label_shared(
        const faiss::IndexSmallLabelShared& idx,
        faiss::IOWriter* f) {
    WRITEVECTOR(idx.ids);
}

void faiss::read_small_label_shared(
        faiss::IndexSmallLabelShared& idx,
        faiss::IOReader* f,
        const faiss::Index* storage) {
    READVECTOR(idx.ids);
}

namespace faiss {

void IndexSmallLabelShared::add(const std::vector<idx_t>& new_ids) {
    ids.insert(ids.end(), new_ids.begin(), new_ids.end());
    // Keep internal ID vector sorted and unique
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

void IndexSmallLabelShared::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* /*params*/) const {
    FAISS_THROW_IF_NOT(k > 0);
    size_t d = storage->d;
    size_t num_ids = ids.size();

    if (num_ids == 0) {
        std::fill_n(distances, n * k, std::numeric_limits<float>::max());
        std::fill_n(labels, n * k, -1);
        return;
    }

    bool is_sim = is_similarity_metric(storage->metric_type);

#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dc(storage->get_distance_computer());

#pragma omp for schedule(guided)
        for (idx_t i = 0; i < n; ++i) {
            dc->set_query(x + i * d);

            float* simi = distances + i * k;
            idx_t* idxi = labels + i * k;

            if (is_sim) {
                minheap_heapify(k, simi, idxi);
            } else {
                maxheap_heapify(k, simi, idxi);
            }

            for (size_t j = 0; j < num_ids; ++j) {
                idx_t global_id = ids[j];
                float dis = (*dc)(global_id);

                if (is_sim) {
                    if (dis > simi[0]) {
                        minheap_replace_top(k, simi, idxi, dis, global_id);
                    }
                } else {
                    if (dis < simi[0]) {
                        maxheap_replace_top(k, simi, idxi, dis, global_id);
                    }
                }
            }

            if (is_sim) {
                minheap_reorder(k, simi, idxi);
            } else {
                maxheap_reorder(k, simi, idxi);
            }
        }
    }
}

} // namespace faiss