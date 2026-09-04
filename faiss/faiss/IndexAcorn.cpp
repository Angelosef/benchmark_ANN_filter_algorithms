#include <faiss/IndexAcorn.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>

#include <bits/stdc++.h>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_set>

#include <omp.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
#include <faiss/index_io.h>
#include <faiss/impl/HierarchicalGraph.cpp>

bool sample_bernoulli(std::mt19937& rng, int N) {
    // std::bernoulli_distribution takes the probability 'p' of returning true
    std::bernoulli_distribution dist(1.0 / N);

    return dist(rng); // Returns true with probability 1/N, false with (1 - 1/N)
}

namespace faiss {

IndexAcorn::IndexAcorn(
        int dimensions,
        int efConstruction,
        int gamma,
        int M,
        int Mbeta,
        MetricType metric)
        : Index(dimensions, metric),
          storage(dimensions, metric),
          efConstruction(efConstruction),
          gamma(gamma),
          M(M),
          Mbeta(Mbeta) {
    this->graph = std::make_unique<HierarchicalGraph>(
            &(this->storage), M, Mbeta, gamma);
    this->rng.seed(42);
    this->ml = 1 / std::log(this->M);
}

IndexAcorn::IndexAcorn(
        const std::string& index_file,
        const std::string& dataset_file) {
    // ---------------------------------------------------------------------
    // Step 1: Read Graph topology & Hyperparameters from index_file
    // ---------------------------------------------------------------------
    {
        faiss::FileIOReader reader(index_file.c_str());
        faiss::IOReader* f = &reader;

        READ1(this->d);
        READ1(this->ntotal);
        READ1(this->verbose);
        READ1(this->is_trained);
        READ1(this->metric_type);

        READ1(this->efConstruction);
        READ1(this->gamma);
        READ1(this->M);
        READ1(this->Mbeta);
        READ1(this->ml);

        this->graph = std::make_unique<HierarchicalGraph>(
                &this->storage, this->M, this->Mbeta, this->gamma);
        read_hierarchical_graph(*this->graph, f);
    }

    // ---------------------------------------------------------------------
    // Step 2: Load storage dataset from dataset_file safely
    // ---------------------------------------------------------------------
    if (!dataset_file.empty()) {
        faiss::FileIOReader dataset_reader(dataset_file.c_str());

        std::unique_ptr<faiss::Index> loaded_storage(
                faiss::read_index(&dataset_reader));

        auto* loaded_flat =
                dynamic_cast<faiss::IndexFlat*>(loaded_storage.get());
        FAISS_THROW_IF_NOT_MSG(
                loaded_flat,
                "Dataset file does not contain a valid IndexFlat.");

        // Copy underlying attributes safely without object slicing
        this->storage.d = loaded_flat->d;
        this->storage.ntotal = loaded_flat->ntotal;
        this->storage.verbose = loaded_flat->verbose;
        this->storage.is_trained = loaded_flat->is_trained;
        this->storage.metric_type = loaded_flat->metric_type;
        this->storage.code_size = loaded_flat->code_size;

        // Swap vector payload buffer directly
        this->storage.codes = std::move(loaded_flat->codes);

        // Keep root index properties aligned with loaded storage
        this->ntotal = loaded_flat->ntotal;
    }
}

void IndexAcorn::writeToFile(
        const std::string& index_file,
        const std::string& dataset_file) const {
    // ---------------------------------------------------------------------
    // Step 1: Write Graph topology & Hyperparameters to index_file
    // ---------------------------------------------------------------------
    {
        faiss::FileIOWriter writer(index_file.c_str());
        faiss::IOWriter* f = &writer;

        WRITE1(this->d);
        WRITE1(this->ntotal);
        WRITE1(this->verbose);
        WRITE1(this->is_trained);
        WRITE1(this->metric_type);

        WRITE1(this->efConstruction);
        WRITE1(this->gamma);
        WRITE1(this->M);
        WRITE1(this->Mbeta);
        WRITE1(this->ml);

        FAISS_THROW_IF_NOT_MSG(this->graph, "Cannot save uninitialized graph.");
        write_hierarchical_graph(*this->graph, f);
    }

    // ---------------------------------------------------------------------
    // Step 2: Write vector payload (IndexFlat) to dataset_file
    // ---------------------------------------------------------------------
    if (!dataset_file.empty()) {
        faiss::FileIOWriter dataset_writer(dataset_file.c_str());
        faiss::write_index(&this->storage, &dataset_writer);
    }
}

void IndexAcorn::reset() {
    this->storage.reset();
    this->ntotal = 0;
    this->graph->clear();
}

void IndexAcorn::reconstruct(idx_t key, float* recons) const {
    this->storage.reconstruct(key, recons);
}

void IndexAcorn::add(idx_t n, const float* x) {
    idx_t offset = this->storage.ntotal;
    this->storage.add(n, x);

#pragma omp parallel
    {
        //   Thread-local RNG to avoid contention/data races on this->rng
        std::mt19937 local_rng(42 + omp_get_thread_num());

#pragma omp for schedule(dynamic, 100)
        for (idx_t i = 0; i < n; i++) {
            idx_t index = offset + i;
            const float* vec = x + i * this->d;

            this->addSingle(index, vec, local_rng);
        }
    }

    std::vector<float> avg_edges = this->graph->avg_num_neighbors();
    /*
    for (int i = 0; i < avg_edges.size(); i++) {
        std::cout << "layer " << i << " avg edge count = " << avg_edges[i]
                  << std::endl;
    }
    this->graph->printEdgePercentiles();
    this->graph->printBidirectionalityStats();
    this->graph->printConnectedComponents();

    */

    this->ntotal = this->storage.ntotal;
}

void IndexAcorn::addSingle(idx_t index, const float* vec, std::mt19937& rng) {
    bool build_phase = true;
    int max_layer = this->graph->getMaxLayerSafe();

    int assigned_layer = this->assignLayer(rng);
    idx_t new_node = this->graph->addNode(assigned_layer, index);

    idx_t entry_node = this->graph->getEntryPoint();

    for (int layer = max_layer; layer > assigned_layer; layer -= 1) {
        std::vector<idx_t> results =
                this->searchLayerSafe(layer, entry_node, vec, 1, 1);
        entry_node = this->graph->getDownwardsNodeSafe(layer, results[0]);
    }
    for (int layer = this->graph->getMaxLayerSafe(); layer > max_layer;
         layer -= 1) {
        new_node = this->graph->getDownwardsNodeSafe(layer, new_node);
    }

    for (int layer = std::min(max_layer, assigned_layer); layer > -1;
         layer -= 1) {
        int active_ef = this->efConstruction;

        std::vector<idx_t> results = this->searchLayerSafe(
                layer, entry_node, vec, active_ef, this->M * this->gamma);
        entry_node = this->graph->getDownwardsNodeSafe(layer, results[0]);

        // add edges
        if (layer > 0) {
            this->graph->addInitialEdges(layer, new_node, results);

            new_node = this->graph->getDownwardsNodeSafe(layer, new_node);
        } else if (layer == 0) {
            this->graph->addInitialBottomEdges(new_node, results);
            std::vector<idx_t> new_neighbors =
                    this->graph->getNeighborsSafe(layer, new_node);
            for (int i = 0; i < new_neighbors.size(); i++) {
                // bool prune_node = sample_bernoulli(rng, this->M);
                bool prune_node =
                        this->graph->getNeighborsSafe(layer, new_neighbors[i])
                                .size() >= this->M * this->gamma;
                if (prune_node) {
                    this->graph->twoHopPruning(layer, new_neighbors[i]);
                }
            }
        }
    }
}

int IndexAcorn::assignLayer(std::mt19937& rng) const {
    double uniform_variable = static_cast<double>(rng() - rng.min()) /
            static_cast<double>(rng.max() - rng.min());

    if (uniform_variable == 0.0) {
        uniform_variable = 1e-9;
    }

    double exp_variable = -this->ml * std::log(uniform_variable);

    return static_cast<int>(exp_variable);
}

std::vector<idx_t> IndexAcorn::searchLayerSafe(
        int layer,
        idx_t entry_node,
        const float* vec,
        int ef,
        int results_size) const {
    std::unique_ptr<faiss::DistanceComputer> dis(
            this->storage.get_distance_computer());

    dis->set_query(vec);

    MinimaxHeapT results(ef);
    MinimaxHeapT final_results(results_size);

    // Min-Heap for greedy search (closest candidates popped first)
    using DistNode = std::pair<float, idx_t>;
    std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>>
            frontier;

    std::unordered_set<idx_t> visited;

    float entry_dist = (*dis)(this->graph->getIndexSafe(layer, entry_node));

    results.push(entry_node, entry_dist);
    final_results.push(entry_node, entry_dist);
    visited.insert(entry_node);
    frontier.push({entry_dist, entry_node});

    while (!frontier.empty()) {
        DistNode frontier_best = frontier.top();
        frontier.pop();

        float frontier_dist = frontier_best.first;
        idx_t frontier_node = frontier_best.second;

        if (results.size() == ef && frontier_dist > results.max()) {
            break;
        }

        const std::vector<idx_t> neighbours =
                this->graph->getNeighborsSafe(layer, frontier_node);

        // Process 1-hop neighbors
        int neighbour_count = 0;
        for (idx_t node : neighbours) {
            if (neighbour_count >= this->M) {
                break;
            }
            if (visited.count(node) == 0) {
                visited.insert(node);
                float dist = (*dis)(this->graph->getIndexSafe(layer, node));
                final_results.push(node, dist);
                if (dist < results.max() || results.size() < ef) {
                    results.push(node, dist);
                    frontier.push({dist, node});
                }
            }
            neighbour_count++;
        }

        // Process 2-hop neighbors for ACORN layer 0
        if (layer == 0) {
            for (size_t i = this->Mbeta; i < neighbours.size(); ++i) {
                if (neighbour_count >= this->M) {
                    break;
                }
                idx_t one_hop_node = neighbours[i];

                // Zero-copy reference to 2-hop neighbors
                const std::vector<idx_t> two_hop_neighbours =
                        this->graph->getNeighborsSafe(layer, one_hop_node);

                for (idx_t node : two_hop_neighbours) {
                    if (neighbour_count >= this->M) {
                        break;
                    }
                    if (visited.count(node) == 0) {
                        visited.insert(node);
                        float dist =
                                (*dis)(this->graph->getIndexSafe(layer, node));
                        final_results.push(node, dist);
                        if (dist < results.max() ||
                            results.size() < results_size) {
                            results.push(node, dist);
                            frontier.push({dist, node});
                        }
                    }
                    neighbour_count++;
                }
            }
        }
    }

    std::vector<idx_t> result_nodes;
    result_nodes.reserve(final_results.size());

    // Dummy buffer for pop_min
    float dummy_dist = 0.0f;
    while (final_results.size() > 0) {
        result_nodes.push_back(final_results.pop_min(&dummy_dist));
    }

    return result_nodes;
}

void IndexAcorn::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (params && params->sel) {
        constexpr int MAX_SAMPLES = 10000;
        int sample_size = std::min(MAX_SAMPLES, static_cast<int>(this->ntotal));

        size_t count = 0;

        std::mt19937 local_rng = this->rng;
        std::uniform_int_distribution<idx_t> dist(0, this->ntotal - 1);

        for (int i = 0; i < sample_size; ++i) {
            idx_t random_vec_id = dist(local_rng);
            if (params->sel->is_member(random_vec_id)) {
                count++;
            }
        }
        float selectivity =
                static_cast<float>(count) / static_cast<float>(sample_size);
        if (selectivity < (1.0f / static_cast<float>(this->gamma))) {
            this->storage.search(n, x, k, distances, labels, params);
            return;
        }
    }

#pragma omp for schedule(dynamic, 100)
    for (idx_t i = 0; i < n; i++) {
        this->searchSingle(
                x + i * this->d,
                k,
                distances + i * k, // Corrected batch offset
                labels + i * k,    // Corrected batch offset
                params);
    }
}

std::vector<idx_t> IndexAcorn::searchLayer(
        int layer,
        idx_t entry_node,
        const float* vec,
        int results_size,
        const IDSelector* sel) const {
    std::unique_ptr<faiss::DistanceComputer> dis(
            this->storage.get_distance_computer());

    dis->set_query(vec);

    MinimaxHeapT results(results_size);

    // Min-Heap for greedy search (closest candidates popped first)
    using DistNode = std::pair<float, idx_t>;
    std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>>
            frontier;

    std::unordered_set<idx_t> visited;

    float entry_dist = (*dis)(this->graph->getIndex(layer, entry_node));

    results.push(entry_node, entry_dist);
    visited.insert(entry_node);
    frontier.push({entry_dist, entry_node});

    while (!frontier.empty()) {
        DistNode frontier_best = frontier.top();
        frontier.pop();

        float frontier_dist = frontier_best.first;
        idx_t frontier_node = frontier_best.second;

        if (results.size() == results_size && frontier_dist > results.max()) {
            break;
        }

        const std::vector<idx_t>& neighbours =
                this->graph->getNeighbors(layer, frontier_node);

        // Process 1-hop neighbors
        int neighbour_count = 0;
        for (idx_t node : neighbours) {
            if (neighbour_count >= this->M) {
                break;
            }
            if ((!sel || sel->is_member(this->graph->getIndex(layer, node))) &&
                visited.count(node) == 0) {
                visited.insert(node);
                neighbour_count++;
                float dist = (*dis)(this->graph->getIndex(layer, node));
                if (dist < results.max() || results.size() < results_size) {
                    results.push(node, dist);
                    frontier.push({dist, node});
                }
            }
        }

        // Process 2-hop neighbors for ACORN layer 0
        if (layer == 0) {
            for (size_t i = this->Mbeta; i < neighbours.size(); ++i) {
                if (neighbour_count >= this->M) {
                    break;
                }
                idx_t one_hop_node = neighbours[i];

                // Zero-copy reference to 2-hop neighbors
                const std::vector<idx_t>& two_hop_neighbours =
                        this->graph->getNeighbors(layer, one_hop_node);

                for (idx_t node : two_hop_neighbours) {
                    if (neighbour_count >= this->M) {
                        break;
                    }
                    if ((!sel ||
                         sel->is_member(this->graph->getIndex(layer, node))) &&
                        visited.count(node) == 0) {
                        visited.insert(node);
                        neighbour_count++;
                        float dist = (*dis)(this->graph->getIndex(layer, node));
                        if (dist < results.max() ||
                            results.size() < results_size) {
                            results.push(node, dist);
                            frontier.push({dist, node});
                        }
                    }
                }
            }
        }
    }

    std::vector<idx_t> result_nodes;
    result_nodes.reserve(results.size());

    // Dummy buffer for pop_min
    float dummy_dist = 0.0f;
    while (results.size() > 0) {
        result_nodes.push_back(results.pop_min(&dummy_dist));
    }

    return result_nodes;
}

void IndexAcorn::searchSingle(
        const float* q_vec,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    bool build_phase = false;
    int max_layer = this->graph->getMaxLayer();
    idx_t entry_node = this->graph->getEntryPoint();

    int efSearch = 16;
    if (params) {
        if (auto acorn_params =
                    dynamic_cast<const SearchParametersAcorn*>(params)) {
            efSearch = acorn_params->efSearch;
        }
    }

    IDSelector* selector = nullptr;
    if (params && params->sel) {
        selector = params->sel;
    }

    for (int layer = max_layer; layer > 0; layer -= 1) {
        std::vector<idx_t> results =
                this->searchLayer(layer, entry_node, q_vec, 1, selector);
        entry_node = this->graph->getDownwardsNode(layer, results[0]);
    }

    std::vector<idx_t> final_candidates =
            this->searchLayer(0, entry_node, q_vec, efSearch, selector);
    std::unique_ptr<faiss::DistanceComputer> dis(
            this->storage.get_distance_computer());

    dis->set_query(q_vec);
    for (int i = 0; i <
         std::min(static_cast<int>(k),
                  static_cast<int>(final_candidates.size()));
         i++) {
        idx_t candidate = final_candidates[i];
        idx_t candidate_index = this->graph->getIndex(0, candidate);
        float dist = (*dis)(candidate_index);
        labels[i] = candidate_index;
        distances[i] = dist;
    }
    for (int i = std::min(
                 static_cast<int>(k),
                 static_cast<int>(final_candidates.size()));
         i < k;
         i++) {
        labels[i] = -1;
        distances[i] = std::numeric_limits<float>::infinity();
    }
}

} // namespace faiss
