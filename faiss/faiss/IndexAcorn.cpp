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
    this->rng.seed(42);
    this->ml = 1 / std::log(this->M);
}

void IndexAcorn::reset() {
    this->storage.reset();
    this->ntotal = 0;
    this->graph.clear();
}

void IndexAcorn::reconstruct(idx_t key, float* recons) const {
    this->storage.reconstruct(key, recons);
}

idx_t IndexAcorn::getFurthest(
        int layer,
        const std::vector<idx_t> candidates,
        idx_t target) {
    if (candidates.empty()) {
        return -1; // Guard against empty candidate lists
    }
    idx_t target_index = this->graph.getIndexSafe(layer, target);
    const float* target_vec = this->storage.get_xb() + target_index * this->d;

    // Get distance computer from storage
    std::unique_ptr<faiss::DistanceComputer> dis(
            this->storage.get_distance_computer());

    dis->set_query(target_vec);

    idx_t furthest_node = candidates[0];
    float max_dist = (*dis)(this->graph.getIndexSafe(layer, candidates[0]));

    for (size_t i = 1; i < candidates.size(); ++i) {
        float dist = (*dis)(this->graph.getIndexSafe(layer, candidates[i]));
        if (dist > max_dist) {
            max_dist = dist;
            furthest_node = candidates[i];
        }
    }

    return furthest_node;
}

void IndexAcorn::add(idx_t n, const float* x) {
    idx_t offset = this->storage.ntotal;
    this->storage.add(n, x);

#pragma omp parallel
    {
        // Thread-local RNG to avoid contention/data races on this->rng
        std::mt19937 local_rng(42 + omp_get_thread_num());

#pragma omp for schedule(dynamic, 100)
        for (idx_t i = 0; i < n; i++) {
            /*
            if (i % 100 == 0) {
                std::cout << "adding node of index " << i << std::endl;
            }
            */

            idx_t index = offset + i;
            const float* vec = x + i * this->d;

            this->addSingle(index, vec, local_rng);
        }
    }

    this->ntotal = this->storage.ntotal;
}

bool IndexAcorn::addEdgeConditionally(
        int layer,
        idx_t new_node,
        idx_t candidate_node) {
    if (new_node == candidate_node) {
        return false;
    }

    // 1. Snapshot neighbors for distance checking
    const std::vector<idx_t> current_neighbours =
            this->graph.getNeighborsSafe(layer, candidate_node);

    size_t max_allowed = static_cast<size_t>(this->M * this->gamma);

    // If candidate has space, try adding directly
    if (current_neighbours.size() < max_allowed) {
        return this->graph.tryReplaceEdge(
                layer, candidate_node, -1, new_node, max_allowed);
    }

    // 2. Find furthest node on snapshot
    std::vector<idx_t> candidates = current_neighbours;
    candidates.push_back(new_node);

    idx_t furthest_node = this->getFurthest(layer, candidates, candidate_node);

    // If new_node is the furthest, we don't add it
    if (furthest_node == new_node) {
        return false;
    }

    // 3. Atomically attempt replacement with validation inside the lock
    return this->graph.tryReplaceEdge(
            layer, candidate_node, furthest_node, new_node, max_allowed);
}

void IndexAcorn::addSingle(idx_t index, const float* vec, std::mt19937& rng) {
    bool build_phase = true;
    int max_layer = this->graph.getMaxLayerSafe();

    int assigned_layer = this->assignLayer(rng);
    idx_t new_node = this->graph.addNode(assigned_layer, index);

    idx_t entry_node = this->graph.getEntryPoint();

    for (int layer = max_layer; layer > assigned_layer; layer -= 1) {
        std::vector<idx_t> results =
                this->searchLayerSafe(layer, entry_node, vec, 1);
        entry_node = this->graph.getDownwardsNodeSafe(layer, results[0]);
    }
    for (int layer = this->graph.getMaxLayerSafe(); layer > max_layer;
         layer -= 1) {
        new_node = this->graph.getDownwardsNodeSafe(layer, new_node);
    }

    for (int layer = std::min(max_layer, assigned_layer); layer > -1;
         layer -= 1) {
        int active_ef = this->efConstruction;
        if (layer == 0)
            active_ef = active_ef / 2;
        std::vector<idx_t> results =
                this->searchLayerSafe(layer, entry_node, vec, active_ef);
        entry_node = this->graph.getDownwardsNodeSafe(layer, results[0]);

        // add edges
        if (layer > 0) {
            for (int i = 0; i < this->M * this->gamma && i < results.size();
                 i++) {
                idx_t candidate_node = results[i];
                this->addEdgeConditionally(layer, new_node, candidate_node);
            }

            new_node = this->graph.getDownwardsNodeSafe(layer, new_node);
        } else if (layer == 0) {
            int added_count = 0;
            for (int i = 0; i < this->Mbeta && i < results.size(); i++) {
                idx_t candidate_node = results[i];
                bool added = this->addEdgeConditionally(
                        layer, new_node, candidate_node);
                if (added) {
                    added_count++;
                }
            }
            std::unordered_set<idx_t> dynamic_neighbours;

            int i = this->Mbeta;
            while (dynamic_neighbours.size() <
                           (this->M * this->gamma - added_count) &&
                   i < results.size()) {
                idx_t candidate_node = results[i];
                if (dynamic_neighbours.count(candidate_node) > 0) {
                    i++;
                    continue;
                }

                bool added = this->addEdgeConditionally(
                        layer, new_node, candidate_node);
                if (added) {
                    const std::vector<idx_t> two_hop_neighbours =
                            this->graph.getNeighborsSafe(0, candidate_node);
                    dynamic_neighbours.insert(candidate_node);
                    for (idx_t node : two_hop_neighbours) {
                        dynamic_neighbours.insert(node);
                    }
                }
                i++;
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

    float entry_dist = (*dis)(this->graph.getIndex(layer, entry_node));

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
                this->graph.getNeighbors(layer, frontier_node);

        // Process 1-hop neighbors
        int neighbour_count = 0;
        for (idx_t node : neighbours) {
            if ((!sel || sel->is_member(this->graph.getIndex(layer, node))) &&
                visited.count(node) == 0) {
                visited.insert(node);
                float dist = (*dis)(this->graph.getIndex(layer, node));
                if (dist < results.max() || results.size() < results_size) {
                    results.push(node, dist);
                    frontier.push({dist, node});
                }
            }
            neighbour_count++;
        }

        // Process 2-hop neighbors for ACORN layer 0
        if (layer == 0) {
            for (size_t i = this->Mbeta; i < neighbours.size(); ++i) {
                idx_t one_hop_node = neighbours[i];

                // Zero-copy reference to 2-hop neighbors
                const std::vector<idx_t>& two_hop_neighbours =
                        this->graph.getNeighbors(layer, one_hop_node);

                for (idx_t node : two_hop_neighbours) {
                    if ((!sel ||
                         sel->is_member(this->graph.getIndex(layer, node))) &&
                        visited.count(node) == 0) {
                        visited.insert(node);
                        float dist = (*dis)(this->graph.getIndex(layer, node));
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
    result_nodes.reserve(results.size());

    // Dummy buffer for pop_min
    float dummy_dist = 0.0f;
    while (results.size() > 0) {
        result_nodes.push_back(results.pop_min(&dummy_dist));
    }

    return result_nodes;
}

std::vector<idx_t> IndexAcorn::searchLayerSafe(
        int layer,
        idx_t entry_node,
        const float* vec,
        int results_size) const {
    std::unique_ptr<faiss::DistanceComputer> dis(
            this->storage.get_distance_computer());

    dis->set_query(vec);

    MinimaxHeapT results(results_size);

    // Min-Heap for greedy search (closest candidates popped first)
    using DistNode = std::pair<float, idx_t>;
    std::priority_queue<DistNode, std::vector<DistNode>, std::greater<DistNode>>
            frontier;

    std::unordered_set<idx_t> visited;

    float entry_dist = (*dis)(this->graph.getIndexSafe(layer, entry_node));

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

        const std::vector<idx_t> neighbours =
                this->graph.getNeighborsSafe(layer, frontier_node);

        // Process 1-hop neighbors
        int neighbour_count = 0;
        for (idx_t node : neighbours) {
            if (visited.count(node) == 0) {
                visited.insert(node);
                float dist = (*dis)(this->graph.getIndexSafe(layer, node));
                if (dist < results.max() || results.size() < results_size) {
                    results.push(node, dist);
                    frontier.push({dist, node});
                }
            }
            neighbour_count++;
        }

        // Process 2-hop neighbors for ACORN layer 0
        if (layer == 0) {
            for (size_t i = this->Mbeta; i < neighbours.size(); ++i) {
                idx_t one_hop_node = neighbours[i];

                // Zero-copy reference to 2-hop neighbors
                const std::vector<idx_t> two_hop_neighbours =
                        this->graph.getNeighborsSafe(layer, one_hop_node);

                for (idx_t node : two_hop_neighbours) {
                    if (neighbour_count >= this->M) {
                        break;
                    }
                    if (visited.count(node) == 0) {
                        visited.insert(node);
                        float dist = (*dis)(this->graph.getIndex(layer, node));
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
    int max_layer = this->graph.getMaxLayer();
    idx_t entry_node = this->graph.getEntryPoint();

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
        entry_node = this->graph.getDownwardsNode(layer, results[0]);
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
        idx_t candidate_index = this->graph.getIndex(0, candidate);
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
