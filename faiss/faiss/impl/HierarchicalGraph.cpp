#include <faiss/impl/HierarchicalGraph.h>
#include <faiss/utils/distances.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <stdexcept>

#include <faiss/impl/DistanceComputer.h>
#include <algorithm>
#include <iterator>
#include <memory>
#include <unordered_set>

#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
#include <deque>
#include <vector>

template <typename T>
static void write_deque(const std::deque<T>& dq, faiss::IOWriter* f) {
    size_t sz = dq.size();
    WRITE1(sz);
    if (sz > 0) {
        std::vector<T> buf(dq.begin(), dq.end());
        WRITEANDCHECK(buf.data(), sz);
    }
}

template <typename T>
static void read_deque(std::deque<T>& dq, faiss::IOReader* f) {
    size_t sz = 0;
    READ1(sz);
    dq.resize(sz);
    if (sz > 0) {
        std::vector<T> buf(sz);
        READANDCHECK(buf.data(), sz);
        dq.assign(buf.begin(), buf.end());
    }
}

namespace faiss {

HierarchicalGraph::HierarchicalGraph(
        Index* storage,
        int M,
        int Mbeta,
        int gamma)
        : storage(storage), M(M), Mbeta(Mbeta), gamma(gamma) {
    omp_init_lock(&(this->expansion_lock));
    this->max_neighbors = M * gamma;
}

static void write_hierarchical_graph(
        const HierarchicalGraph& graph,
        faiss::IOWriter* f) {
    WRITE1(graph.M);
    WRITE1(graph.Mbeta);
    WRITE1(graph.gamma);
    WRITE1(graph.max_neighbors);

    // 1. Serialize adjacency lists (graph.graph)
    size_t num_levels = graph.graph.size();
    WRITE1(num_levels);
    for (size_t l = 0; l < num_levels; ++l) {
        size_t n_nodes = graph.graph[l].size();
        WRITE1(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            const auto& neighbors = graph.graph[l][i];
            size_t deg = neighbors.size();
            WRITE1(deg);
            if (deg > 0) {
                WRITEANDCHECK(neighbors.data(), deg);
            }
        }
    }

    // 2. Serialize indexes independently using its actual size
    size_t num_levels_idx = graph.indexes.size();
    WRITE1(num_levels_idx);
    for (size_t l = 0; l < num_levels_idx; ++l) {
        write_deque(graph.indexes[l], f);
    }

    // 3. Serialize downwards_edges independently using its actual size
    size_t num_levels_down = graph.downwards_edges.size();
    WRITE1(num_levels_down);
    for (size_t l = 0; l < num_levels_down; ++l) {
        write_deque(graph.downwards_edges[l], f);
    }
}

static void read_hierarchical_graph(
        HierarchicalGraph& graph,
        faiss::IOReader* f) {
    READ1(graph.M);
    READ1(graph.Mbeta);
    READ1(graph.gamma);
    READ1(graph.max_neighbors);

    // 1. Deserialize graph.graph
    size_t num_levels = 0;
    READ1(num_levels);
    graph.graph.resize(num_levels);
    for (size_t l = 0; l < num_levels; ++l) {
        size_t n_nodes = 0;
        READ1(n_nodes);
        graph.graph[l].resize(n_nodes);
        for (size_t i = 0; i < n_nodes; ++i) {
            size_t deg = 0;
            READ1(deg);
            graph.graph[l][i].resize(deg);
            if (deg > 0) {
                READANDCHECK(graph.graph[l][i].data(), deg);
            }
        }
    }

    // 2. Deserialize graph.indexes
    size_t num_levels_idx = 0;
    READ1(num_levels_idx);
    graph.indexes.resize(num_levels_idx);
    for (size_t l = 0; l < num_levels_idx; ++l) {
        read_deque(graph.indexes[l], f);
    }

    // 3. Deserialize graph.downwards_edges
    size_t num_levels_down = 0;
    READ1(num_levels_down);
    graph.downwards_edges.resize(num_levels_down);
    for (size_t l = 0; l < num_levels_down; ++l) {
        read_deque(graph.downwards_edges[l], f);
    }
}

HierarchicalGraph::~HierarchicalGraph() {
    this->clear();
    omp_destroy_lock(&(this->expansion_lock));
}

idx_t HierarchicalGraph::addNode(int node_layer, idx_t index) {
    omp_set_lock(&(this->expansion_lock));

    // Ensure layer metadata structures exist up to node_layer
    if (static_cast<int>(this->graph.size()) <= node_layer) {
        int old_size = this->graph.size();
        this->graph.resize(node_layer + 1);
        this->indexes.resize(node_layer + 1);
        this->node_locks.resize(node_layer + 1);
        if (node_layer > 0) {
            this->downwards_edges.resize(node_layer);
        }
    }

    // Insert node sequentially across layers 0 through node_layer
    for (int layer = 0; layer <= node_layer; layer++) {
        this->indexes[layer].push_back(index);

        // Initialize and push a raw OpenMP lock into std::deque
        this->node_locks[layer].emplace_back();
        omp_init_lock(&(this->node_locks[layer].back()));

        if (layer > 0) {
            // Point downward edge to the node index at layer - 1
            this->downwards_edges[layer - 1].push_back(
                    this->indexes[layer - 1].size() - 1);
        }

        this->graph[layer].emplace_back(); // Empty adjacency vector
    }

    idx_t node_id = this->indexes[node_layer].size() - 1;
    omp_unset_lock(&(this->expansion_lock));
    return node_id;
}

float HierarchicalGraph::calcDistance(int layer, idx_t node1, idx_t node2)
        const {
    size_t d = this->storage->d;
    std::vector<float> v1(d);
    std::vector<float> v2(d);
    idx_t p1 = this->getIndex(layer, node1);
    idx_t p2 = this->getIndex(layer, node2);

    storage->reconstruct(p1, v1.data());
    storage->reconstruct(p2, v2.data());

    // 2. Compute distance based on the metric type of the storage
    if (storage->metric_type == faiss::METRIC_L2) {
        return faiss::fvec_L2sqr(
                v1.data(), v2.data(), d); // Squared L2 distance
        // Use std::sqrt(faiss::fvec_L2sqr(...)) if you need un-squared
        // Euclidean distance
    } else if (storage->metric_type == faiss::METRIC_INNER_PRODUCT) {
        return faiss::fvec_inner_product(v1.data(), v2.data(), d);
    } else {
        throw std::runtime_error("Unsupported metric type");
    }
}

int HierarchicalGraph::find_index(int layer, idx_t node, idx_t vector_id)
        const {
    size_t d = this->storage->d;
    std::vector<float> q_vec(d);
    idx_t node_idx = this->getIndex(layer, node);
    storage->reconstruct(node_idx, q_vec.data());

    std::unique_ptr<faiss::DistanceComputer> dis(
            this->storage->get_distance_computer());

    dis->set_query(q_vec.data());

    // Compute target distance for vector_id against itself
    float query_dist = (*dis)(vector_id);

    // Lambda replacing the nested function
    // potentially dangerous operation without locking?
    auto get_cost = [&](idx_t neighbor) {
        return (*dis)(this->getIndex(layer, neighbor));
    };

    const auto& neighbors = this->graph[layer][node];

    auto it = std::lower_bound(
            neighbors.begin(),
            neighbors.end(),
            query_dist,
            [&](idx_t neighbor, float target) {
                return get_cost(neighbor) < target;
            });

    return static_cast<int>(std::distance(neighbors.begin(), it));
}

bool HierarchicalGraph::addDirectedEdge(int layer, idx_t node1, idx_t node2) {
    if (node1 == node2) {
        return false;
    }
    idx_t first = std::min(node1, node2);
    idx_t second = std::max(node1, node2);

    omp_lock_t* lock_first = &(this->node_locks[layer][first]);
    omp_lock_t* lock_second = &(this->node_locks[layer][second]);

    omp_set_lock(lock_first);
    omp_set_lock(lock_second);

    bool added_edge;
    std::vector<idx_t> node1_neighbors = this->getNeighbors(layer, node1);
    if (node1_neighbors.size() < this->max_neighbors) {
        // insert to sorted list
        added_edge = true;
        idx_t vector_id = this->getIndex(layer, node2);
        int insertion_index = this->find_index(layer, node1, vector_id);
        auto& neighbors = this->graph[layer][node1];
        neighbors.insert(neighbors.begin() + insertion_index, node2);
    } else {
        idx_t furthest_node = node1_neighbors.back();
        float furthest_dist = this->calcDistance(layer, node1, furthest_node);
        float node2_dist = this->calcDistance(layer, node1, node2);

        if (furthest_dist < node2_dist) {
            added_edge = false;
        } else {
            // insert node2 and remove last
            added_edge = true;
            idx_t vector_id = this->getIndex(layer, node2);
            int insertion_index = this->find_index(layer, node1, vector_id);
            auto& neighbors = this->graph[layer][node1];
            neighbors.insert(neighbors.begin() + insertion_index, node2);
            idx_t last_neighbor = neighbors.back();
            neighbors.pop_back();
            this->removeDirectedEdge(layer, last_neighbor, node1);
        }
    }

    omp_unset_lock(lock_first);
    omp_unset_lock(lock_second);

    return added_edge;
}

bool HierarchicalGraph::removeDirectedEdge(
        int layer,
        idx_t node1,
        idx_t node2) {
    int i = 0;
    bool found = false;
    auto& neighbors = this->graph[layer][node1];
    while (!found && i < neighbors.size()) {
        if (neighbors[i] == node2) {
            found = true;
            neighbors.erase(neighbors.begin() + i);
        }
        i++;
    }
    return found;
}

void HierarchicalGraph::addInitialEdges(
        int layer,
        idx_t new_node,
        std::vector<idx_t> candidates) {
    // candidates are meant to be sorted regarding dist to new_node
    int added_count = 0;
    for (int i = 0; i < candidates.size() && added_count < this->max_neighbors;
         i++) {
        bool added = this->addDirectedEdge(layer, candidates[i], new_node);
        if (added) {
            omp_lock_t* lock_new_node = &(this->node_locks[layer][new_node]);
            omp_set_lock(lock_new_node);

            added_count++;
            this->graph[layer][new_node].push_back(candidates[i]);
            omp_unset_lock(lock_new_node);
        }
    }
}

void HierarchicalGraph::twoHopPruning(int layer, idx_t node) {
    std::vector<idx_t> old_neighbors = this->getNeighborsSafe(layer, node);
    std::vector<idx_t> updated_neighbors;
    std::vector<idx_t> discarded_neighbors;
    int added_count = 0;

    for (int i = 0; i < this->Mbeta && i < old_neighbors.size(); i++) {
        updated_neighbors.push_back(old_neighbors[i]);
        added_count++;
    }

    std::unordered_set<idx_t> dynamic_neighbors;

    int i = this->Mbeta;
    while ((added_count + dynamic_neighbors.size()) < this->max_neighbors &&
           i < old_neighbors.size()) {
        idx_t candidate_node = old_neighbors[i];
        if (dynamic_neighbors.count(candidate_node) > 0) {
            i++;
            discarded_neighbors.push_back(candidate_node);
            continue;
        }

        updated_neighbors.push_back(candidate_node);
        added_count++;
        const std::vector<idx_t> two_hop_neighbors =
                this->getNeighborsSafe(layer, candidate_node);
        for (idx_t node : two_hop_neighbors) {
            dynamic_neighbors.insert(node);
        }
        i++;
    }
    for (int idx = i; idx < old_neighbors.size(); idx++) {
        discarded_neighbors.push_back(old_neighbors[i]);
    }

    // discard old neighbor adjacency list and replace with new one
    omp_lock_t* lock_node = &(this->node_locks[layer][node]);
    omp_set_lock(lock_node);
    this->graph[layer][node] = std::move(updated_neighbors);
    omp_unset_lock(lock_node);

    for (int j = 0; j < discarded_neighbors.size(); j++) {
        idx_t discarded_node = discarded_neighbors[j];
        omp_lock_t* lock_node = &(this->node_locks[layer][discarded_node]);
        omp_set_lock(lock_node);
        this->removeDirectedEdge(layer, discarded_neighbors[j], node);
        omp_unset_lock(lock_node);
    }
}

void HierarchicalGraph::addInitialBottomEdges(
        idx_t new_node,
        std::vector<idx_t> candidates) {
    int layer = 0;
    std::vector<idx_t> neighbors;
    int added_count = 0;

    for (int i = 0; i < this->Mbeta && i < candidates.size(); i++) {
        bool added = this->addDirectedEdge(layer, candidates[i], new_node);
        if (added) {
            neighbors.push_back(candidates[i]);
            added_count++;
        }
    }

    std::unordered_set<idx_t> dynamic_neighbors;

    int i = this->Mbeta;
    while ((added_count + dynamic_neighbors.size()) < this->max_neighbors &&
           i < candidates.size()) {
        idx_t candidate_node = candidates[i];
        if (dynamic_neighbors.count(candidate_node) > 0) {
            i++;
            continue;
        }

        bool added = this->addDirectedEdge(layer, candidate_node, new_node);
        if (added) {
            added_count++;
            neighbors.push_back(candidate_node);
            const std::vector<idx_t> two_hop_neighbors =
                    this->getNeighborsSafe(layer, candidate_node);
            for (idx_t node : two_hop_neighbors) {
                dynamic_neighbors.insert(node);
            }
        }
        i++;
    }

    omp_lock_t* lock_node = &(this->node_locks[layer][new_node]);
    omp_set_lock(lock_node);
    this->graph[layer][new_node] = std::move(neighbors);
    omp_unset_lock(lock_node);
}

const std::vector<idx_t> HierarchicalGraph::getNeighborsSafe(
        int layer,
        idx_t node) const {
    omp_lock_t* lock =
            const_cast<omp_lock_t*>(&(this->node_locks[layer][node]));

    omp_set_lock(lock);
    std::vector<idx_t> copy = this->graph[layer][node];
    omp_unset_lock(lock);

    return copy;
}

const std::vector<idx_t>& HierarchicalGraph::getNeighbors(int layer, idx_t node)
        const {
    return this->graph[layer][node];
}

idx_t HierarchicalGraph::getIndexSafe(int layer, idx_t node) const {
    omp_lock_t* lock =
            const_cast<omp_lock_t*>(&(this->node_locks[layer][node]));

    omp_set_lock(lock);
    idx_t index = this->getIndex(layer, node);
    omp_unset_lock(lock);

    return index;
}

idx_t HierarchicalGraph::getIndex(int layer, idx_t node) const {
    return this->indexes[layer][node];
}

idx_t HierarchicalGraph::getDownwardsNodeSafe(int layer, idx_t node) const {
    omp_set_lock(const_cast<omp_lock_t*>(&(this->expansion_lock)));

    idx_t down_node = this->getDownwardsNode(layer, node);
    omp_unset_lock(const_cast<omp_lock_t*>(&(this->expansion_lock)));

    return down_node;
}

idx_t HierarchicalGraph::getDownwardsNode(int layer, idx_t node) const {
    if (layer <= 0) {
        return node;
    }

    if (layer - 1 >= static_cast<int>(this->downwards_edges.size()) ||
        node >= static_cast<idx_t>(this->downwards_edges[layer - 1].size())) {
        return node; // Fallback to avoid crash
    }

    idx_t down_node = this->downwards_edges[layer - 1][node];

    return down_node;
}

int HierarchicalGraph::getEntryPoint() const {
    return 0;
}

int HierarchicalGraph::getMaxLayerSafe() const {
    omp_set_lock(const_cast<omp_lock_t*>(&(this->expansion_lock)));
    int max_layer = this->getMaxLayer();
    omp_unset_lock(const_cast<omp_lock_t*>(&(this->expansion_lock)));
    return max_layer;
}

int HierarchicalGraph::getMaxLayer() const {
    int max_layer = static_cast<int>(this->graph.size()) - 1;

    return std::max(0, max_layer);
}

void HierarchicalGraph::clear() {
    omp_set_lock(&(this->expansion_lock));
    // Destroy all initialized node locks before clearing
    for (size_t layer = 0; layer < node_locks.size(); ++layer) {
        for (size_t i = 0; i < node_locks[layer].size(); ++i) {
            omp_destroy_lock(&(node_locks[layer][i]));
        }
    }
    this->graph.clear();
    this->downwards_edges.clear();
    this->indexes.clear();
    this->node_locks.clear();
    omp_unset_lock(&(this->expansion_lock));
}

void HierarchicalGraph::print() const {
    std::cout << "\n========== Hierarchical Graph ==========\n";

    for (size_t layer = 0; layer < graph.size(); ++layer) {
        std::cout << "Layer " << layer << '\n';

        for (size_t node = 0; node < graph[layer].size(); ++node) {
            std::cout << "Node " << node << " [storage=" << indexes[layer][node]
                      << "]";

            std::cout << " -> ";

            for (idx_t nb : graph[layer][node]) {
                std::cout << nb;

                if (nb >= 0 && nb < static_cast<idx_t>(indexes[layer].size())) {
                    std::cout << "(storage=" << indexes[layer][nb] << ")";
                } else {
                    std::cout << "(INVALID)";
                }

                std::cout << " ";
            }

            std::cout << '\n';
        }

        std::cout << '\n';
    }

    std::cout << "========================================\n";
}

std::vector<float> HierarchicalGraph::avg_num_neighbors() const {
    std::vector<float> avg_edges_per_layer;
    for (int i = 0; i < this->graph.size(); i++) {
        int edge_count = 0;
        for (int j = 0; j < this->graph[i].size(); j++) {
            edge_count += this->graph[i][j].size();
        }
        float avg_edges =
                static_cast<float>(edge_count / this->graph[i].size());
        avg_edges_per_layer.push_back(avg_edges);
    }
    return avg_edges_per_layer;
}

} // namespace faiss