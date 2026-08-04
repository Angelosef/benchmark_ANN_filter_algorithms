#include <faiss/impl/HierarchicalGraph.h>
#include <algorithm>
#include <cassert>
#include <iostream>

namespace faiss {

HierarchicalGraph::HierarchicalGraph() {
    omp_init_lock(&(this->expansion_lock));
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

void HierarchicalGraph::addEdge(int layer, idx_t node1, idx_t node2) {
    // Node lock pointers in std::deque are guaranteed to remain stable!
    omp_lock_t* lock1 = &(this->node_locks[layer][node1]);
    omp_lock_t* lock2 = &(this->node_locks[layer][node2]);

    omp_set_lock(lock1);
    this->graph[layer][node1].push_back(node2);
    omp_unset_lock(lock1);

    omp_set_lock(lock2);
    this->graph[layer][node2].push_back(node1);
    omp_unset_lock(lock2);
}

void HierarchicalGraph::removeEdge(int layer, idx_t node1, idx_t node2) {
    omp_lock_t* lock1 = &(this->node_locks[layer][node1]);
    omp_lock_t* lock2 = &(this->node_locks[layer][node2]);

    omp_set_lock(lock1);
    auto& neighbors1 = this->graph[layer][node1];
    neighbors1.erase(
            std::remove(neighbors1.begin(), neighbors1.end(), node2),
            neighbors1.end());
    omp_unset_lock(lock1);

    omp_set_lock(lock2);
    auto& neighbors2 = this->graph[layer][node2];
    neighbors2.erase(
            std::remove(neighbors2.begin(), neighbors2.end(), node1),
            neighbors2.end());
    omp_unset_lock(lock2);
}

bool HierarchicalGraph::tryReplaceEdge(
        int layer,
        idx_t node,
        idx_t node_to_remove,
        idx_t node_to_add,
        size_t max_neighbors) {
    // Always lock in deterministic order to prevent deadlocks
    idx_t first = std::min(node, node_to_add);
    idx_t second = std::max(node, node_to_add);

    omp_lock_t* lock_first = &(this->node_locks[layer][first]);
    omp_lock_t* lock_second = &(this->node_locks[layer][second]);

    omp_set_lock(lock_first);
    if (first != second) {
        omp_set_lock(lock_second);
    }

    auto& neighbors = this->graph[layer][node];
    bool modified = false;

    // 1. If we still have room, just add the new edge
    if (neighbors.size() < max_neighbors) {
        neighbors.push_back(node_to_add);
        this->graph[layer][node_to_add].push_back(node);
        modified = true;
    }
    // 2. Otherwise, check if node_to_remove is STILL a neighbor
    else if (node_to_remove != -1) {
        auto it = std::find(neighbors.begin(), neighbors.end(), node_to_remove);
        if (it != neighbors.end()) {
            // Safely swap furthest_node for node_to_add
            *it = node_to_add;

            // Update node_to_add back-link
            this->graph[layer][node_to_add].push_back(node);

            // Remove node back-link from node_to_remove (requires lock on
            // node_to_remove)
            omp_lock_t* lock_remove =
                    &(this->node_locks[layer][node_to_remove]);
            omp_set_lock(lock_remove);
            auto& rem_neighbors = this->graph[layer][node_to_remove];
            rem_neighbors.erase(
                    std::remove(
                            rem_neighbors.begin(), rem_neighbors.end(), node),
                    rem_neighbors.end());
            omp_unset_lock(lock_remove);

            modified = true;
        }
    }

    if (first != second) {
        omp_unset_lock(lock_second);
    }
    omp_unset_lock(lock_first);

    return modified;
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

} // namespace faiss