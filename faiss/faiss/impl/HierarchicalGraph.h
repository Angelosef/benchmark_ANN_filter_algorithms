#pragma once

#include <faiss/Index.h>
#include <vector>

namespace faiss {

struct HierarchicalGraph {
    // layer - node - node
    std::vector<std::vector<std::vector<idx_t>>> graph;
    std::vector<std::vector<idx_t>> indexes;
    std::vector<std::vector<idx_t>> downwards_edges;

    HierarchicalGraph() = default;
    idx_t addNode(int node_layer, idx_t index);
    void addEdge(int layer, idx_t node1, idx_t node2);
    void removeEdge(int layer, idx_t node1, idx_t node2);
    const std::vector<idx_t>& getNeighbors(int layer, idx_t node) const;
    idx_t getIndex(int layer, idx_t node) const;
    idx_t getDownwardsNode(int layer, idx_t node) const;
    int getEntryPoint() const;
    int getMaxLayer() const;
    void clear();
    void print() const;
};

} // namespace faiss
