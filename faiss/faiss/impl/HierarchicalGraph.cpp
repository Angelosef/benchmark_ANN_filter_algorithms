#include <bits/stdc++.h>
#include <faiss/impl/HierarchicalGraph.h>
#include <algorithm>
#include <iostream>

namespace faiss {

idx_t HierarchicalGraph::addNode(int node_layer, idx_t index) {
    int max_layer = this->graph.size() - 1;

    for (int layer = 0; layer <= std::min(max_layer, node_layer); layer++) {
        this->indexes[layer].push_back(index);
        if (layer > 0) {
            this->downwards_edges[layer - 1].push_back(
                    this->indexes[layer - 1].size() - 1);
        }
        std::vector<idx_t> adj_list = {};
        this->graph[layer].push_back(adj_list);
    }

    for (int layer = max_layer + 1; layer <= node_layer; layer++) {
        std::vector<idx_t> index_layer = {index};
        this->indexes.push_back(index_layer);
        if (layer > 0) {
            std::vector<idx_t> layer_down_edges = {
                    static_cast<idx_t>(this->indexes[layer - 1].size() - 1)};
            this->downwards_edges.push_back(layer_down_edges);
        }
        std::vector<std::vector<idx_t>> layer_adj_lists = {};
        std::vector<idx_t> adj_list = {};
        layer_adj_lists.push_back(adj_list);
        this->graph.push_back(layer_adj_lists);
    }
    return this->indexes[node_layer].size() - 1;
}

void HierarchicalGraph::addEdge(int layer, idx_t node1, idx_t node2) {
    this->graph[layer][node1].push_back(node2);
    this->graph[layer][node2].push_back(node1);
}

void HierarchicalGraph::removeEdge(int layer, idx_t node1, idx_t node2) {
    this->graph[layer][node1].erase(
            std::remove(
                    this->graph[layer][node1].begin(),
                    this->graph[layer][node1].end(),
                    node2),
            this->graph[layer][node1].end());
    this->graph[layer][node2].erase(
            std::remove(
                    this->graph[layer][node2].begin(),
                    this->graph[layer][node2].end(),
                    node1),
            this->graph[layer][node2].end());
}

const std::vector<idx_t>& HierarchicalGraph::getNeighbors(int layer, idx_t node)
        const {
    assert(layer >= 0);
    assert(layer < graph.size());

    assert(node >= 0);
    assert(node < graph[layer].size());

    return this->graph[layer][node];
}

idx_t HierarchicalGraph::getIndex(int layer, idx_t node) const {
    return this->indexes[layer][node];
}

idx_t HierarchicalGraph::getDownwardsNode(int layer, idx_t node) const {
    if (layer == 0) {
        return node;
    }
    return this->downwards_edges[layer - 1][node];
}

int HierarchicalGraph::getEntryPoint() const {
    return 0;
}

int HierarchicalGraph::getMaxLayer() const {
    return this->graph.size() - 1;
}

void HierarchicalGraph::clear() {
    this->graph.clear();
    this->downwards_edges.clear();
    this->indexes.clear();
}

#include <iostream>

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

                if (nb >= 0 && nb < indexes[layer].size()) {
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
