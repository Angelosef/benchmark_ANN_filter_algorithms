#ifndef FAISS_HIERARCHICAL_GRAPH_H
#define FAISS_HIERARCHICAL_GRAPH_H

#include <faiss/Index.h>
#include <omp.h>
#include <deque>
#include <iomanip>
#include <map>
#include <vector>

namespace faiss {
struct LayerPercentiles {
    int layer;
    size_t node_count;
    size_t min_edges;
    size_t max_edges;
    // Maps percentile rank (e.g., 1, 10, 20... 90, 99) to edge count
    std::map<int, size_t> percentiles;
};

struct ComponentStats {
    int layer;
    size_t total_nodes;
    size_t num_components;
    size_t largest_component_size;
    float lcc_percentage;  // % of layer nodes in the main component
    size_t isolated_nodes; // Nodes with 0 edges
};

struct LayerSortingStats {
    int layer;
    size_t total_nodes;
    size_t total_neighbors_evaluated;
    size_t fully_sorted_nodes;
    size_t total_inversions;
    float avg_inversions_per_node;
    float inverted_pairs_pct; // % of adjacent neighbor pairs out of order
    float sorted_nodes_pct; // % of nodes whose adjacency lists are 100% sorted
};

struct HierarchicalGraph {
   private:
    std::vector<std::deque<omp_lock_t>> node_locks;
    omp_lock_t expansion_lock;

    bool addDirectedEdge(int layer, idx_t node1, idx_t node2);
    int find_index(int layer, idx_t node, idx_t vector_id) const;
    bool removeDirectedEdge(int layer, idx_t node1, idx_t node2);

   public:
    Index* storage;
    int M;
    int Mbeta;
    int gamma;
    size_t max_neighbors;
    // std::deque ensures element pointers remain valid upon expansion
    std::vector<std::deque<std::vector<idx_t>>> graph;
    std::vector<std::deque<idx_t>> indexes;
    std::vector<std::deque<idx_t>> downwards_edges;

    HierarchicalGraph(Index* storage, int M, int Mbeta, int gamma);
    HierarchicalGraph() = default;

    ~HierarchicalGraph();

    idx_t addNode(int node_layer, idx_t index);
    void addInitialEdges(
            int layer,
            idx_t new_node,
            std::vector<idx_t> candidates);
    void twoHopPruning(int layer, idx_t node);
    void removeLast(int layer, idx_t node);

    const std::vector<idx_t>& getNeighbors(int layer, idx_t node) const;
    const std::vector<idx_t> getNeighborsSafe(
            int layer,
            idx_t node,
            int num_neighbors = -1) const;
    idx_t getIndex(int layer, idx_t node) const;
    idx_t getIndexSafe(int layer, idx_t node) const;
    idx_t getDownwardsNode(int layer, idx_t node) const;

    int getEntryPoint() const;
    int getMaxLayer() const;
    int getMaxLayerSafe() const;

    void clear();
    void print() const;
    std::vector<float> avg_num_neighbors() const;

    // Returns a vector of percentile stats for every layer in the graph
    std::vector<LayerPercentiles> getEdgePercentiles() const;

    // Helper to compute and directly print percentiles to stdout
    void printEdgePercentiles() const;
    std::vector<float> checkBidirectionalitySampling(
            size_t samples_per_layer = 10000,
            uint32_t seed = 42) const;

    // Helper to directly print the results to stdout
    void printBidirectionalityStats(size_t samples_per_layer = 10000) const;
    std::vector<ComponentStats> getConnectedComponents() const;

    // Prints component statistics per layer
    void printConnectedComponents() const;
    std::vector<LayerSortingStats> getSortingStats() const;
    void printSortingStats() const;
};

} // namespace faiss

#endif // FAISS_HIERARCHICAL_GRAPH_H