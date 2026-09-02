#ifndef FAISS_HIERARCHICAL_GRAPH_H
#define FAISS_HIERARCHICAL_GRAPH_H

#include <faiss/Index.h>
#include <omp.h>
#include <deque>
#include <vector>

namespace faiss {

struct HierarchicalGraph {
   private:
    std::vector<std::deque<omp_lock_t>> node_locks;
    omp_lock_t expansion_lock;

    bool addDirectedEdge(int layer, idx_t node1, idx_t node2);
    int find_index(int layer, idx_t node, idx_t vector_id) const;
    bool removeDirectedEdge(int layer, idx_t node1, idx_t node2);
    float calcDistance(int layer, idx_t node1, idx_t node2) const;

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
    void addInitialBottomEdges(idx_t new_node, std::vector<idx_t> candidates);
    void twoHopPruning(int layer, idx_t node);

    const std::vector<idx_t>& getNeighbors(int layer, idx_t node) const;
    const std::vector<idx_t> getNeighborsSafe(int layer, idx_t node) const;
    idx_t getIndex(int layer, idx_t node) const;
    idx_t getIndexSafe(int layer, idx_t node) const;
    idx_t getDownwardsNode(int layer, idx_t node) const;
    idx_t getDownwardsNodeSafe(int layer, idx_t node) const;

    int getEntryPoint() const;
    int getMaxLayer() const;
    int getMaxLayerSafe() const;

    void clear();
    void print() const;
};

} // namespace faiss

#endif // FAISS_HIERARCHICAL_GRAPH_H