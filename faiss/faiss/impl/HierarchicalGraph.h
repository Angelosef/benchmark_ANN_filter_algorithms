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

   public:
    // std::deque ensures element pointers remain valid upon expansion
    std::vector<std::deque<std::vector<idx_t>>> graph;
    std::vector<std::deque<idx_t>> indexes;
    std::vector<std::deque<idx_t>> downwards_edges;

    HierarchicalGraph();
    ~HierarchicalGraph();

    idx_t addNode(int node_layer, idx_t index);
    void addEdge(int layer, idx_t node1, idx_t node2);
    void removeEdge(int layer, idx_t node1, idx_t node2);
    bool tryReplaceEdge(
            int layer,
            idx_t node,
            idx_t node_to_remove,
            idx_t node_to_add,
            size_t max_neighbors);

    const std::vector<idx_t> getNeighbors(int layer, idx_t node) const;
    idx_t getIndex(int layer, idx_t node) const;
    idx_t getDownwardsNode(int layer, idx_t node) const;

    int getEntryPoint() const;
    int getMaxLayer() const;

    void clear();
    void print() const;
};

} // namespace faiss

#endif // FAISS_HIERARCHICAL_GRAPH_H