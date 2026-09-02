#ifndef FAISS_INDEX_ACORN_H
#define FAISS_INDEX_ACORN_H

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/HierarchicalGraph.h>
#include <memory>
#include <random>
#include <vector>

namespace faiss {

struct SearchParametersAcorn : SearchParameters {
    int efSearch = 16;
    ~SearchParametersAcorn() override = default;
};

struct IndexAcorn : Index {
   private:
    std::unique_ptr<HierarchicalGraph> graph;

   public:
    IndexFlat storage;
    int efConstruction;
    int gamma;
    int M;
    int Mbeta;
    float ml;

    std::mt19937 rng;

    explicit IndexAcorn(
            int dimensions,
            int efConstruction,
            int gamma,
            int M,
            int Mbeta,
            MetricType metric = METRIC_L2);

    IndexAcorn(const std::string& index_file, const std::string& dataset_file);
    void writeToFile(
            const std::string& index_file,
            const std::string& dataset_file = "") const;

    ~IndexAcorn() override = default;

    void add(idx_t n, const float* x) override;

    void addSingle(idx_t index, const float* vec, std::mt19937& rng);

    int assignLayer(std::mt19937& rng) const;

    void reset() override;

    void reconstruct(idx_t key, float* recons) const override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void searchSingle(
            const float* q_vec,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const;

    std::vector<idx_t> searchLayer(
            int layer,
            idx_t entry_node,
            const float* vec,
            int results_size,
            const IDSelector* sel = nullptr) const;

    std::vector<idx_t> searchLayerSafe(
            int layer,
            idx_t entry_node,
            const float* vec,
            int results_size) const;
};

} // namespace faiss

#endif