#include <faiss/IndexACORN.h>
#include <vector>
#include <iostream>
#include <cstdlib>

int main() {
    int d = 4;      // small dimension
    int nb = 10;    // database size
    int nq = 2;     // number of queries
    int k = 2;

    int M = 4;
    int gamma = 2;
    int M_beta = 4;


    // 2. Create fake database
    std::vector<float> xb(nb * d);
    for (int i = 0; i < nb * d; i++) {
        xb[i] = drand48();  // random floats
    }

    // fake metadata (2 categories: 0 or 1)
    std::vector<int> metadata(nb);
    for (int i = 0; i < nb; i++) {
        metadata[i] = i % 2;
    }

    std::cout << "Metadata:\n";
    for (int i = 0; i < nb; i++) {
        std::cout << i << ": " << metadata[i] << "\n";
    }

    faiss::IndexACORNFlat index(d, M, gamma, metadata, M_beta);

    // add to index
    index.add(nb, xb.data());

    // 3. Create queries
    std::vector<float> xq(nq * d);
    for (int i = 0; i < nq * d; i++) {
        xq[i] = drand48();
    }

    // each query wants a category
    std::vector<int> aq = {0, 1};

    // 4. Build filter mask
    std::vector<char> filter_ids_map(nq * nb);

    for (int qi = 0; qi < nq; qi++) {
        for (int xi = 0; xi < nb; xi++) {
            filter_ids_map[qi * nb + xi] =
                (metadata[xi] == aq[qi]);
        }
    }

    // 5. Output buffers
    std::vector<faiss::idx_t> I(nq * k);
    std::vector<float> D(nq * k);

    // 6. Search
    index.search(nq, xq.data(), k, D.data(), I.data(), filter_ids_map.data());

    // 7. Print results
    for (int i = 0; i < nq; i++) {
        std::cout << "Query " << i << ":\n";
        for (int j = 0; j < k; j++) {
            std::cout << "  id=" << I[i * k + j]
                      << " dist=" << D[i * k + j] << "\n";
        }
    }

    return 0;
}
