#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <faiss/IndexACORN.h>
#include <vector>

namespace py = pybind11;

class PyACORN {
public:
    faiss::IndexACORNFlat* index;
    int d;

    PyACORN(int d, int M, int gamma,
            std::vector<int> metadata, int M_beta) : d(d) {

        index = new faiss::IndexACORNFlat(d, M, gamma, metadata, M_beta);
    }

    void add(py::array_t<float> xb) {
        auto buf = xb.request();

        int n = buf.shape[0];
        float* ptr = (float*)buf.ptr;

        index->add(n, ptr);
    }

    py::tuple search(py::array_t<float> xq,
                     int k,
                     py::array_t<char> filter_map) {

        auto qbuf = xq.request();
        auto fbuf = filter_map.request();

        int nq = qbuf.shape[0];

        float* qptr = (float*)qbuf.ptr;
        char* fptr = (char*)fbuf.ptr;

        std::vector<float> D(nq * k);
        std::vector<faiss::idx_t> I(nq * k);

        index->search(nq, qptr, k, D.data(), I.data(), fptr);

        // return as numpy arrays
        return py::make_tuple(
            py::array_t<float>({nq, k}, D.data()),
            py::array_t<faiss::idx_t>({nq, k}, I.data())
        );
    }

    ~PyACORN() {
        delete index;
    }
};

PYBIND11_MODULE(acorn, m) {
    py::class_<PyACORN>(m, "ACORNIndex")
        .def(py::init<int,int,int,std::vector<int>,int>())
        .def("add", &PyACORN::add)
        .def("search", &PyACORN::search)
        .def_property_readonly("entry_point", [](PyACORN &self) {
            return self.index->acorn.entry_point;
        })
        .def_property("efSearch",
        [](PyACORN &self) { return self.index->acorn.efSearch; },
        [](PyACORN &self, int ef) { self.index->acorn.efSearch = ef; }
        );
}
