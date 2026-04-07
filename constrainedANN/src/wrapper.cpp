#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "FilterIndex.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(filter_index_py, m) {
    py::class_<FilterIndex>(m, "FilterIndex")
        .def(py::init([](py::array_t<float> data, size_t nc, 
                         std::vector<std::vector<std::string>> props, 
                         std::string algo, int mode) {
            // Get pointer and dimensions from Numpy
            auto buf = data.request();
            float* ptr = static_cast<float*>(buf.ptr);
            size_t nb = buf.shape[0];
            size_t d = buf.shape[1];
            
            return new FilterIndex(ptr, d, nb, nc, props, algo, mode);
        }), py::keep_alive<1, 2>())
        .def("get_index", &FilterIndex::get_index)
        .def("loadIndex", &FilterIndex::loadIndex)
        .def("query", [](FilterIndex &self, py::array_t<float> queries, 
                         std::vector<std::vector<std::string>> qprops, 
                         int k, int nprobe) {
            auto buf = queries.request();
            float* ptr = static_cast<float*>(buf.ptr);
            size_t nq = buf.shape[0];
            
            self.query(ptr, nq, qprops, k, nprobe);
            
            // Return the neighbor_set as a numpy array
            auto result = py::array_t<int32_t>({nq, (size_t)k});
            std::memcpy(result.mutable_data(), self.neighbor_set, nq * k * sizeof(int32_t));
            return result;
        });
}
