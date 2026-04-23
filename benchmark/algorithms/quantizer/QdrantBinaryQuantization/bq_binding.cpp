// pybind11 binding for QdrantBQ BinaryQuantizer.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "BinaryQuantizer.h"

namespace py = pybind11;

namespace {

bq::Encoding parse_encoding(int v) {
    if (v == 1) return bq::Encoding::OneBit;
    if (v == 2) return bq::Encoding::TwoBits;
    throw std::invalid_argument("encoding must be 1 (OneBit) or 2 (TwoBits)");
}

bq::QueryEncoding parse_query_encoding(const std::string& s) {
    if (s == "same" || s == "SameAsStorage") return bq::QueryEncoding::SameAsStorage;
    if (s == "scalar4")                      return bq::QueryEncoding::Scalar4Bits;
    if (s == "scalar8")                      return bq::QueryEncoding::Scalar8Bits;
    throw std::invalid_argument("query_encoding must be 'same', 'scalar4', or 'scalar8'");
}

bq::Metric parse_metric(const std::string& s) {
    if (s == "l2" || s == "L2") return bq::Metric::L2;
    if (s == "ip" || s == "IP") return bq::Metric::IP;
    if (s == "hamming" || s == "Hamming") return bq::Metric::Hamming;
    throw std::invalid_argument("metric must be 'l2', 'ip', or 'hamming'");
}

}  // namespace

class PyBinaryQuantizer {
 public:
    PyBinaryQuantizer(std::size_t d,
                      int encoding,
                      const std::string& query_encoding,
                      const std::string& metric,
                      int num_threads)
        : d_(d),
          k_db_(encoding),
          index_(std::make_unique<bq::BinaryQuantizer>(
              d, parse_encoding(encoding),
              parse_query_encoding(query_encoding),
              parse_metric(metric))) {
        index_->set_num_threads(num_threads);
    }

    void train(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        auto buf = data.request();
        if (buf.ndim != 2) throw std::runtime_error("train: data must be 2-D");
        if (static_cast<std::size_t>(buf.shape[1]) != d_)
            throw std::runtime_error("train: dim mismatch");
        index_->train(static_cast<std::size_t>(buf.shape[0]),
                      static_cast<const float*>(buf.ptr));
    }

    void add(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        auto buf = data.request();
        if (buf.ndim != 2) throw std::runtime_error("add: data must be 2-D");
        if (static_cast<std::size_t>(buf.shape[1]) != d_)
            throw std::runtime_error("add: dim mismatch");
        index_->add(static_cast<std::size_t>(buf.shape[0]),
                    static_cast<const float*>(buf.ptr));
    }

    std::pair<py::array_t<std::int64_t>, py::array_t<float>>
    search(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
           std::size_t k) {
        auto buf = queries.request();
        if (buf.ndim != 2) throw std::runtime_error("search: queries must be 2-D");
        if (static_cast<std::size_t>(buf.shape[1]) != d_)
            throw std::runtime_error("search: dim mismatch");
        std::size_t nq = static_cast<std::size_t>(buf.shape[0]);
        py::array_t<std::int64_t> labels({nq, k});
        py::array_t<float>        dists ({nq, k});
        index_->search(nq, static_cast<const float*>(buf.ptr), k,
                       dists.mutable_data(), labels.mutable_data());
        return {labels, dists};
    }

    void set_num_threads(int n) { index_->set_num_threads(n); }

    std::size_t ntotal()    const { return index_->ntotal; }
    std::size_t code_size() const { return index_->code_size; }
    std::size_t get_dim()   const { return d_; }
    int         k_db()      const { return k_db_; }

 private:
    std::size_t d_;
    int         k_db_;
    std::unique_ptr<bq::BinaryQuantizer> index_;
};

PYBIND11_MODULE(bq_cpp, m) {
    m.doc() = "Qdrant Binary Quantization Python bindings";

    py::class_<PyBinaryQuantizer>(m, "PyBinaryQuantizer")
        .def(py::init<std::size_t, int, const std::string&, const std::string&, int>(),
             py::arg("d"),
             py::arg("encoding")       = 1,
             py::arg("query_encoding") = "same",
             py::arg("metric")         = "l2",
             py::arg("num_threads")    = 1)
        .def("train",           &PyBinaryQuantizer::train,           py::arg("data"))
        .def("add",             &PyBinaryQuantizer::add,             py::arg("data"))
        .def("search",          &PyBinaryQuantizer::search,          py::arg("queries"), py::arg("k"))
        .def("set_num_threads", &PyBinaryQuantizer::set_num_threads, py::arg("n"))
        .def("ntotal",          &PyBinaryQuantizer::ntotal)
        .def("code_size",       &PyBinaryQuantizer::code_size)
        .def("get_dim",         &PyBinaryQuantizer::get_dim)
        .def("k_db",            &PyBinaryQuantizer::k_db);
}
