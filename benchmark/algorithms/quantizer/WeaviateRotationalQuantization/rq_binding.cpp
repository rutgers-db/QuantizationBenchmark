// pybind11 binding for Weaviate RotationalQuantizer.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

#include "rq.h"

namespace py = pybind11;

namespace {
rq::Metric parse_metric(const std::string& s) {
    if (s == "l2" || s == "L2") return rq::Metric::L2;
    if (s == "ip" || s == "IP") return rq::Metric::IP;
    throw std::invalid_argument("metric must be 'l2' or 'ip'");
}
}  // namespace

class PyRotationalQuantizer {
 public:
    PyRotationalQuantizer(int d, int bits, const std::string& metric,
                          std::uint64_t seed, int num_threads)
        : d_(d),
          index_(std::make_unique<rq::RotationalQuantizer>(
              d, bits, parse_metric(metric), seed)) {
        index_->set_num_threads(num_threads);
    }

    void train(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        // RotationalQuantizer::train is a no-op (data-independent rotation),
        // but we still validate shape for early error reporting.
        auto buf = data.request();
        if (buf.ndim != 2) throw std::runtime_error("train: data must be 2-D");
        index_->train(static_cast<std::size_t>(buf.shape[0]),
                      static_cast<const float*>(buf.ptr));
    }

    void add(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        auto buf = data.request();
        if (buf.ndim != 2) throw std::runtime_error("add: data must be 2-D");
        if (static_cast<int>(buf.shape[1]) != d_)
            throw std::runtime_error("add: dim mismatch");
        index_->add(static_cast<std::size_t>(buf.shape[0]),
                    static_cast<const float*>(buf.ptr));
    }

    std::pair<py::array_t<std::int64_t>, py::array_t<float>>
    search(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
           std::size_t k) {
        auto buf = queries.request();
        if (buf.ndim != 2) throw std::runtime_error("search: queries must be 2-D");
        if (static_cast<int>(buf.shape[1]) != d_)
            throw std::runtime_error("search: dim mismatch");
        std::size_t nq = static_cast<std::size_t>(buf.shape[0]);
        py::array_t<std::int64_t> labels({nq, k});
        py::array_t<float>        dists ({nq, k});
        index_->search(nq, static_cast<const float*>(buf.ptr), k,
                       dists.mutable_data(), labels.mutable_data());
        return {labels, dists};
    }

    void set_num_threads(int n) { index_->set_num_threads(n); }

    std::size_t ntotal()    const { return index_->ntotal(); }
    std::size_t code_size() const { return index_->code_size(); }
    int         get_dim()   const { return index_->d(); }
    int         out_dim()   const { return index_->out_dim(); }
    int         bits()      const { return index_->bits(); }

 private:
    int d_;
    std::unique_ptr<rq::RotationalQuantizer> index_;
};

PYBIND11_MODULE(rq_cpp, m) {
    m.doc() = "Weaviate Rotational Quantization Python bindings";

    py::class_<PyRotationalQuantizer>(m, "PyRotationalQuantizer")
        .def(py::init<int, int, const std::string&, std::uint64_t, int>(),
             py::arg("d"),
             py::arg("bits")        = 8,
             py::arg("metric")      = "l2",
             py::arg("seed")        = 0x517cc1b727220a95ULL,
             py::arg("num_threads") = 1)
        .def("train",           &PyRotationalQuantizer::train,           py::arg("data"))
        .def("add",             &PyRotationalQuantizer::add,             py::arg("data"))
        .def("search",          &PyRotationalQuantizer::search,          py::arg("queries"), py::arg("k"))
        .def("set_num_threads", &PyRotationalQuantizer::set_num_threads, py::arg("n"))
        .def("ntotal",          &PyRotationalQuantizer::ntotal)
        .def("code_size",       &PyRotationalQuantizer::code_size)
        .def("get_dim",         &PyRotationalQuantizer::get_dim)
        .def("out_dim",         &PyRotationalQuantizer::out_dim)
        .def("bits",            &PyRotationalQuantizer::bits);
}
