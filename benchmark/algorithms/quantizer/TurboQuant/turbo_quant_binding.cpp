// pybind11 binding for TurboQuantIndex (unified)
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <memory>
#include <vector>

#include "turbo_quant_index_unified.hpp"

namespace py = pybind11;
using TQIndex = turboquant::TurboQuantIndex;

class PyTurboQuant {
 public:
  PyTurboQuant(std::size_t dim,
               std::size_t bitwidth,
               int mode_int,            // 0 = kMSE, 1 = kInnerProduct
               std::size_t num_threads,
               std::uint64_t seed,
               int rotation_type_int,   // 0 = kHadamard, 1 = kDense
               bool use_data_centroid,
               std::size_t nlist,
               std::size_t nprobe)
      : dim_(dim)
  {
    TQIndex::Config cfg;
    cfg.dim               = dim;
    cfg.bitwidth          = bitwidth;
    cfg.mode              = (mode_int == 0) ? TQIndex::Mode::kMSE
                                            : TQIndex::Mode::kInnerProduct;
    cfg.rotation_type     = (rotation_type_int == 0)
                            ? TQIndex::RotationType::kHadamard
                            : TQIndex::RotationType::kDense;
    cfg.use_data_centroid = use_data_centroid;
    cfg.num_threads       = num_threads;
    cfg.seed              = seed;
    cfg.nlist             = nlist;
    cfg.nprobe            = nprobe;
    index_ = std::make_unique<TQIndex>(cfg);
  }

  // ------------------------------------------------------------------ //
  //  train / add                                                         //
  // ------------------------------------------------------------------ //
  void train(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    auto buf = data.request();
    if (buf.ndim != 2) throw std::runtime_error("train: data must be 2-D");
    index_->train(static_cast<std::size_t>(buf.shape[0]),
                  static_cast<const float*>(buf.ptr));
  }

  void add(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    auto buf = data.request();
    if (buf.ndim != 2) throw std::runtime_error("add: data must be 2-D");
    std::size_t n = static_cast<std::size_t>(buf.shape[0]);
    const float* ptr = static_cast<const float*>(buf.ptr);
    index_->add(n, ptr);
    // keep a copy of the added data for getMSE and estimate_distance
    data_copy_.assign(ptr, ptr + n * dim_);
    n_data_ = n;
    recon_cache_.clear();  // invalidate lazily-built cache
  }

  // ------------------------------------------------------------------ //
  //  search — returns (labels, distances), each shape (nq, k)           //
  // ------------------------------------------------------------------ //
  std::pair<py::array_t<std::int64_t>, py::array_t<float>>
  search(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
         std::size_t k)
  {
    auto buf = queries.request();
    if (buf.ndim != 2) throw std::runtime_error("search: queries must be 2-D");
    std::size_t nq = static_cast<std::size_t>(buf.shape[0]);

    py::array_t<std::int64_t> labels({nq, k});
    py::array_t<float>        dists ({nq, k});
    // Convenience overload: always uses L2 (Euclidean) distance
    index_->query(nq, static_cast<const float*>(buf.ptr), k,
                  dists.mutable_data(), labels.mutable_data());
    return {labels, dists};
  }

  // ------------------------------------------------------------------ //
  //  reconstruct — decode all (or first n) entries, shape (n, dim)      //
  // ------------------------------------------------------------------ //
  py::array_t<float> reconstruct(std::size_t n = 0) {
    if (n == 0 || n > index_->ntotal()) n = index_->ntotal();
    py::array_t<float> out({n, dim_});
    index_->reconstruct(n, out.mutable_data());
    return out;
  }

  // ------------------------------------------------------------------ //
  //  getMSE — mean squared reconstruction error                         //
  // ------------------------------------------------------------------ //
  float getMSE() {
    if (n_data_ == 0 || data_copy_.empty()) return 0.0f;
    py::array_t<float> recon_arr = reconstruct(n_data_);
    const float* recon = recon_arr.data();
    const float* orig  = data_copy_.data();
    double mse = 0.0;
    for (std::size_t i = 0; i < n_data_; ++i) {
      for (std::size_t j = 0; j < dim_; ++j) {
        float d = orig[i * dim_ + j] - recon[i * dim_ + j];
        mse += static_cast<double>(d * d);
      }
    }
    return static_cast<float>(mse / static_cast<double>(n_data_));
  }

  // ------------------------------------------------------------------ //
  //  set_query / estimate_distance  (used by graph-index traversal)     //
  // ------------------------------------------------------------------ //
  void set_query(py::array_t<float, py::array::c_style | py::array::forcecast> query) {
    auto buf = query.request();
    if (buf.ndim != 2 || static_cast<std::size_t>(buf.shape[1]) != dim_)
      throw std::runtime_error("set_query: expected shape (1, dim)");
    const float* qptr = static_cast<const float*>(buf.ptr);
    stored_query_.assign(qptr, qptr + dim_);

    // Lazily build reconstruction cache
    if (recon_cache_.empty() && n_data_ > 0) {
      recon_cache_.resize(n_data_ * dim_);
      index_->reconstruct(n_data_, recon_cache_.data());
    }
  }

  float estimate_distance(std::size_t idx) {
    if (stored_query_.empty()) return std::numeric_limits<float>::max();
    const float* q = stored_query_.data();
    const float* x = recon_cache_.empty()
                       ? (data_copy_.data() + idx * dim_)
                       : (recon_cache_.data() + idx * dim_);
    float dist = 0.0f;
    for (std::size_t j = 0; j < dim_; ++j) {
      float d = q[j] - x[j];
      dist += d * d;
    }
    return std::sqrt(dist);
  }

  std::size_t ntotal()   const { return index_->ntotal(); }
  std::size_t get_dim()  const { return dim_; }
  std::size_t bitwidth() const { return index_->bitwidth(); }

 private:
  std::unique_ptr<TQIndex> index_;
  std::size_t dim_;
  std::size_t n_data_ = 0;
  std::vector<float> data_copy_;    // original float32 data (added via add())
  std::vector<float> recon_cache_;  // lazily-computed reconstruction cache
  std::vector<float> stored_query_; // set_query state
};


PYBIND11_MODULE(turbo_quant_cpp, m) {
  m.doc() = "TurboQuant (unified) Python bindings";

  py::class_<PyTurboQuant>(m, "PyTurboQuant")
      .def(py::init<std::size_t, std::size_t, int, std::size_t, std::uint64_t,
                    int, bool, std::size_t, std::size_t>(),
           py::arg("dim"),
           py::arg("bitwidth"),
           py::arg("mode")              = 0,          // 0 = kMSE
           py::arg("num_threads")       = 1,
           py::arg("seed")              = 123456789ULL,
           py::arg("rotation_type")     = 1,          // 1 = kDense
           py::arg("use_data_centroid") = true,
           py::arg("nlist")             = 1,
           py::arg("nprobe")            = 1)
      .def("train",             &PyTurboQuant::train,             py::arg("data"))
      .def("add",               &PyTurboQuant::add,               py::arg("data"))
      .def("search",            &PyTurboQuant::search,            py::arg("queries"), py::arg("k"))
      .def("reconstruct",       &PyTurboQuant::reconstruct,       py::arg("n") = 0)
      .def("getMSE",            &PyTurboQuant::getMSE)
      .def("set_query",         &PyTurboQuant::set_query,         py::arg("query"))
      .def("estimate_distance", &PyTurboQuant::estimate_distance, py::arg("idx"))
      .def("ntotal",            &PyTurboQuant::ntotal)
      .def("get_dim",           &PyTurboQuant::get_dim)
      .def("bitwidth",          &PyTurboQuant::bitwidth);
}
