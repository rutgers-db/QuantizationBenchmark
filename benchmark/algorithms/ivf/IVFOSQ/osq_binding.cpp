#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "OSQIndex.h"

#include <algorithm>
#include <functional>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

osq::Similarity parse_similarity(const std::string& space) {
  if (space == "l2") return osq::Similarity::EUCLIDEAN;
  if (space == "cosine") return osq::Similarity::COSINE;
  if (space == "ip" || space == "inner_product") return osq::Similarity::MAX_INNER_PRODUCT;
  throw std::invalid_argument("unsupported space: " + space);
}

osq::ScalarEncoding parse_encoding(int nbit, int query_nbit) {
  if (nbit == 8 && query_nbit == 8) return osq::ScalarEncoding::UNSIGNED_BYTE;
  if (nbit == 7 && query_nbit == 7) return osq::ScalarEncoding::SEVEN_BIT;
  if (nbit == 4 && query_nbit == 4) return osq::ScalarEncoding::PACKED_NIBBLE;
  if (nbit == 1 && query_nbit == 4) return osq::ScalarEncoding::SINGLE_BIT_QUERY_NIBBLE;
  if (nbit == 2 && query_nbit == 4) return osq::ScalarEncoding::DIBIT_QUERY_NIBBLE;
  throw std::invalid_argument("unsupported OSQ encoding combination");
}

inline void compute_residual_avx(
    const float* query,
    const float* centroid,
    float* residual,
    size_t dims) {
#if defined(__AVX__)
  size_t i = 0;
  for (; i + 8 <= dims; i += 8) {
    const __m256 q = _mm256_loadu_ps(query + i);
    const __m256 c = _mm256_loadu_ps(centroid + i);
    _mm256_storeu_ps(residual + i, _mm256_sub_ps(q, c));
  }
  for (; i < dims; ++i) {
    residual[i] = query[i] - centroid[i];
  }
#else
  for (size_t i = 0; i < dims; ++i) {
    residual[i] = query[i] - centroid[i];
  }
#endif
}

class PyOSQIndex {
 public:
  PyOSQIndex(size_t dims, const std::string& space, int nbit, int query_nbit = -1)
      : dims_(dims), space_(space), nbit_(nbit) {
    if (query_nbit < 0) {
      query_nbit_ = (nbit == 1 || nbit == 2) ? 4 : nbit;
    } else {
      query_nbit_ = query_nbit;
    }
    index_ = std::make_unique<osq::OSQIndex>(
        dims_,
        parse_similarity(space_),
        parse_encoding(nbit_, query_nbit_));
  }

  void set_num_threads(int n) { index_->set_num_threads(n); }

  void train(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    auto buf = data.request();
    if (buf.ndim != 2 || static_cast<size_t>(buf.shape[1]) != dims_) {
      throw std::runtime_error("train data must have shape (n, dims)");
    }
    index_->train(static_cast<size_t>(buf.shape[0]), static_cast<const float*>(buf.ptr));
  }

  void add(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    auto buf = data.request();
    if (buf.ndim != 2 || static_cast<size_t>(buf.shape[1]) != dims_) {
      throw std::runtime_error("add data must have shape (n, dims)");
    }
    index_->add(static_cast<size_t>(buf.shape[0]), static_cast<const float*>(buf.ptr));
  }

  void build(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
    train(data);
    add(data);
  }

  std::pair<py::array_t<int64_t>, py::array_t<float>> search(
      py::array_t<float, py::array::c_style | py::array::forcecast> queries,
      size_t k) const {
    auto buf = queries.request();
    if (buf.ndim != 2 || static_cast<size_t>(buf.shape[1]) != dims_) {
      throw std::runtime_error("queries must have shape (nq, dims)");
    }

    const size_t nq = static_cast<size_t>(buf.shape[0]);
    py::array_t<int64_t> labels({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
    py::array_t<float> scores({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
    index_->search(
        nq,
        static_cast<const float*>(buf.ptr),
        k,
        static_cast<float*>(scores.request().ptr),
        reinterpret_cast<osq::idx_t*>(labels.request().ptr));
    return std::make_pair(labels, scores);
  }

  void set_query(py::array_t<float, py::array::c_style | py::array::forcecast> query) {
    auto buf = query.request();
    if (buf.ndim != 1 || static_cast<size_t>(buf.shape[0]) != dims_) {
      throw std::runtime_error("query must have shape (dims,)");
    }
    cached_query_ = std::make_unique<osq::OSQIndex::EncodedQuery>(
        index_->encode_query(static_cast<const float*>(buf.ptr)));
  }

  float score_id(int64_t id) const {
    if (!cached_query_) {
      throw std::runtime_error("set_query must be called before score_id");
    }
    return index_->score(*cached_query_, static_cast<osq::idx_t>(id));
  }

  float score_with_query(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      int64_t id) const {
    auto buf = query.request();
    if (buf.ndim != 1 || static_cast<size_t>(buf.shape[0]) != dims_) {
      throw std::runtime_error("query must have shape (dims,)");
    }
    auto encoded = index_->encode_query(static_cast<const float*>(buf.ptr));
    return index_->score(encoded, static_cast<osq::idx_t>(id));
  }

  std::pair<py::array_t<int64_t>, py::array_t<float>> search_subset(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast> ids,
      size_t k) const {
    auto qbuf = query.request();
    auto ibuf = ids.request();
    if (qbuf.ndim != 1 || static_cast<size_t>(qbuf.shape[0]) != dims_) {
      throw std::runtime_error("query must have shape (dims,)");
    }
    if (ibuf.ndim != 1) {
      throw std::runtime_error("ids must be a 1D array");
    }
    if (k == 0) {
      return std::make_pair(py::array_t<int64_t>(0), py::array_t<float>(0));
    }

    const int64_t* id_ptr = static_cast<const int64_t*>(ibuf.ptr);
    const size_t n = static_cast<size_t>(ibuf.shape[0]);
    auto encoded = index_->encode_query(static_cast<const float*>(qbuf.ptr));

    std::vector<std::pair<float, int64_t>> heap;
    heap.reserve(std::min(k, n));

    for (size_t i = 0; i < n; ++i) {
      const int64_t id = id_ptr[i];
      const float score = index_->score(encoded, static_cast<osq::idx_t>(id));
      if (heap.size() < k) {
        heap.emplace_back(score, id);
        if (heap.size() == k) {
          std::make_heap(heap.begin(), heap.end(), std::greater<std::pair<float, int64_t>>());
        }
      } else if (score > heap.front().first) {
        std::pop_heap(heap.begin(), heap.end(), std::greater<std::pair<float, int64_t>>());
        heap.back() = {score, id};
        std::push_heap(heap.begin(), heap.end(), std::greater<std::pair<float, int64_t>>());
      }
    }

    std::sort(heap.begin(), heap.end(), [](const auto& a, const auto& b) {
      return a.first > b.first;
    });

    py::array_t<int64_t> labels(static_cast<py::ssize_t>(k));
    py::array_t<float> scores(static_cast<py::ssize_t>(k));
    auto lbuf = labels.request();
    auto sbuf = scores.request();
    int64_t* lptr = static_cast<int64_t*>(lbuf.ptr);
    float* sptr = static_cast<float*>(sbuf.ptr);
    for (size_t i = 0; i < k; ++i) {
      if (i < heap.size()) {
        lptr[i] = heap[i].second;
        sptr[i] = heap[i].first;
      } else {
        lptr[i] = -1;
        sptr[i] = -std::numeric_limits<float>::infinity();
      }
    }
    return std::make_pair(labels, scores);
  }

  py::array_t<float> reconstruct(int64_t id) const {
    py::array_t<float> output(static_cast<py::ssize_t>(dims_));
    auto out = output.request();
    index_->reconstruct(static_cast<osq::idx_t>(id), static_cast<float*>(out.ptr));
    return output;
  }

  py::array_t<float> reconstruct_all() const {
    const size_t n = index_->ntotal();
    py::array_t<float> output(
        {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(dims_)});
    auto out = output.request();
    float* ptr = static_cast<float*>(out.ptr);
    for (size_t i = 0; i < n; ++i) {
      index_->reconstruct(static_cast<osq::idx_t>(i), ptr + i * dims_);
    }
    return output;
  }

  size_t ntotal() const { return index_->ntotal(); }
  size_t d() const { return dims_; }
  int nbit() const { return nbit_; }
  int query_nbit() const { return query_nbit_; }
  std::string space() const { return space_; }

 private:
  size_t dims_;
  std::string space_;
  int nbit_;
  int query_nbit_;
  std::unique_ptr<osq::OSQIndex> index_;
  std::unique_ptr<osq::OSQIndex::EncodedQuery> cached_query_;
};

class PyIVFOSQIndex {
 public:
  PyIVFOSQIndex(size_t dims, const std::string& space, int nbit, int query_nbit = -1)
      : dims_(dims), space_(space), nbit_(nbit), num_threads_(1) {
    if (query_nbit < 0) {
      query_nbit_ = (nbit == 1 || nbit == 2) ? 4 : nbit;
    } else {
      query_nbit_ = query_nbit;
    }
    index_ = std::make_unique<osq::OSQIndex>(
        dims_,
        parse_similarity(space_),
        parse_encoding(nbit_, query_nbit_));
  }

  void set_num_threads(int n) {
    num_threads_ = std::max(1, n);
    index_->set_num_threads(num_threads_);
  }

  void build(
      py::array_t<float, py::array::c_style | py::array::forcecast> residuals,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast> list_offsets,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast> list_ids) {
    auto rbuf = residuals.request();
    auto obuf = list_offsets.request();
    auto ibuf = list_ids.request();
    if (rbuf.ndim != 2 || static_cast<size_t>(rbuf.shape[1]) != dims_) {
      throw std::runtime_error("residuals must have shape (n, dims)");
    }
    if (obuf.ndim != 1 || ibuf.ndim != 1) {
      throw std::runtime_error("list_offsets and list_ids must be 1D arrays");
    }
    if (static_cast<size_t>(obuf.shape[0]) < 1) {
      throw std::runtime_error("list_offsets must contain at least one element");
    }
    const auto* offsets = static_cast<const int64_t*>(obuf.ptr);
    const auto* ids = static_cast<const int64_t*>(ibuf.ptr);
    if (offsets[static_cast<size_t>(obuf.shape[0]) - 1] != ibuf.shape[0]) {
      throw std::runtime_error("last list_offsets entry must equal list_ids length");
    }

    nlist_ = static_cast<size_t>(obuf.shape[0] - 1);
    invlist_offsets_.assign(offsets, offsets + obuf.shape[0]);
    invlist_ids_.assign(ids, ids + ibuf.shape[0]);

    const size_t n = static_cast<size_t>(rbuf.shape[0]);
    index_->train(n, static_cast<const float*>(rbuf.ptr));
    index_->add(n, static_cast<const float*>(rbuf.ptr));
  }

  std::pair<py::array_t<int64_t>, py::array_t<float>> search_preassigned(
      py::array_t<float, py::array::c_style | py::array::forcecast> queries,
      py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast> probe_lists,
      size_t k) const {
    auto qbuf = queries.request();
    auto cbuf = centroids.request();
    auto pbuf = probe_lists.request();

    if (qbuf.ndim != 2 || static_cast<size_t>(qbuf.shape[1]) != dims_) {
      throw std::runtime_error("queries must have shape (nq, dims)");
    }
    if (cbuf.ndim != 2 || static_cast<size_t>(cbuf.shape[1]) != dims_) {
      throw std::runtime_error("centroids must have shape (nlist, dims)");
    }
    if (pbuf.ndim != 2 || static_cast<size_t>(pbuf.shape[0]) != static_cast<size_t>(qbuf.shape[0])) {
      throw std::runtime_error("probe_lists must have shape (nq, nprobe)");
    }
    if (k == 0) {
      const std::vector<py::ssize_t> shape = {
          static_cast<py::ssize_t>(qbuf.shape[0]),
          static_cast<py::ssize_t>(0)};
      return std::make_pair(
          py::array_t<int64_t>(shape),
          py::array_t<float>(shape));
    }

    const size_t nq = static_cast<size_t>(qbuf.shape[0]);
    const size_t nprobe = static_cast<size_t>(pbuf.shape[1]);
    const auto* qptr = static_cast<const float*>(qbuf.ptr);
    const auto* cptr = static_cast<const float*>(cbuf.ptr);
    const auto* pptr = static_cast<const int64_t*>(pbuf.ptr);

    py::array_t<int64_t> labels({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
    py::array_t<float> scores({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
    auto lbuf = labels.request();
    auto sbuf = scores.request();
    auto* lptr = static_cast<int64_t*>(lbuf.ptr);
    auto* sptr = static_cast<float*>(sbuf.ptr);

#pragma omp parallel for num_threads(num_threads_) if (nq > 1)
    for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
      std::vector<float> residual_query(dims_);
      std::vector<std::pair<float, int64_t>> heap;
      heap.reserve(k);

      for (size_t pj = 0; pj < nprobe; ++pj) {
        const int64_t list_no = pptr[qi * nprobe + pj];
        if (list_no < 0 || static_cast<size_t>(list_no) >= nlist_) {
          continue;
        }
        const float* query = qptr + qi * dims_;
        const float* centroid = cptr + static_cast<size_t>(list_no) * dims_;
        compute_residual_avx(query, centroid, residual_query.data(), dims_);
        auto encoded = index_->encode_query(residual_query.data());

        const size_t begin = static_cast<size_t>(invlist_offsets_[static_cast<size_t>(list_no)]);
        const size_t end = static_cast<size_t>(invlist_offsets_[static_cast<size_t>(list_no) + 1]);
        for (size_t pos = begin; pos < end; ++pos) {
          const int64_t id = invlist_ids_[pos];
          const float score = index_->score(encoded, static_cast<osq::idx_t>(id));
          if (heap.size() < k) {
            heap.emplace_back(score, id);
            if (heap.size() == k) {
              std::make_heap(heap.begin(), heap.end(), std::greater<std::pair<float, int64_t>>());
            }
          } else if (score > heap.front().first) {
            std::pop_heap(heap.begin(), heap.end(), std::greater<std::pair<float, int64_t>>());
            heap.back() = {score, id};
            std::push_heap(heap.begin(), heap.end(), std::greater<std::pair<float, int64_t>>());
          }
        }
      }

      std::sort(heap.begin(), heap.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
      });

      for (size_t i = 0; i < k; ++i) {
        const size_t out = static_cast<size_t>(qi) * k + i;
        if (i < heap.size()) {
          lptr[out] = heap[i].second;
          sptr[out] = heap[i].first;
        } else {
          lptr[out] = -1;
          sptr[out] = -std::numeric_limits<float>::infinity();
        }
      }
    }

    return std::make_pair(labels, scores);
  }

  py::array_t<float> reconstruct_all() const {
    const size_t n = index_->ntotal();
    py::array_t<float> output(
        {static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(dims_)});
    auto out = output.request();
    float* ptr = static_cast<float*>(out.ptr);
    for (size_t i = 0; i < n; ++i) {
      index_->reconstruct(static_cast<osq::idx_t>(i), ptr + i * dims_);
    }
    return output;
  }

  float score_residual_query(
      py::array_t<float, py::array::c_style | py::array::forcecast> residual_query,
      int64_t id) const {
    auto buf = residual_query.request();
    if (buf.ndim != 1 || static_cast<size_t>(buf.shape[0]) != dims_) {
      throw std::runtime_error("residual_query must have shape (dims,)");
    }
    auto encoded = index_->encode_query(static_cast<const float*>(buf.ptr));
    return index_->score(encoded, static_cast<osq::idx_t>(id));
  }

  float score_query_with_centroid(
      py::array_t<float, py::array::c_style | py::array::forcecast> query,
      py::array_t<float, py::array::c_style | py::array::forcecast> centroid,
      int64_t id) const {
    auto qbuf = query.request();
    auto cbuf = centroid.request();
    if (qbuf.ndim != 1 || static_cast<size_t>(qbuf.shape[0]) != dims_) {
      throw std::runtime_error("query must have shape (dims,)");
    }
    if (cbuf.ndim != 1 || static_cast<size_t>(cbuf.shape[0]) != dims_) {
      throw std::runtime_error("centroid must have shape (dims,)");
    }
    std::vector<float> residual_query(dims_);
    compute_residual_avx(
        static_cast<const float*>(qbuf.ptr),
        static_cast<const float*>(cbuf.ptr),
        residual_query.data(),
        dims_);
    auto encoded = index_->encode_query(residual_query.data());
    return index_->score(encoded, static_cast<osq::idx_t>(id));
  }

 private:
  size_t dims_;
  std::string space_;
  int nbit_;
  int query_nbit_;
  int num_threads_;
  size_t nlist_ = 0;
  std::unique_ptr<osq::OSQIndex> index_;
  std::vector<int64_t> invlist_offsets_;
  std::vector<int64_t> invlist_ids_;
};

}  // namespace

PYBIND11_MODULE(osq_cpp, m) {
  py::class_<PyOSQIndex>(m, "PyOSQIndex")
      .def(py::init<size_t, const std::string&, int, int>(),
           py::arg("dims"),
           py::arg("space"),
           py::arg("nbit"),
           py::arg("query_nbit") = -1)
      .def("set_num_threads", &PyOSQIndex::set_num_threads)
      .def("train", &PyOSQIndex::train)
      .def("add", &PyOSQIndex::add)
      .def("build", &PyOSQIndex::build)
      .def("search", &PyOSQIndex::search)
      .def("set_query", &PyOSQIndex::set_query)
      .def("score_id", &PyOSQIndex::score_id)
      .def("score_with_query", &PyOSQIndex::score_with_query)
      .def("search_subset", &PyOSQIndex::search_subset)
      .def("reconstruct", &PyOSQIndex::reconstruct)
      .def("reconstruct_all", &PyOSQIndex::reconstruct_all)
      .def_property_readonly("ntotal", &PyOSQIndex::ntotal)
      .def_property_readonly("d", &PyOSQIndex::d)
      .def_property_readonly("nbit", &PyOSQIndex::nbit)
      .def_property_readonly("query_nbit", &PyOSQIndex::query_nbit)
      .def_property_readonly("space", &PyOSQIndex::space);

  py::class_<PyIVFOSQIndex>(m, "PyIVFOSQIndex")
      .def(py::init<size_t, const std::string&, int, int>(),
           py::arg("dims"),
           py::arg("space"),
           py::arg("nbit"),
           py::arg("query_nbit") = -1)
      .def("set_num_threads", &PyIVFOSQIndex::set_num_threads)
      .def("build", &PyIVFOSQIndex::build)
      .def("search_preassigned", &PyIVFOSQIndex::search_preassigned)
      .def("reconstruct_all", &PyIVFOSQIndex::reconstruct_all)
      .def("score_residual_query", &PyIVFOSQIndex::score_residual_query)
      .def("score_query_with_centroid", &PyIVFOSQIndex::score_query_with_centroid);
}
