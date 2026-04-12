#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <faiss/VectorTransform.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/code_distance/code_distance.h>
#include <faiss/utils/Heap.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace py = pybind11;

namespace {

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

template <typename DecoderT>
inline float distance_single_code_faiss(
    const size_t M,
    const size_t nbits,
    const float* sim_table,
    const uint8_t* code) {
  return faiss::distance_single_code<DecoderT>(M, nbits, sim_table, code);
}

template <typename DecoderT>
inline void distance_four_codes_faiss(
    const size_t M,
    const size_t nbits,
    const float* sim_table,
    const uint8_t* code0,
    const uint8_t* code1,
    const uint8_t* code2,
    const uint8_t* code3,
    float& result0,
    float& result1,
    float& result2,
    float& result3) {
  faiss::distance_four_codes<DecoderT>(
      M,
      nbits,
      sim_table,
      code0,
      code1,
      code2,
      code3,
      result0,
      result1,
      result2,
      result3);
}

}  // namespace

class ResidualOpqIvfpqIndex {
 public:
  ResidualOpqIvfpqIndex(size_t dims, size_t d_out, size_t m, size_t nbits)
      : dims_(dims),
        d_out_(d_out),
        m_(m),
        nbits_(nbits),
        code_size_((m * nbits + 7) / 8),
        num_threads_(1),
        opq_(static_cast<int>(dims), static_cast<int>(d_out), false),
        pq_(d_out, m, nbits) {
    if (m_ == 0 || d_out_ == 0 || d_out_ % m_ != 0) {
      throw std::invalid_argument("d_out must be divisible by m");
    }
    if (!(nbits_ == 4 || nbits_ == 8)) {
      throw std::invalid_argument("only 4-bit and 8-bit PQ codes are supported");
    }
  }

  void set_num_threads(int n) {
    num_threads_ = std::max(1, n);
  }

  void build(
      py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
      py::array_t<float, py::array::c_style | py::array::forcecast> opq_matrix,
      py::array_t<float, py::array::c_style | py::array::forcecast> pq_centroids,
      py::array_t<float, py::array::c_style | py::array::forcecast> transformed_residuals,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast> list_offsets,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast> list_ids) {
    auto cbuf = centroids.request();
    auto abuf = opq_matrix.request();
    auto pqbuf = pq_centroids.request();
    auto tbuf = transformed_residuals.request();
    auto obuf = list_offsets.request();
    auto ibuf = list_ids.request();

    if (cbuf.ndim != 2 || static_cast<size_t>(cbuf.shape[1]) != dims_) {
      throw std::runtime_error("centroids must have shape (nlist, dims)");
    }
    if (abuf.ndim != 2 || static_cast<size_t>(abuf.shape[0]) != d_out_ ||
        static_cast<size_t>(abuf.shape[1]) != dims_) {
      throw std::runtime_error("opq_matrix must have shape (d_out, dims)");
    }
    if (pqbuf.ndim != 1 ||
        static_cast<size_t>(pqbuf.shape[0]) != m_ * (size_t(1) << nbits_) * (d_out_ / m_)) {
      throw std::runtime_error("pq_centroids must be a flat array of length M * ksub * dsub");
    }
    if (tbuf.ndim != 2 || static_cast<size_t>(tbuf.shape[1]) != d_out_) {
      throw std::runtime_error("transformed_residuals must have shape (ntotal, d_out)");
    }
    if (obuf.ndim != 1 || ibuf.ndim != 1) {
      throw std::runtime_error("list_offsets and list_ids must be 1D arrays");
    }
    if (static_cast<size_t>(obuf.shape[0]) < 1) {
      throw std::runtime_error("list_offsets must contain at least one element");
    }
    const auto* offsets = static_cast<const int64_t*>(obuf.ptr);
    if (offsets[static_cast<size_t>(obuf.shape[0]) - 1] != ibuf.shape[0]) {
      throw std::runtime_error("last list_offsets entry must equal list_ids length");
    }
    if (static_cast<size_t>(tbuf.shape[0]) != static_cast<size_t>(ibuf.shape[0])) {
      throw std::runtime_error("transformed_residuals row count must match list_ids length");
    }

    nlist_ = static_cast<size_t>(cbuf.shape[0]);
    ntotal_ = static_cast<size_t>(ibuf.shape[0]);

    centroids_.resize(static_cast<size_t>(cbuf.size));
    std::memcpy(
        centroids_.data(),
        static_cast<const float*>(cbuf.ptr),
        static_cast<size_t>(cbuf.size) * sizeof(float));
    list_offsets_.assign(offsets, offsets + obuf.shape[0]);
    list_ids_.assign(
        static_cast<const int64_t*>(ibuf.ptr),
        static_cast<const int64_t*>(ibuf.ptr) + ibuf.shape[0]);

    faiss::LinearTransform opq_local(
        static_cast<int>(dims_),
        static_cast<int>(d_out_),
        false);
    opq_local.A.resize(static_cast<size_t>(abuf.size));
    std::memcpy(
        opq_local.A.data(),
        static_cast<const float*>(abuf.ptr),
        static_cast<size_t>(abuf.size) * sizeof(float));
    opq_local.b.clear();
    opq_local.is_trained = true;
    opq_local.is_orthonormal = true;
    faiss::ProductQuantizer pq_local(d_out_, m_, nbits_);
    pq_local.centroids.resize(static_cast<size_t>(pqbuf.size));
    std::memcpy(
        pq_local.centroids.data(),
        static_cast<const float*>(pqbuf.ptr),
        static_cast<size_t>(pqbuf.size) * sizeof(float));
    pq_local.sync_transposed_centroids();

    codes_.resize(ntotal_ * code_size_);
    pq_local.compute_codes(
        static_cast<const float*>(tbuf.ptr),
        codes_.data(),
        ntotal_);
    opq_ = std::move(opq_local);
    pq_ = std::move(pq_local);

    const int64_t max_id =
        *std::max_element(list_ids_.begin(), list_ids_.end());
    id_to_offset_.assign(static_cast<size_t>(max_id + 1), -1);
    id_to_list_.assign(static_cast<size_t>(max_id + 1), -1);
    for (size_t list_no = 0; list_no < nlist_; ++list_no) {
      const size_t begin = static_cast<size_t>(list_offsets_[list_no]);
      const size_t end = static_cast<size_t>(list_offsets_[list_no + 1]);
      for (size_t pos = begin; pos < end; ++pos) {
        const int64_t id = list_ids_[pos];
        id_to_offset_[static_cast<size_t>(id)] = static_cast<int64_t>(pos);
        id_to_list_[static_cast<size_t>(id)] = static_cast<int64_t>(list_no);
      }
    }

    current_query_.assign(dims_, 0.0f);
    current_query_valid_ = false;
  }

  std::pair<py::array_t<int64_t>, py::array_t<float>> search_preassigned(
      py::array_t<float, py::array::c_style | py::array::forcecast> queries,
      py::array_t<int64_t, py::array::c_style | py::array::forcecast> probe_lists,
      size_t k) const {
    auto qbuf = queries.request();
    auto pbuf = probe_lists.request();
    if (qbuf.ndim != 2 || static_cast<size_t>(qbuf.shape[1]) != dims_) {
      throw std::runtime_error("queries must have shape (nq, dims)");
    }
    if (pbuf.ndim != 2 || static_cast<size_t>(pbuf.shape[0]) != static_cast<size_t>(qbuf.shape[0])) {
      throw std::runtime_error("probe_lists must have shape (nq, nprobe)");
    }

    const size_t nq = static_cast<size_t>(qbuf.shape[0]);
    const size_t nprobe = static_cast<size_t>(pbuf.shape[1]);
    const float* qptr = static_cast<const float*>(qbuf.ptr);
    const int64_t* pptr = static_cast<const int64_t*>(pbuf.ptr);

    py::array_t<int64_t> labels({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
    py::array_t<float> distances({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
    auto lbuf = labels.request();
    auto dbuf = distances.request();
    auto* lptr = static_cast<int64_t*>(lbuf.ptr);
    auto* dptr = static_cast<float*>(dbuf.ptr);

    if (k == 0) {
      return std::make_pair(labels, distances);
    }

    using ResultHeap = faiss::CMax<float, int64_t>;

#pragma omp parallel num_threads(num_threads_) if (nq > 1)
    {
      std::vector<float> residual_query(dims_);
      std::vector<float> transformed_query(d_out_);
      std::vector<float> distance_table(m_ * pq_.ksub);

#pragma omp for
      for (int64_t qi = 0; qi < static_cast<int64_t>(nq); ++qi) {
        const float* query = qptr + static_cast<size_t>(qi) * dims_;
        float* heap_dis = dptr + static_cast<size_t>(qi) * k;
        int64_t* heap_ids = lptr + static_cast<size_t>(qi) * k;

        faiss::heap_heapify<ResultHeap>(k, heap_dis, heap_ids);

        for (size_t pj = 0; pj < nprobe; ++pj) {
          const int64_t list_no = pptr[static_cast<size_t>(qi) * nprobe + pj];
          if (list_no < 0 || static_cast<size_t>(list_no) >= nlist_) {
            continue;
          }
          const float* centroid =
              centroids_.data() + static_cast<size_t>(list_no) * dims_;
          compute_residual_avx(query, centroid, residual_query.data(), dims_);
          opq_.apply_noalloc(1, residual_query.data(), transformed_query.data());
          pq_.compute_distance_table(transformed_query.data(), distance_table.data());
          scan_list(
              static_cast<size_t>(list_no),
              distance_table.data(),
              k,
              heap_dis,
              heap_ids);
        }

        faiss::heap_reorder<ResultHeap>(k, heap_dis, heap_ids);
      }
    }

    return std::make_pair(labels, distances);
  }

  void set_query(py::array_t<float, py::array::c_style | py::array::forcecast> query) {
    auto qbuf = query.request();
    if (qbuf.ndim != 1 || static_cast<size_t>(qbuf.shape[0]) != dims_) {
      throw std::runtime_error("query must have shape (dims,)");
    }
    current_query_.assign(
        static_cast<const float*>(qbuf.ptr),
        static_cast<const float*>(qbuf.ptr) + dims_);
    current_query_valid_ = true;
  }

  float score_id(int64_t id) const {
    if (!current_query_valid_) {
      throw std::runtime_error("set_query must be called before score_id");
    }
    if (id < 0 || static_cast<size_t>(id) >= id_to_offset_.size() ||
        id_to_offset_[static_cast<size_t>(id)] < 0) {
      throw std::runtime_error("invalid id");
    }

    const size_t list_no = static_cast<size_t>(id_to_list_[static_cast<size_t>(id)]);
    const size_t offset = static_cast<size_t>(id_to_offset_[static_cast<size_t>(id)]);
    std::vector<float> residual_query(dims_);
    std::vector<float> transformed_query(d_out_);
    std::vector<float> distance_table(m_ * pq_.ksub);

    const float* centroid = centroids_.data() + list_no * dims_;
    compute_residual_avx(current_query_.data(), centroid, residual_query.data(), dims_);
    opq_.apply_noalloc(1, residual_query.data(), transformed_query.data());
    pq_.compute_distance_table(transformed_query.data(), distance_table.data());
    return score_code(offset, distance_table.data());
  }

  py::array_t<float> reconstruct_all() const {
    py::array_t<float> output(
        {static_cast<py::ssize_t>(ntotal_), static_cast<py::ssize_t>(dims_)});
    auto out = output.request();
    float* out_ptr = static_cast<float*>(out.ptr);

    std::vector<float> transformed_recons(ntotal_ * d_out_);
    std::vector<float> residual_recons(ntotal_ * dims_);
    pq_.decode(codes_.data(), transformed_recons.data(), ntotal_);
    opq_.reverse_transform(ntotal_, transformed_recons.data(), residual_recons.data());

    std::fill(out_ptr, out_ptr + ntotal_ * dims_, 0.0f);
    for (size_t pos = 0; pos < ntotal_; ++pos) {
      const int64_t id = list_ids_[pos];
      const size_t list_no = static_cast<size_t>(id_to_list_[static_cast<size_t>(id)]);
      const float* centroid = centroids_.data() + list_no * dims_;
      const float* residual = residual_recons.data() + pos * dims_;
      float* dst = out_ptr + static_cast<size_t>(id) * dims_;
      for (size_t j = 0; j < dims_; ++j) {
        dst[j] = residual[j] + centroid[j];
      }
    }

    return output;
  }

 private:
  float score_code(size_t offset, const float* distance_table) const {
    const uint8_t* code = codes_.data() + offset * code_size_;
    if (nbits_ == 8) {
      return distance_single_code_faiss<faiss::PQDecoder8>(
          m_, nbits_, distance_table, code);
    }
    return distance_single_code_faiss<faiss::PQDecoderGeneric>(
        m_, nbits_, distance_table, code);
  }

  void scan_list(
      size_t list_no,
      const float* distance_table,
      size_t k,
      float* heap_dis,
      int64_t* heap_ids) const {
    const size_t begin = static_cast<size_t>(list_offsets_[list_no]);
    const size_t end = static_cast<size_t>(list_offsets_[list_no + 1]);
    size_t pos = begin;
    if (nbits_ == 8) {
      for (; pos + 4 <= end; pos += 4) {
        const uint8_t* code0 = codes_.data() + pos * code_size_;
        const uint8_t* code1 = code0 + code_size_;
        const uint8_t* code2 = code1 + code_size_;
        const uint8_t* code3 = code2 + code_size_;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        distance_four_codes_faiss<faiss::PQDecoder8>(
            m_, nbits_, distance_table, code0, code1, code2, code3, d0, d1, d2, d3);
        push_candidate(pos + 0, d0, k, heap_dis, heap_ids);
        push_candidate(pos + 1, d1, k, heap_dis, heap_ids);
        push_candidate(pos + 2, d2, k, heap_dis, heap_ids);
        push_candidate(pos + 3, d3, k, heap_dis, heap_ids);
      }
    } else {
      for (; pos + 4 <= end; pos += 4) {
        const uint8_t* code0 = codes_.data() + pos * code_size_;
        const uint8_t* code1 = code0 + code_size_;
        const uint8_t* code2 = code1 + code_size_;
        const uint8_t* code3 = code2 + code_size_;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        distance_four_codes_faiss<faiss::PQDecoderGeneric>(
            m_, nbits_, distance_table, code0, code1, code2, code3, d0, d1, d2, d3);
        push_candidate(pos + 0, d0, k, heap_dis, heap_ids);
        push_candidate(pos + 1, d1, k, heap_dis, heap_ids);
        push_candidate(pos + 2, d2, k, heap_dis, heap_ids);
        push_candidate(pos + 3, d3, k, heap_dis, heap_ids);
      }
    }
    for (; pos < end; ++pos) {
      push_candidate(pos, score_code(pos, distance_table), k, heap_dis, heap_ids);
    }
  }

  void push_candidate(
      size_t offset,
      float dist,
      size_t k,
      float* heap_dis,
      int64_t* heap_ids) const {
    const int64_t id = list_ids_[offset];
    if (dist < heap_dis[0]) {
      faiss::heap_replace_top<faiss::CMax<float, int64_t>>(
          k, heap_dis, heap_ids, dist, id);
    }
  }

  size_t dims_;
  size_t d_out_;
  size_t m_;
  size_t nbits_;
  size_t code_size_;
  int num_threads_;

  size_t nlist_ = 0;
  size_t ntotal_ = 0;

  faiss::LinearTransform opq_;
  faiss::ProductQuantizer pq_;

  std::vector<float> centroids_;
  std::vector<uint8_t> codes_;
  std::vector<int64_t> list_offsets_;
  std::vector<int64_t> list_ids_;
  std::vector<int64_t> id_to_offset_;
  std::vector<int64_t> id_to_list_;

  std::vector<float> current_query_;
  bool current_query_valid_;
};

PYBIND11_MODULE(residual_opq_ivfpq_cpp, m) {
  py::class_<ResidualOpqIvfpqIndex>(m, "ResidualOpqIvfpqIndex")
      .def(py::init<size_t, size_t, size_t, size_t>())
      .def("set_num_threads", &ResidualOpqIvfpqIndex::set_num_threads)
      .def("build", &ResidualOpqIvfpqIndex::build)
      .def("search_preassigned", &ResidualOpqIvfpqIndex::search_preassigned)
      .def("set_query", &ResidualOpqIvfpqIndex::set_query)
      .def("score_id", &ResidualOpqIvfpqIndex::score_id)
      .def("reconstruct_all", &ResidualOpqIvfpqIndex::reconstruct_all);
}
