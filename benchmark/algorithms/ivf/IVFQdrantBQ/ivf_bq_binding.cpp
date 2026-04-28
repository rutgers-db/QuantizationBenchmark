// IVF wrapper around Qdrant Binary Quantization.
//
// The wrapper assumes the caller has already:
//   1. Built coarse (k-means) centroids over the training data.
//   2. Assigned every db vector to its nearest centroid.
//   3. Computed residual = x - centroid[assignment].
//   4. Sorted residuals so they are grouped by inverted list (passed in via
//      list_offsets / list_ids).
//
// This module then trains a single flat BinaryQuantizer over the residuals
// and, at query time, encodes each probed list's residual query separately
// before scoring against just that list's ids.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__AVX__)
#include <immintrin.h>
#endif

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

// 64-byte aligned scratch for one encoded query. The BQ AVX-512 / AVX-2
// kernels use aligned loads, so unaligned scratch would fault.
struct AlignedQCode {
    uint64_t* p = nullptr;
    size_t    n = 0;

    explicit AlignedQCode(size_t n_words) : n(n_words) {
        if (n == 0) return;
        size_t bytes = ((n * sizeof(uint64_t)) + 63u) & ~size_t{63u};
        void* raw = nullptr;
        if (posix_memalign(&raw, 64, bytes) != 0) throw std::bad_alloc();
        p = static_cast<uint64_t*>(raw);
    }
    ~AlignedQCode() { if (p) std::free(p); }
    AlignedQCode(const AlignedQCode&) = delete;
    AlignedQCode& operator=(const AlignedQCode&) = delete;
};

inline void compute_residual(const float* q, const float* c, float* out, size_t d) {
#if defined(__AVX__)
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_sub_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(c + i)));
    }
    for (; i < d; ++i) out[i] = q[i] - c[i];
#else
    for (size_t i = 0; i < d; ++i) out[i] = q[i] - c[i];
#endif
}

}  // namespace

class PyIVFBQIndex {
 public:
    PyIVFBQIndex(std::size_t d,
                 int encoding,
                 const std::string& query_encoding,
                 const std::string& metric)
        : d_(d),
          metric_(parse_metric(metric)),
          index_(std::make_unique<bq::BinaryQuantizer>(
              d, parse_encoding(encoding),
              parse_query_encoding(query_encoding),
              parse_metric(metric))) {}

    void set_num_threads(int n) {
        num_threads_ = std::max(1, n);
        index_->set_num_threads(num_threads_);
    }

    // DB codes are independent of the query-side encoding; switching it
    // post-build is safe and reuses the existing index.
    void set_query_encoding(const std::string& qe) {
        index_->set_query_encoding(parse_query_encoding(qe));
    }

    // Train + encode the residuals, then take ownership of the inverted-list
    // metadata. `residuals` must already be ordered to match list_ids[].
    void build(py::array_t<float, py::array::c_style | py::array::forcecast> residuals,
               py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> list_offsets,
               py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> list_ids) {
        auto rbuf = residuals.request();
        auto obuf = list_offsets.request();
        auto ibuf = list_ids.request();
        if (rbuf.ndim != 2 || static_cast<std::size_t>(rbuf.shape[1]) != d_)
            throw std::runtime_error("residuals must have shape (n, dims)");
        if (obuf.ndim != 1 || ibuf.ndim != 1)
            throw std::runtime_error("list_offsets and list_ids must be 1D arrays");
        if (obuf.shape[0] < 1)
            throw std::runtime_error("list_offsets must contain at least one element");
        const auto* offsets = static_cast<const std::int64_t*>(obuf.ptr);
        if (offsets[obuf.shape[0] - 1] != ibuf.shape[0])
            throw std::runtime_error("last list_offsets entry must equal list_ids length");

        nlist_ = static_cast<std::size_t>(obuf.shape[0] - 1);
        invlist_offsets_.assign(offsets, offsets + obuf.shape[0]);
        invlist_ids_.assign(static_cast<const std::int64_t*>(ibuf.ptr),
                            static_cast<const std::int64_t*>(ibuf.ptr) + ibuf.shape[0]);

        const std::size_t n = static_cast<std::size_t>(rbuf.shape[0]);
        const float* rptr = static_cast<const float*>(rbuf.ptr);
        index_->train(n, rptr);
        index_->add  (n, rptr);
    }

    std::pair<py::array_t<std::int64_t>, py::array_t<float>>
    search_preassigned(
        py::array_t<float, py::array::c_style | py::array::forcecast> queries,
        py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
        py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> probe_lists,
        std::size_t k) const {
        auto qbuf = queries.request();
        auto cbuf = centroids.request();
        auto pbuf = probe_lists.request();
        if (qbuf.ndim != 2 || static_cast<std::size_t>(qbuf.shape[1]) != d_)
            throw std::runtime_error("queries must have shape (nq, dims)");
        if (cbuf.ndim != 2 || static_cast<std::size_t>(cbuf.shape[1]) != d_)
            throw std::runtime_error("centroids must have shape (nlist, dims)");
        if (pbuf.ndim != 2 || pbuf.shape[0] != qbuf.shape[0])
            throw std::runtime_error("probe_lists must have shape (nq, nprobe)");

        const std::size_t nq     = static_cast<std::size_t>(qbuf.shape[0]);
        const std::size_t nprobe = static_cast<std::size_t>(pbuf.shape[1]);
        const std::size_t qwords = index_->query_code_words();
        const float* qptr = static_cast<const float*>(qbuf.ptr);
        const float* cptr = static_cast<const float*>(cbuf.ptr);
        const auto*  pptr = static_cast<const std::int64_t*>(pbuf.ptr);

        py::array_t<std::int64_t> labels({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
        py::array_t<float>        dists ({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
        auto* lout = static_cast<std::int64_t*>(labels.request().ptr);
        auto* dout = static_cast<float*>(dists.request().ptr);

        const bool ip = (metric_ == bq::Metric::IP);
        const float pad = ip ? -std::numeric_limits<float>::infinity()
                             :  std::numeric_limits<float>::infinity();

        if (k == 0) return {labels, dists};

        // For L2/Hamming we want top-k smallest → max-heap by distance.
        // For IP we want top-k largest      → min-heap by distance.
        auto better = [ip](float a, float b) { return ip ? (a > b) : (a < b); };

#pragma omp parallel for num_threads(num_threads_) if (nq > 1)
        for (long long qi = 0; qi < static_cast<long long>(nq); ++qi) {
            std::vector<float> residual(d_);
            AlignedQCode qcode(qwords);
            std::vector<float> list_dists;   // per-list scratch, reused across probes
            std::vector<std::pair<float, std::int64_t>> heap;
            heap.reserve(k);

            // Heap orientation: top() must be the *worst* of the current
            // top-k. For "smaller=better" → max-heap → comparator a<b.
            // For "larger=better"  → min-heap → comparator a>b.
            auto heap_cmp = [ip](const std::pair<float, std::int64_t>& a,
                                 const std::pair<float, std::int64_t>& b) {
                return ip ? (a.first > b.first) : (a.first < b.first);
            };

            for (std::size_t pj = 0; pj < nprobe; ++pj) {
                const std::int64_t list_no = pptr[qi * nprobe + pj];
                if (list_no < 0 || static_cast<std::size_t>(list_no) >= nlist_) continue;
                const float* q = qptr + qi * d_;
                const float* c = cptr + static_cast<std::size_t>(list_no) * d_;
                compute_residual(q, c, residual.data(), d_);
                index_->encode_query(residual.data(), qcode.p);

                const std::size_t begin = static_cast<std::size_t>(invlist_offsets_[list_no]);
                const std::size_t end   = static_cast<std::size_t>(invlist_offsets_[list_no + 1]);
                const std::size_t list_size = end - begin;
                if (list_size == 0) continue;
                if (list_dists.size() < list_size) list_dists.resize(list_size);
                index_->score_range(qcode.p, begin, end, list_dists.data());
                for (std::size_t m = 0; m < list_size; ++m) {
                    const std::int64_t id = invlist_ids_[begin + m];
                    const float dist = list_dists[m];
                    if (heap.size() < k) {
                        heap.emplace_back(dist, id);
                        if (heap.size() == k) std::make_heap(heap.begin(), heap.end(), heap_cmp);
                    } else if (better(dist, heap.front().first)) {
                        std::pop_heap(heap.begin(), heap.end(), heap_cmp);
                        heap.back() = {dist, id};
                        std::push_heap(heap.begin(), heap.end(), heap_cmp);
                    }
                }
            }

            std::sort(heap.begin(), heap.end(),
                      [ip](const auto& a, const auto& b) {
                          return ip ? (a.first > b.first) : (a.first < b.first);
                      });

            for (std::size_t i = 0; i < k; ++i) {
                const std::size_t out = static_cast<std::size_t>(qi) * k + i;
                if (i < heap.size()) {
                    lout[out] = heap[i].second;
                    dout[out] = heap[i].first;
                } else {
                    lout[out] = -1;
                    dout[out] = pad;
                }
            }
        }

        return {labels, dists};
    }

    std::size_t ntotal()    const { return index_->ntotal; }
    std::size_t code_size() const { return index_->code_size; }
    std::size_t get_dim()   const { return d_; }
    int         k_db()      const { return index_->k_db(); }

 private:
    std::size_t d_;
    bq::Metric  metric_;
    int         num_threads_ = 1;
    std::size_t nlist_ = 0;
    std::unique_ptr<bq::BinaryQuantizer> index_;
    // The wrapper stores residuals in inverted-list order, so the local
    // "encoded position" pos directly indexes into both the BQ codes and
    // invlist_ids_ (which maps it back to the user-facing global id).
    std::vector<std::int64_t> invlist_offsets_;
    std::vector<std::int64_t> invlist_ids_;
};

PYBIND11_MODULE(ivf_bq_cpp, m) {
    m.doc() = "IVF wrapper around Qdrant Binary Quantization";

    py::class_<PyIVFBQIndex>(m, "PyIVFBQIndex")
        .def(py::init<std::size_t, int, const std::string&, const std::string&>(),
             py::arg("d"),
             py::arg("encoding")       = 1,
             py::arg("query_encoding") = "same",
             py::arg("metric")         = "l2")
        .def("set_num_threads",    &PyIVFBQIndex::set_num_threads,    py::arg("n"))
        .def("set_query_encoding", &PyIVFBQIndex::set_query_encoding, py::arg("query_encoding"))
        .def("build",              &PyIVFBQIndex::build,
             py::arg("residuals"), py::arg("list_offsets"), py::arg("list_ids"))
        .def("search_preassigned", &PyIVFBQIndex::search_preassigned,
             py::arg("queries"), py::arg("centroids"),
             py::arg("probe_lists"), py::arg("k"))
        .def("ntotal",             &PyIVFBQIndex::ntotal)
        .def("code_size",          &PyIVFBQIndex::code_size)
        .def("get_dim",            &PyIVFBQIndex::get_dim)
        .def("k_db",               &PyIVFBQIndex::k_db);
}
