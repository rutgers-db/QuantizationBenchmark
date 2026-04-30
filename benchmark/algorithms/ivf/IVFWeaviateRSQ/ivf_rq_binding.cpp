// IVF wrapper around Weaviate Rotational Quantization (uniform RQ + BRQ).
//
// Like the BQ wrapper, the caller pre-clusters and supplies residuals in
// inverted-list order. The wrapper trains a single flat RotationalQuantizer
// on the residuals; at query time, each probed list's residual query is
// encoded once and scored only against that list's ids.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "rq.h"

namespace py = pybind11;

namespace {

rq::Metric parse_metric(const std::string& s) {
    if (s == "l2" || s == "L2") return rq::Metric::L2;
    if (s == "ip" || s == "IP") return rq::Metric::IP;
    throw std::invalid_argument("metric must be 'l2' or 'ip'");
}

inline void compute_residual(const float* q, const float* c, float* out, std::size_t d) {
#if defined(__AVX__)
    std::size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        _mm256_storeu_ps(out + i,
            _mm256_sub_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(c + i)));
    }
    for (; i < d; ++i) out[i] = q[i] - c[i];
#else
    for (std::size_t i = 0; i < d; ++i) out[i] = q[i] - c[i];
#endif
}

}  // namespace

class PyIVFRQIndex {
 public:
    PyIVFRQIndex(int d, int bits, const std::string& metric,
                 std::uint64_t seed)
        : d_(d),
          metric_(parse_metric(metric)),
          index_(std::make_unique<rq::RotationalQuantizer>(
              d, bits, parse_metric(metric), seed)) {}

    void set_num_threads(int n) {
        num_threads_ = std::max(1, n);
        index_->set_num_threads(num_threads_);
    }

    void build(py::array_t<float, py::array::c_style | py::array::forcecast> residuals,
               py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> list_offsets,
               py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> list_ids) {
        auto rbuf = residuals.request();
        auto obuf = list_offsets.request();
        auto ibuf = list_ids.request();
        if (rbuf.ndim != 2 || static_cast<int>(rbuf.shape[1]) != d_)
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
        // Train is a no-op for RQ (data-independent rotation), but kept for
        // interface symmetry.
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
        if (qbuf.ndim != 2 || static_cast<int>(qbuf.shape[1]) != d_)
            throw std::runtime_error("queries must have shape (nq, dims)");
        if (cbuf.ndim != 2 || static_cast<int>(cbuf.shape[1]) != d_)
            throw std::runtime_error("centroids must have shape (nlist, dims)");
        if (pbuf.ndim != 2 || pbuf.shape[0] != qbuf.shape[0])
            throw std::runtime_error("probe_lists must have shape (nq, nprobe)");

        const std::size_t nq     = static_cast<std::size_t>(qbuf.shape[0]);
        const std::size_t nprobe = static_cast<std::size_t>(pbuf.shape[1]);
        const std::size_t dims   = static_cast<std::size_t>(d_);
        const float* qptr = static_cast<const float*>(qbuf.ptr);
        const float* cptr = static_cast<const float*>(cbuf.ptr);
        const auto*  pptr = static_cast<const std::int64_t*>(pbuf.ptr);

        py::array_t<std::int64_t> labels({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
        py::array_t<float>        dists ({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
        auto* lout = static_cast<std::int64_t*>(labels.request().ptr);
        auto* dout = static_cast<float*>(dists.request().ptr);

        if (k == 0) return {labels, dists};

        const bool ip = (metric_ == rq::Metric::IP);
        const float pad = ip ? -std::numeric_limits<float>::infinity()
                             :  std::numeric_limits<float>::infinity();

        auto better = [ip](float a, float b) { return ip ? (a > b) : (a < b); };

#pragma omp parallel for num_threads(num_threads_) if (nq > 1)
        for (long long qi = 0; qi < static_cast<long long>(nq); ++qi) {
            std::vector<float> residual(dims);
            std::vector<std::pair<float, std::int64_t>> heap;
            heap.reserve(k);

            auto heap_cmp = [ip](const std::pair<float, std::int64_t>& a,
                                 const std::pair<float, std::int64_t>& b) {
                return ip ? (a.first > b.first) : (a.first < b.first);
            };

            for (std::size_t pj = 0; pj < nprobe; ++pj) {
                const std::int64_t list_no = pptr[qi * nprobe + pj];
                if (list_no < 0 || static_cast<std::size_t>(list_no) >= nlist_) continue;
                const float* q = qptr + qi * dims;
                const float* c = cptr + static_cast<std::size_t>(list_no) * dims;
                compute_residual(q, c, residual.data(), dims);
                auto eq = index_->encode_query_full(residual.data());

                const std::size_t begin = static_cast<std::size_t>(invlist_offsets_[list_no]);
                const std::size_t end   = static_cast<std::size_t>(invlist_offsets_[list_no + 1]);
                for (std::size_t pos = begin; pos < end; ++pos) {
                    const std::int64_t id = invlist_ids_[pos];
                    const float dist = index_->score_one(eq, pos);
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

    std::size_t ntotal()    const { return index_->ntotal(); }
    std::size_t code_size() const { return index_->code_size(); }
    int         get_dim()   const { return index_->d(); }
    int         out_dim()   const { return index_->out_dim(); }
    int         bits()      const { return index_->bits(); }

 private:
    int        d_;
    rq::Metric metric_;
    int        num_threads_ = 1;
    std::size_t nlist_ = 0;
    std::unique_ptr<rq::RotationalQuantizer> index_;
    std::vector<std::int64_t> invlist_offsets_;
    std::vector<std::int64_t> invlist_ids_;
};

PYBIND11_MODULE(ivf_rq_cpp, m) {
    m.doc() = "IVF wrapper around Weaviate Rotational Quantization";

    py::class_<PyIVFRQIndex>(m, "PyIVFRQIndex")
        .def(py::init<int, int, const std::string&, std::uint64_t>(),
             py::arg("d"),
             py::arg("bits")   = 8,
             py::arg("metric") = "l2",
             py::arg("seed")   = 0x517cc1b727220a95ULL)
        .def("set_num_threads",    &PyIVFRQIndex::set_num_threads, py::arg("n"))
        .def("build",              &PyIVFRQIndex::build,
             py::arg("residuals"), py::arg("list_offsets"), py::arg("list_ids"))
        .def("search_preassigned", &PyIVFRQIndex::search_preassigned,
             py::arg("queries"), py::arg("centroids"),
             py::arg("probe_lists"), py::arg("k"))
        .def("ntotal",             &PyIVFRQIndex::ntotal)
        .def("code_size",          &PyIVFRQIndex::code_size)
        .def("get_dim",            &PyIVFRQIndex::get_dim)
        .def("out_dim",            &PyIVFRQIndex::out_dim)
        .def("bits",               &PyIVFRQIndex::bits);
}
