// pybind11 binding for a self-contained IVF + E8-lattice 1-bit index.
//
// Quantization pipeline mirrors RaBitQ 1-bit: FHT-Kac rotation, residual from
// cluster centroid, RaBitQ factors (f_add, f_rescale, f_error). The residual
// is quantized per-8-dim-block to one of the 240 E8 root vectors, packed as
// 1 byte per block (d/8 bytes per vector — same budget as RaBitQ 1-bit).
//
// Search path: per-query 256-entry LUT per block. Codes are stored in
// block-major tiles of 16 vectors so that one `_mm512_i32gather_ps` loads
// the LUT contribution for 16 vectors at a time; the inner product is
// accumulated in a single zmm register across all blocks. Tail vectors
// (< 16) fall back to the scalar gather path.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/rotator.hpp"
#include "rabitqlib/utils/space.hpp"

#include "e8_codebook.h"

namespace py = pybind11;

using PID = rabitqlib::PID;
using e8lib::kBlockDim;
using e8lib::kCodebookSize;
using e8lib::get_e8_codebook;

namespace {

constexpr float kConstEpsilon = 1.9f;
constexpr size_t kTile = 16;  // AVX-512 tile width

inline float dot8(const float* a, const float* b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] +
           a[4] * b[4] + a[5] * b[5] + a[6] * b[6] + a[7] * b[7];
}

/* Per-list storage. codes are laid out in block-major tiles: for tiles of 16
 * vectors, block b of tile t occupies 16 consecutive bytes at
 *    codes_bm[t * n_blocks * 16 + b * 16 + v].
 * The trailing tail (< 16 vectors) stays in the same layout but with only
 * `tail_size` valid lanes; the scalar tail path reads plain code[b]. */
struct ListStorage {
    std::vector<PID> ids;
    std::vector<uint8_t> codes_bm;       // block-major, 16-wide tiles
    std::vector<float> f_add;
    std::vector<float> f_rescale;
    size_t size() const { return ids.size(); }
};

}  // anonymous

class IVFE8 {
public:
    IVFE8(size_t n, size_t dim, size_t nlist, int nthread,
          const std::string& metric, const std::string& rotator)
        : n_(n), dim_(dim), nlist_(nlist), nthread_(nthread) {

        (void)metric;

        padded_dim_ = rabitqlib::round_up_to_multiple(dim_, 64);
        if (padded_dim_ % kBlockDim != 0) {
            throw std::runtime_error("padded_dim must be multiple of 8");
        }
        n_blocks_ = padded_dim_ / kBlockDim;

        rabitqlib::RotatorType rtype =
            (rotator == "matrix") ? rabitqlib::RotatorType::MatrixRotator
                                  : rabitqlib::RotatorType::FhtKacRotator;
        std::srand(dim_ + padded_dim_);
        rotator_.reset(rabitqlib::choose_rotator<float>(dim_, rtype, padded_dim_));

        rotated_centroids_.assign(nlist_ * padded_dim_, 0.0f);
        centroids_.assign(nlist_ * dim_, 0.0f);
        lists_.resize(nlist_);
    }

    void construct(py::array_t<float, py::array::c_style | py::array::forcecast> data,
                   py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
                   py::array_t<uint32_t, py::array::c_style | py::array::forcecast> cluster_ids) {
        auto d = data.request();
        auto c = centroids.request();
        auto ids = cluster_ids.request();
        if (d.ndim != 2) throw std::runtime_error("data must be 2-D");
        if (c.ndim != 2) throw std::runtime_error("centroids must be 2-D");
        if (ids.ndim != 1) throw std::runtime_error("cluster_ids must be 1-D");

        size_t nb = static_cast<size_t>(d.shape[0]);
        const float* dptr = static_cast<const float*>(d.ptr);
        const float* cptr = static_cast<const float*>(c.ptr);
        const uint32_t* idptr = static_cast<const uint32_t*>(ids.ptr);

        std::memcpy(centroids_.data(), cptr, nlist_ * dim_ * sizeof(float));
        for (size_t i = 0; i < nlist_; ++i) {
            rotator_->rotate(centroids_.data() + i * dim_,
                             rotated_centroids_.data() + i * padded_dim_);
        }

        std::vector<size_t> counts(nlist_, 0);
        for (size_t i = 0; i < nb; ++i) {
            if (idptr[i] >= nlist_) throw std::runtime_error("cluster id out of range");
            ++counts[idptr[i]];
        }
        for (size_t l = 0; l < nlist_; ++l) {
            size_t sz = counts[l];
            size_t padded_sz = ((sz + kTile - 1) / kTile) * kTile;
            lists_[l].ids.resize(sz);
            lists_[l].codes_bm.assign(padded_sz * n_blocks_, 0);
            lists_[l].f_add.assign(padded_sz, 0.0f);
            lists_[l].f_rescale.assign(padded_sz, 0.0f);
        }

        std::vector<size_t> cursor(nlist_, 0);
        std::vector<size_t> pos(nb);
        for (size_t i = 0; i < nb; ++i) {
            uint32_t cid = idptr[i];
            pos[i] = cursor[cid]++;
            lists_[cid].ids[pos[i]] = static_cast<PID>(i);
        }

        omp_set_num_threads(nthread_);
        const float* cb = get_e8_codebook().data();

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim_);
            std::vector<float> residual(padded_dim_);
            std::vector<float> reconstructed(padded_dim_);

            #pragma omp for schedule(static)
            for (int64_t i = 0; i < (int64_t)nb; ++i) {
                uint32_t cid = idptr[i];
                size_t p = pos[i];

                rotator_->rotate(dptr + i * dim_, rotated.data());
                const float* cent_rot = rotated_centroids_.data() + cid * padded_dim_;
                for (size_t k = 0; k < padded_dim_; ++k) {
                    residual[k] = rotated[k] - cent_rot[k];
                }

                // Quantize + build reconstruction for factor computation.
                uint8_t code_scratch[4096 / kBlockDim];  // >= n_blocks_ for padded_dim<=4096
                for (size_t b = 0; b < n_blocks_; ++b) {
                    uint8_t k = get_e8_codebook().encode_block(residual.data() + b * kBlockDim);
                    code_scratch[b] = k;
                    const float* cw = cb + k * kBlockDim;
                    float* rec = reconstructed.data() + b * kBlockDim;
                    for (size_t dd = 0; dd < kBlockDim; ++dd) rec[dd] = cw[dd];
                }

                // Scatter codes into block-major tile layout.
                size_t tile = p / kTile;
                size_t lane = p % kTile;
                uint8_t* tile_base = lists_[cid].codes_bm.data() + tile * kTile * n_blocks_;
                for (size_t b = 0; b < n_blocks_; ++b) {
                    tile_base[b * kTile + lane] = code_scratch[b];
                }

                float l2_sqr = rabitqlib::l2norm_sqr<float>(residual.data(), padded_dim_);
                float ip_resi_qr = rabitqlib::dot_product<float>(residual.data(),
                                                                 reconstructed.data(),
                                                                 padded_dim_);
                float ip_cent_qr = rabitqlib::dot_product<float>(cent_rot,
                                                                 reconstructed.data(),
                                                                 padded_dim_);
                float norm_qr_sq = rabitqlib::l2norm_sqr<float>(reconstructed.data(),
                                                                padded_dim_);

                if (ip_resi_qr == 0.0f) {
                    ip_resi_qr = std::numeric_limits<float>::infinity();
                }

                float f_add     = l2_sqr + (2.0f * l2_sqr * ip_cent_qr / ip_resi_qr);
                float f_rescale = -2.0f * l2_sqr / ip_resi_qr;
                lists_[cid].f_add[p]     = f_add;
                lists_[cid].f_rescale[p] = f_rescale;
            }
        }
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>>
    search_batch(py::array_t<float, py::array::c_style | py::array::forcecast> queries,
                 size_t k, size_t nprobe) {
        auto q = queries.request();
        if (q.ndim != 2) throw std::runtime_error("queries must be 2-D");
        size_t nq = static_cast<size_t>(q.shape[0]);
        const float* qptr = static_cast<const float*>(q.ptr);

        py::array_t<int64_t> I({static_cast<py::ssize_t>(nq),
                                static_cast<py::ssize_t>(k)});
        py::array_t<float> D({static_cast<py::ssize_t>(nq),
                              static_cast<py::ssize_t>(k)});
        int64_t* Iptr = static_cast<int64_t*>(I.request().ptr);
        float* Dptr = static_cast<float*>(D.request().ptr);

        omp_set_num_threads(nthread_);

        const size_t padded_dim = padded_dim_;
        const size_t n_blocks = n_blocks_;

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim);
            std::vector<float> lut(n_blocks * kCodebookSize);
            std::vector<std::pair<float, size_t>> probe_list(nlist_);
            std::vector<float> ip_tile(kTile);

            #pragma omp for schedule(dynamic, 8)
            for (int64_t qi = 0; qi < (int64_t)nq; ++qi) {
                const float* query = qptr + qi * dim_;
                rotator_->rotate(query, rotated.data());

                for (size_t l = 0; l < nlist_; ++l) {
                    float dsq = rabitqlib::euclidean_sqr<float>(
                        rotated.data(),
                        rotated_centroids_.data() + l * padded_dim,
                        padded_dim);
                    probe_list[l] = {dsq, l};
                }
                size_t np = std::min(nprobe, nlist_);
                std::partial_sort(probe_list.begin(),
                                  probe_list.begin() + np,
                                  probe_list.end());

                // Build LUT: for each block, compute 256 inner products
                // <q_b, cb[k]> using the transposed codebook. Each iteration
                // produces 16 LUT entries via 8 broadcast-FMAs.
                const float* cb_T = get_e8_codebook().data_transposed();
                for (size_t b = 0; b < n_blocks; ++b) {
                    const float* qb = rotated.data() + b * kBlockDim;
                    float* row = lut.data() + b * kCodebookSize;
                    // Hoist broadcasts outside the slab loop.
                    __m512 q0 = _mm512_set1_ps(qb[0]);
                    __m512 q1 = _mm512_set1_ps(qb[1]);
                    __m512 q2 = _mm512_set1_ps(qb[2]);
                    __m512 q3 = _mm512_set1_ps(qb[3]);
                    __m512 q4 = _mm512_set1_ps(qb[4]);
                    __m512 q5 = _mm512_set1_ps(qb[5]);
                    __m512 q6 = _mm512_set1_ps(qb[6]);
                    __m512 q7 = _mm512_set1_ps(qb[7]);
                    for (size_t ck = 0; ck < kCodebookSize; ck += 16) {
                        __m512 acc = _mm512_mul_ps(
                            _mm512_load_ps(cb_T + 0 * kCodebookSize + ck), q0);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 1 * kCodebookSize + ck), q1, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 2 * kCodebookSize + ck), q2, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 3 * kCodebookSize + ck), q3, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 4 * kCodebookSize + ck), q4, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 5 * kCodebookSize + ck), q5, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 6 * kCodebookSize + ck), q6, acc);
                        acc = _mm512_fmadd_ps(
                            _mm512_load_ps(cb_T + 7 * kCodebookSize + ck), q7, acc);
                        _mm512_storeu_ps(row + ck, acc);
                    }
                }

                std::priority_queue<std::pair<float, PID>> topk;

                for (size_t pp = 0; pp < np; ++pp) {
                    float g_add = probe_list[pp].first;
                    size_t l = probe_list[pp].second;
                    const auto& L = lists_[l];
                    size_t sz = L.size();
                    if (sz == 0) continue;

                    size_t n_full_tiles = sz / kTile;
                    size_t tail = sz - n_full_tiles * kTile;

                    // ---------------- SIMD full-tile path ----------------
                    const uint8_t* L_codes = L.codes_bm.data();
                    const float* L_fadd = L.f_add.data();
                    const float* L_fres = L.f_rescale.data();
                    const PID* L_ids = L.ids.data();
                    const size_t tile_stride = kTile * n_blocks;
                    // Prefetch first tile before entering the loop.
                    if (n_full_tiles > 0) {
                        for (size_t b = 0; b < n_blocks; b += 4) {
                            _mm_prefetch(reinterpret_cast<const char*>(L_codes + b * kTile),
                                         _MM_HINT_T0);
                        }
                    }
                    for (size_t t = 0; t < n_full_tiles; ++t) {
                        const uint8_t* tile_base = L_codes + t * tile_stride;
                        // Prefetch next tile's codes.
                        if (t + 1 < n_full_tiles) {
                            const uint8_t* next_tile = L_codes + (t + 1) * tile_stride;
                            for (size_t b = 0; b < n_blocks; b += 4) {
                                _mm_prefetch(reinterpret_cast<const char*>(next_tile + b * kTile),
                                             _MM_HINT_T0);
                            }
                        }

                        // Dual accumulators let the OOO core issue two
                        // gathers in flight (gather throughput is sub-1 per
                        // cycle on most Intel cores).
                        __m512 acc0 = _mm512_setzero_ps();
                        __m512 acc1 = _mm512_setzero_ps();
                        size_t b = 0;
                        size_t b_pair_end = (n_blocks / 2) * 2;
                        for (; b < b_pair_end; b += 2) {
                            __m128i c0 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(tile_base + (b + 0) * kTile));
                            __m128i c1 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(tile_base + (b + 1) * kTile));
                            __m512i i0 = _mm512_cvtepu8_epi32(c0);
                            __m512i i1 = _mm512_cvtepu8_epi32(c1);
                            const float* lut0 = lut.data() + (b + 0) * kCodebookSize;
                            const float* lut1 = lut.data() + (b + 1) * kCodebookSize;
                            __m512 v0 = _mm512_i32gather_ps(i0, lut0, 4);
                            __m512 v1 = _mm512_i32gather_ps(i1, lut1, 4);
                            acc0 = _mm512_add_ps(acc0, v0);
                            acc1 = _mm512_add_ps(acc1, v1);
                        }
                        for (; b < n_blocks; ++b) {
                            __m128i codes16 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(tile_base + b * kTile));
                            __m512i idx = _mm512_cvtepu8_epi32(codes16);
                            const float* lut_b = lut.data() + b * kCodebookSize;
                            __m512 vals = _mm512_i32gather_ps(idx, lut_b, 4);
                            acc0 = _mm512_add_ps(acc0, vals);
                        }
                        __m512 acc = _mm512_add_ps(acc0, acc1);
                        __m512 fadd = _mm512_loadu_ps(L_fadd + t * kTile);
                        __m512 fres = _mm512_loadu_ps(L_fres + t * kTile);
                        __m512 est  = _mm512_fmadd_ps(fres, acc,
                                        _mm512_add_ps(_mm512_set1_ps(g_add), fadd));

                        // Bound filter: if heap is full, only lanes below
                        // current top-k bound may evict the max. Skip the
                        // entire tile when no lane qualifies.
                        if (topk.size() >= k) {
                            __m512 bound = _mm512_set1_ps(topk.top().first);
                            __mmask16 win_mask =
                                _mm512_cmp_ps_mask(est, bound, _CMP_LT_OQ);
                            if (win_mask == 0) continue;

                            _mm512_storeu_ps(ip_tile.data(), est);
                            const PID* ids16 = L_ids + t * kTile;
                            while (win_mask) {
                                unsigned v = __builtin_ctz(win_mask);
                                float e = ip_tile[v];
                                if (e < topk.top().first) {
                                    topk.pop();
                                    topk.emplace(e, ids16[v]);
                                }
                                win_mask &= (win_mask - 1);
                            }
                        } else {
                            _mm512_storeu_ps(ip_tile.data(), est);
                            const PID* ids16 = L_ids + t * kTile;
                            for (size_t v = 0; v < kTile; ++v) {
                                float e = ip_tile[v];
                                if (topk.size() < k) {
                                    topk.emplace(e, ids16[v]);
                                } else if (e < topk.top().first) {
                                    topk.pop();
                                    topk.emplace(e, ids16[v]);
                                }
                            }
                        }
                    }

                    // ---------------- scalar tail path ----------------
                    if (tail > 0) {
                        const uint8_t* tile_base =
                            L.codes_bm.data() + n_full_tiles * kTile * n_blocks;
                        for (size_t v = 0; v < tail; ++v) {
                            float ip = 0.0f;
                            for (size_t b = 0; b < n_blocks; ++b) {
                                uint8_t c = tile_base[b * kTile + v];
                                ip += lut[b * kCodebookSize + c];
                            }
                            size_t p = n_full_tiles * kTile + v;
                            float est = g_add + L.f_add[p] + L.f_rescale[p] * ip;
                            PID id = L.ids[p];
                            if (topk.size() < k) {
                                topk.emplace(est, id);
                            } else if (est < topk.top().first) {
                                topk.pop();
                                topk.emplace(est, id);
                            }
                        }
                    }
                }

                // Write top-k results.
                for (size_t j = 0; j < k; ++j) {
                    Iptr[qi * k + j] = -1;
                    Dptr[qi * k + j] = std::numeric_limits<float>::infinity();
                }
                std::vector<std::pair<float, PID>> ordered;
                ordered.reserve(topk.size());
                while (!topk.empty()) {
                    ordered.push_back(topk.top());
                    topk.pop();
                }
                std::reverse(ordered.begin(), ordered.end());
                for (size_t j = 0; j < ordered.size(); ++j) {
                    Iptr[qi * k + j] = static_cast<int64_t>(ordered[j].second);
                    Dptr[qi * k + j] = ordered[j].first;
                }
            }
        }
        return {I, D};
    }

private:
    size_t n_;
    size_t dim_;
    size_t padded_dim_;
    size_t nlist_;
    size_t n_blocks_;
    int nthread_;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    std::vector<float> centroids_;
    std::vector<float> rotated_centroids_;

    std::vector<ListStorage> lists_;
};

PYBIND11_MODULE(e8_cpp, m) {
    m.doc() = "IVF + E8-lattice 1-bit quantization (AVX-512 gather)";
    py::class_<IVFE8>(m, "IVFE8")
        .def(py::init<size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("construct", &IVFE8::construct,
             py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"))
        .def("search_batch", &IVFE8::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"));
}
