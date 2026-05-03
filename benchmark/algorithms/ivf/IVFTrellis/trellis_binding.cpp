// pybind11 binding for IVF + Trellis-coded scalar quantization (TCQ-style).
//
// Quantizer:
//   * K = 2 bits/step, S = 64 states ⇒ 2 b/d, IDX = 256 (4-ZMM in-register
//     vpermi2b kernel, see in-register-lut.md).
//   * Code per vector = ceil(D*K/8) bytes laid out per-list in SoA-per-step
//     blocks of 64 vectors:
//       block layout: 16 * D bytes
//       step t -> 16 bytes at offset t*16; lane n's 2-bit input lives at
//                 byte (n/4), bit ((n%4)*2)
//   * Estimator factors are the same RaBitQ-style f_add / f_rescale used by
//     IVFE8NoLut: est ‖q-x‖² = g_add + f_add + f_rescale * <q_rot, ĉ>.
//
// Search:
//   * For each of nprobe lists, run the SIMD kernel over full 64-vec blocks
//     and a scalar tail over the remainder, then top-k.

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

#include "trellis_codebook.h"

namespace py = pybind11;
using PID = rabitqlib::PID;
using namespace trellislib;

namespace {

constexpr size_t kTile = 64;                     // 64-way SIMD parallelism
constexpr size_t kStepBytes = (kTile * kK) / 8;  // 16 bytes per step
constexpr size_t kBlockBytesPerStep = kStepBytes;

// list-major code storage: tile-major, padded to multiples of 64.
// The last (partial) tile's unused lanes are zero-initialized; the SIMD
// kernel scans the whole padded list and post-processing skips lane indices
// p >= valid_size, so no separate AoS / scalar-tail path is needed.
struct ListStorage {
    std::vector<PID>     ids;            // size = valid_size; not padded
    std::vector<uint8_t> codes;          // (n_tiles * 16 * D) bytes
    std::vector<float>   f_add;          // padded to multiple of 64
    std::vector<float>   f_rescale;      // padded to multiple of 64
    size_t valid_size = 0;               // actual number of vectors in list
    size_t size() const { return valid_size; }
};

// Pack one vector's bit-stream code into the right (lane, step) slots of the
// SoA-per-step tile block at `tile_base` (which holds 16*D bytes).
inline void scatter_into_tile(const uint8_t* code_aos, size_t D,
                              uint8_t* tile_base, size_t lane) {
    const uint8_t bit = uint8_t(lane * kK);            // 0..126
    const uint8_t byte_off = bit >> 3;                 // 0..15
    const uint8_t shift = bit & 7;                     // 0,2,4,6
    const uint8_t mask = uint8_t((kBr - 1) << shift);
    for (size_t t = 0; t < D; ++t) {
        uint32_t u = bs_get(code_aos, t);
        uint8_t* dst = tile_base + t * 16 + byte_off;
        *dst = uint8_t((*dst & ~mask) | uint8_t((u & (kBr - 1)) << shift));
    }
}

// SIMD kernel for K=2 with shift-register state (QTIP-style).
//   * Output table = 4 ZMM (256B), 2 vpermi2b + blend per step.
//   * State transition is just `idx & kStateMask` — no second table lookup.
// Returns 64 fp32 IPs in `out_ips`.
static inline void ip_simd_64(const float* q, size_t D,
                              const uint8_t* block, float* out_ips) {
    const auto& cb = get_trellis_codebook();
    const __m512i out_lut0 = _mm512_load_si512((const __m512i*)(cb.out_i8 +   0));
    const __m512i out_lut1 = _mm512_load_si512((const __m512i*)(cb.out_i8 +  64));
    const __m512i out_lut2 = _mm512_load_si512((const __m512i*)(cb.out_i8 + 128));
    const __m512i out_lut3 = _mm512_load_si512((const __m512i*)(cb.out_i8 + 192));

    static const uint64_t shift_pat_qw[8] = {
        0x0E0C0A0806040200ULL, 0x1E1C1A1816141210ULL,
        0x2E2C2A2826242220ULL, 0x3E3C3A3836343230ULL,
        0x0E0C0A0806040200ULL, 0x1E1C1A1816141210ULL,
        0x2E2C2A2826242220ULL, 0x3E3C3A3836343230ULL,
    };
    const __m512i shifts        = _mm512_loadu_si512((const __m512i*)shift_pat_qw);
    const __m512i mask03        = _mm512_set1_epi8(0x03);
    const __m512i mask7F        = _mm512_set1_epi8(0x7F);
    const __m512i state_mask_v  = _mm512_set1_epi8(int8_t(kStateMask));   // 0x3F
    const __m512  v_scale       = _mm512_set1_ps(kScale);
    const __m512i perm_dup_qw   = _mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);

    __m512i state_v = _mm512_setzero_si512();
    __m512  acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512  acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();

    for (size_t t = 0; t < D; ++t) {
        __m128i raw16 = _mm_loadu_si128((const __m128i*)(block + t * 16));
        __m512i raw64 = _mm512_castsi128_si512(raw16);
        __m512i data  = _mm512_permutexvar_epi64(perm_dup_qw, raw64);
        __m512i fields = _mm512_multishift_epi64_epi8(shifts, data);
        __m512i input_v = _mm512_and_si512(fields, mask03);

        // idx = (state << K) | input  (full 8-bit key, ranges 0..255)
        __m512i idx_v      = _mm512_or_si512(_mm512_slli_epi32(state_v, kK), input_v);
        __m512i idx_masked = _mm512_and_si512(idx_v, mask7F);
        __mmask64 mask_hi  = _mm512_movepi8_mask(idx_v);

        // Output table lookup (still needs the double vpermi2b for 256-byte LUT).
        __m512i out_lo = _mm512_permutex2var_epi8(out_lut0, idx_v,      out_lut1);
        __m512i out_hi = _mm512_permutex2var_epi8(out_lut2, idx_masked, out_lut3);
        __m512i out_v  = _mm512_mask_blend_epi8(mask_hi, out_lo, out_hi);

        __m512  q_t = _mm512_set1_ps(q[t]);
        __m512i o0 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(out_v, 0));
        __m512i o1 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(out_v, 1));
        __m512i o2 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(out_v, 2));
        __m512i o3 = _mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(out_v, 3));
        __m512  f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(o0), v_scale);
        __m512  f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(o1), v_scale);
        __m512  f2 = _mm512_mul_ps(_mm512_cvtepi32_ps(o2), v_scale);
        __m512  f3 = _mm512_mul_ps(_mm512_cvtepi32_ps(o3), v_scale);
        acc0 = _mm512_fmadd_ps(q_t, f0, acc0);
        acc1 = _mm512_fmadd_ps(q_t, f1, acc1);
        acc2 = _mm512_fmadd_ps(q_t, f2, acc2);
        acc3 = _mm512_fmadd_ps(q_t, f3, acc3);

        // QTIP-style shift-register: state = idx & state_mask. One AND.
        state_v = _mm512_and_si512(idx_v, state_mask_v);
    }
    _mm512_storeu_ps(out_ips +  0, acc0);
    _mm512_storeu_ps(out_ips + 16, acc1);
    _mm512_storeu_ps(out_ips + 32, acc2);
    _mm512_storeu_ps(out_ips + 48, acc3);
}

}  // anonymous

class IVFTrellis {
public:
    IVFTrellis(size_t n, size_t dim, size_t nlist, int nthread,
               const std::string& metric, const std::string& rotator = "fht")
        : n_(n), dim_(dim), nlist_(nlist), nthread_(nthread) {
        (void)metric;

        // Pad to a multiple of 64 for FHT-Kac happiness; trellis itself only
        // needs multiple-of-1.  E8NoLut also pads to 64.
        padded_dim_ = rabitqlib::round_up_to_multiple(dim_, 64);
        code_bytes_ = (padded_dim_ * kK + 7) / 8;

        rabitqlib::RotatorType rtype =
            (rotator == "matrix") ? rabitqlib::RotatorType::MatrixRotator
                                  : rabitqlib::RotatorType::FhtKacRotator;
        std::srand(static_cast<unsigned>(dim_ + padded_dim_));
        rotator_.reset(rabitqlib::choose_rotator<float>(dim_, rtype, padded_dim_));

        rotated_centroids_.assign(nlist_ * padded_dim_, 0.0f);
        centroids_.assign(nlist_ * dim_, 0.0f);
        lists_.resize(nlist_);

        // Touch the singleton so the codebook is built before any thread
        // races on it inside an OpenMP region.
        (void)get_trellis_codebook();
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

        // Per-list sizing & padding to multiples of 64.
        std::vector<size_t> counts(nlist_, 0);
        for (size_t i = 0; i < nb; ++i) {
            if (idptr[i] >= nlist_) throw std::runtime_error("cluster id out of range");
            ++counts[idptr[i]];
        }
        for (size_t l = 0; l < nlist_; ++l) {
            size_t sz = counts[l];
            size_t padded_sz = ((sz + kTile - 1) / kTile) * kTile;
            size_t n_tiles = padded_sz / kTile;
            lists_[l].ids.resize(sz);
            lists_[l].valid_size = sz;
            lists_[l].codes.assign(n_tiles * 16 * padded_dim_, 0);
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

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim_);
            std::vector<float> residual(padded_dim_);
            std::vector<float> reconstructed(padded_dim_);
            std::vector<uint8_t> code_local(code_bytes_);
            ViterbiScratch scratch;

            #pragma omp for schedule(static)
            for (int64_t i = 0; i < (int64_t)nb; ++i) {
                uint32_t cid = idptr[i];
                size_t p = pos[i];

                rotator_->rotate(dptr + i * dim_, rotated.data());
                const float* cent_rot = rotated_centroids_.data() + cid * padded_dim_;
                for (size_t k = 0; k < padded_dim_; ++k) {
                    residual[k] = rotated[k] - cent_rot[k];
                }

                std::fill(code_local.begin(), code_local.end(), uint8_t(0));
                viterbi_encode(residual.data(), padded_dim_,
                               code_local.data(), reconstructed.data(), scratch);

                // Scatter into SoA-per-step tile for SIMD scan.
                size_t tile = p / kTile;
                size_t lane = p % kTile;
                uint8_t* tile_base =
                    lists_[cid].codes.data() + tile * 16 * padded_dim_;
                scatter_into_tile(code_local.data(), padded_dim_, tile_base, lane);

                // RaBitQ-style factors (unbiased L2 estimator):
                //   est ‖q-x‖² = g_add + l2_sqr
                //                + 2*l2_sqr * <c, ĉ> / <r, ĉ>
                //                - 2*l2_sqr / <r, ĉ> * <q, ĉ>
                //  where l2_sqr = ‖r‖², r = R(x) - R(c), ĉ = decoded code.
                float l2_sqr     = rabitqlib::l2norm_sqr<float>(residual.data(),
                                                                padded_dim_);
                float ip_resi_qr = rabitqlib::dot_product<float>(residual.data(),
                                                                 reconstructed.data(),
                                                                 padded_dim_);
                float ip_cent_qr = rabitqlib::dot_product<float>(cent_rot,
                                                                 reconstructed.data(),
                                                                 padded_dim_);
                if (ip_resi_qr == 0.0f) {
                    ip_resi_qr = std::numeric_limits<float>::infinity();
                }
                lists_[cid].f_add    [p] = l2_sqr + (2.0f * l2_sqr * ip_cent_qr / ip_resi_qr);
                lists_[cid].f_rescale[p] = -2.0f * l2_sqr / ip_resi_qr;
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
        py::array_t<float>   D({static_cast<py::ssize_t>(nq),
                                static_cast<py::ssize_t>(k)});
        int64_t* Iptr = static_cast<int64_t*>(I.request().ptr);
        float*   Dptr = static_cast<float*>(D.request().ptr);

        omp_set_num_threads(nthread_);

        const size_t padded_dim = padded_dim_;

        #pragma omp parallel
        {
            std::vector<float> rotated(padded_dim);
            std::vector<std::pair<float, size_t>> probe_list(nlist_);
            alignas(64) float ip_buf[kTile];

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

                std::priority_queue<std::pair<float, PID>> topk;

                for (size_t pp = 0; pp < np; ++pp) {
                    float g_add = probe_list[pp].first;
                    size_t l = probe_list[pp].second;
                    const auto& L = lists_[l];
                    size_t sz = L.size();
                    if (sz == 0) continue;

                    size_t n_tiles = (sz + kTile - 1) / kTile;
                    const uint8_t* L_codes = L.codes.data();
                    const float*   L_fadd  = L.f_add.data();
                    const float*   L_fres  = L.f_rescale.data();
                    const PID*     L_ids   = L.ids.data();
                    const size_t   tile_stride = 16 * padded_dim;

                    for (size_t t = 0; t < n_tiles; ++t) {
                        const uint8_t* tile_base = L_codes + t * tile_stride;
                        ip_simd_64(rotated.data(), padded_dim, tile_base, ip_buf);
                        size_t base_p = t * kTile;
                        size_t lanes = std::min(kTile, sz - base_p);
                        for (size_t v = 0; v < lanes; ++v) {
                            float ip = ip_buf[v];
                            size_t p = base_p + v;
                            float est = g_add + L_fadd[p] + L_fres[p] * ip;
                            PID id = L_ids[p];
                            if (topk.size() < k) {
                                topk.emplace(est, id);
                            } else if (est < topk.top().first) {
                                topk.pop();
                                topk.emplace(est, id);
                            }
                        }
                    }
                }

                for (size_t j = 0; j < k; ++j) {
                    Iptr[qi * k + j] = -1;
                    Dptr[qi * k + j] = std::numeric_limits<float>::infinity();
                }
                std::vector<std::pair<float, PID>> ordered;
                ordered.reserve(topk.size());
                while (!topk.empty()) { ordered.push_back(topk.top()); topk.pop(); }
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
    size_t code_bytes_;
    int    nthread_;

    std::unique_ptr<rabitqlib::Rotator<float>> rotator_;
    std::vector<float> centroids_;
    std::vector<float> rotated_centroids_;
    std::vector<ListStorage> lists_;
};

PYBIND11_MODULE(trellis_cpp, m) {
    m.doc() = "IVF + Trellis-coded scalar quantization (TCQ-style, K=2/S=64)";
    py::class_<IVFTrellis>(m, "IVFTrellis")
        .def(py::init<size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("construct", &IVFTrellis::construct,
             py::arg("data"), py::arg("centroids"), py::arg("cluster_ids"))
        .def("search_batch", &IVFTrellis::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"));
}
