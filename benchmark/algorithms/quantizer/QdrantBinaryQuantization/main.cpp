// Demo / benchmark for BinaryQuantizer using standard fvecs / ivecs.
//
// Usage:
//   ./bq_demo <base.fvecs> <query.fvecs> <gt.ivecs>
//             [--k=10] [--encoding=1|2]
//             [--query-encoding=same|scalar4|scalar8]
//             [--metric=L2|IP|Hamming]
//
// Prints timings, recall@1 and recall@k against the supplied ivecs GT,
// and the SIMD kernel that was picked at runtime.

#include "BinaryQuantizer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace {

template <typename T>
std::vector<T> read_xvecs(const std::string& path, size_t& out_n, size_t& out_d) {
    static_assert(sizeof(T) == 4, "fvecs/ivecs elements are 4 bytes");
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);

    f.seekg(0, std::ios::end);
    std::streamoff bytes = f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes <= 0) throw std::runtime_error("empty file: " + path);

    int32_t dim0 = 0;
    f.read(reinterpret_cast<char*>(&dim0), sizeof(int32_t));
    if (!f || dim0 <= 0) throw std::runtime_error("bad header in: " + path);
    f.seekg(0, std::ios::beg);

    const size_t row_bytes = sizeof(int32_t) + static_cast<size_t>(dim0) * sizeof(T);
    if (static_cast<size_t>(bytes) % row_bytes != 0) {
        throw std::runtime_error("file size not a multiple of record size: " + path);
    }
    const size_t n = static_cast<size_t>(bytes) / row_bytes;

    std::vector<T> data(n * static_cast<size_t>(dim0));
    for (size_t i = 0; i < n; ++i) {
        int32_t dim_i = 0;
        f.read(reinterpret_cast<char*>(&dim_i), sizeof(int32_t));
        if (dim_i != dim0) throw std::runtime_error("non-uniform dim in: " + path);
        f.read(reinterpret_cast<char*>(data.data() + i * static_cast<size_t>(dim0)),
               static_cast<std::streamsize>(dim0) * sizeof(T));
        if (!f) throw std::runtime_error("read failure in: " + path);
    }
    out_n = n;
    out_d = static_cast<size_t>(dim0);
    return data;
}

const char* kernel_name(bq::Kernel k) {
    switch (k) {
        case bq::Kernel::AVX512VPopcnt: return "AVX-512 VPOPCNTDQ";
        case bq::Kernel::AVX2:          return "AVX-2 (Mula shuffle LUT)";
        case bq::Kernel::Scalar:        return "scalar __builtin_popcountll";
    }
    return "?";
}

// Very small flag parser for --key=value on argv.
std::string_view flag(int argc, char** argv, std::string_view key,
                      std::string_view def) {
    const std::string prefix = std::string("--") + std::string(key) + "=";
    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a.rfind(prefix, 0) == 0) return a.substr(prefix.size());
    }
    return def;
}

bq::Encoding parse_encoding(std::string_view s) {
    if (s == "1") return bq::Encoding::OneBit;
    if (s == "2") return bq::Encoding::TwoBits;
    throw std::runtime_error("--encoding must be 1 or 2");
}

bq::QueryEncoding parse_query_encoding(std::string_view s) {
    if (s == "same")    return bq::QueryEncoding::SameAsStorage;
    if (s == "scalar4") return bq::QueryEncoding::Scalar4Bits;
    if (s == "scalar8") return bq::QueryEncoding::Scalar8Bits;
    throw std::runtime_error("--query-encoding must be same | scalar4 | scalar8");
}

bq::Metric parse_metric(std::string_view s) {
    if (s == "L2" || s == "l2") return bq::Metric::L2;
    if (s == "IP" || s == "ip") return bq::Metric::IP;
    if (s == "Hamming" || s == "hamming" || s == "H" || s == "h") return bq::Metric::Hamming;
    throw std::runtime_error("--metric must be L2 | IP | Hamming");
}

const char* encoding_str(bq::Encoding e) {
    return e == bq::Encoding::OneBit ? "OneBit" : "TwoBits";
}
const char* query_encoding_str(bq::QueryEncoding e) {
    switch (e) {
        case bq::QueryEncoding::SameAsStorage: return "SameAsStorage";
        case bq::QueryEncoding::Scalar4Bits:   return "Scalar4Bits";
        case bq::QueryEncoding::Scalar8Bits:   return "Scalar8Bits";
    }
    return "?";
}
const char* metric_str(bq::Metric m) {
    switch (m) {
        case bq::Metric::L2:      return "L2";
        case bq::Metric::IP:      return "IP";
        case bq::Metric::Hamming: return "Hamming";
    }
    return "?";
}

} // anonymous

int main(int argc, char** argv) {
    // First three positional args must be base / query / gt paths.
    if (argc < 4 || std::string_view(argv[1]).rfind("--", 0) == 0) {
        std::fprintf(stderr,
            "usage: %s <base.fvecs> <query.fvecs> <gt.ivecs>\n"
            "          [--k=10] [--encoding=1|2] [--query-encoding=same|scalar4|scalar8]\n"
            "          [--metric=L2|IP|Hamming]\n",
            argv[0]);
        return 1;
    }
    const std::string base_path  = argv[1];
    const std::string query_path = argv[2];
    const std::string gt_path    = argv[3];

    const size_t k = static_cast<size_t>(std::atoi(std::string(flag(argc, argv, "k", "10")).c_str()));
    const bq::Encoding       enc  = parse_encoding       (flag(argc, argv, "encoding",       "1"));
    const bq::QueryEncoding  qenc = parse_query_encoding (flag(argc, argv, "query-encoding", "same"));
    const bq::Metric         met  = parse_metric         (flag(argc, argv, "metric",         "L2"));

    // --- load ---
    size_t n_base = 0, d_base = 0;
    std::vector<float> base = read_xvecs<float>(base_path, n_base, d_base);

    size_t n_query = 0, d_query = 0;
    std::vector<float> query = read_xvecs<float>(query_path, n_query, d_query);
    if (d_query != d_base) {
        std::fprintf(stderr, "dim mismatch: base d=%zu, query d=%zu\n", d_base, d_query);
        return 2;
    }

    size_t n_gt = 0, d_gt = 0;
    std::vector<int32_t> gt = read_xvecs<int32_t>(gt_path, n_gt, d_gt);
    if (n_gt != n_query) {
        std::fprintf(stderr, "gt rows (%zu) != query rows (%zu)\n", n_gt, n_query);
        return 3;
    }
    if (d_gt < k) {
        std::fprintf(stderr, "gt has %zu cols but k=%zu\n", d_gt, k);
        return 4;
    }

    std::printf("BinaryQuantizer benchmark\n");
    std::printf("  base:   %s  n=%zu d=%zu\n", base_path.c_str(),  n_base,  d_base);
    std::printf("  query:  %s  n=%zu d=%zu\n", query_path.c_str(), n_query, d_query);
    std::printf("  gt:     %s  n=%zu d=%zu\n", gt_path.c_str(),    n_gt,    d_gt);
    std::printf("  k=%zu  encoding=%s  query-encoding=%s  metric=%s\n",
                k, encoding_str(enc), query_encoding_str(qenc), metric_str(met));
    std::printf("  kernel=%s\n", kernel_name(bq::active_kernel()));

    // --- build ---
    bq::BinaryQuantizer index(d_base, enc, qenc, met);

    auto t0 = std::chrono::steady_clock::now();
    index.train(n_base, base.data());
    auto t1 = std::chrono::steady_clock::now();
    index.add(n_base, base.data());
    auto t2 = std::chrono::steady_clock::now();

    // --- search ---
    std::vector<float>   D(n_query * k);
    std::vector<int64_t> I(n_query * k);
    index.search(n_query, query.data(), k, D.data(), I.data());
    auto t3 = std::chrono::steady_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    std::printf("train:  %.2f ms\n", ms(t0, t1));
    std::printf("add:    %.2f ms  (%.2f MB codes, %zu B/code)\n",
                ms(t1, t2),
                static_cast<double>(index.ntotal * index.code_size) / (1024.0 * 1024.0),
                index.code_size);
    const double search_ms = ms(t2, t3);
    std::printf("search: %.2f ms total, %.2f us/query, %.0f QPS\n",
                search_ms, search_ms * 1000.0 / n_query, n_query * 1000.0 / search_ms);

    // --- recall vs ivecs ground truth ---
    size_t top1_hits = 0, topk_hits = 0;
    for (size_t qi = 0; qi < n_query; ++qi) {
        if (I[qi * k + 0] == static_cast<int64_t>(gt[qi * d_gt + 0])) ++top1_hits;
        std::unordered_set<int64_t> gtset;
        gtset.reserve(k * 2);
        for (size_t j = 0; j < k; ++j) {
            gtset.insert(static_cast<int64_t>(gt[qi * d_gt + j]));
        }
        for (size_t j = 0; j < k; ++j) {
            if (gtset.count(I[qi * k + j])) ++topk_hits;
        }
    }
    std::printf("recall@1:  %.4f\n", static_cast<double>(top1_hits) / n_query);
    std::printf("recall@%zu: %.4f\n", k, static_cast<double>(topk_hits) / (n_query * k));
    return 0;
}
