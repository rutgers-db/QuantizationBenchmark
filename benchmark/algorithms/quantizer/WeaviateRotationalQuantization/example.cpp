// Example / benchmark for RotationalQuantizer on fvecs/ivecs datasets
// (SIFT / GIST / DEEP-style ANN benchmarks).
//
// Usage:
//   rq_example <base.fvecs> <query.fvecs> <groundtruth.ivecs> [k=10]
//
// fvecs layout: for each vector: int32 d, then d float32 values.
// ivecs layout: for each row: int32 k, then k int32 ids.

#include "rq.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Reads an fvecs file into a flat [n * d] float array.
// Returns the pair (n, d).
static std::pair<size_t, int> read_fvecs(const std::string& path, std::vector<float>& out) {
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open " + path);
  std::fseek(fp, 0, SEEK_END);
  long fsize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  int32_t d = 0;
  if (std::fread(&d, sizeof(int32_t), 1, fp) != 1) {
    std::fclose(fp);
    throw std::runtime_error("empty file " + path);
  }
  long row_bytes = 4 + static_cast<long>(d) * 4;
  if (fsize % row_bytes != 0) {
    std::fclose(fp);
    throw std::runtime_error("corrupt fvecs: " + path);
  }
  size_t n = static_cast<size_t>(fsize / row_bytes);
  out.resize(n * static_cast<size_t>(d));
  std::fseek(fp, 0, SEEK_SET);
  for (size_t i = 0; i < n; i++) {
    int32_t dim;
    if (std::fread(&dim, sizeof(int32_t), 1, fp) != 1 || dim != d) {
      std::fclose(fp);
      throw std::runtime_error("fvecs dim mismatch in " + path);
    }
    if (std::fread(out.data() + i * d, sizeof(float), d, fp) != static_cast<size_t>(d)) {
      std::fclose(fp);
      throw std::runtime_error("fvecs short read in " + path);
    }
  }
  std::fclose(fp);
  return {n, static_cast<int>(d)};
}

// Reads an ivecs file into a flat [n * w] int32 array. Returns (n, w).
static std::pair<size_t, int> read_ivecs(const std::string& path, std::vector<int32_t>& out) {
  std::FILE* fp = std::fopen(path.c_str(), "rb");
  if (!fp) throw std::runtime_error("cannot open " + path);
  std::fseek(fp, 0, SEEK_END);
  long fsize = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);

  int32_t w = 0;
  if (std::fread(&w, sizeof(int32_t), 1, fp) != 1) {
    std::fclose(fp);
    throw std::runtime_error("empty file " + path);
  }
  long row_bytes = 4 + static_cast<long>(w) * 4;
  size_t n = static_cast<size_t>(fsize / row_bytes);
  out.resize(n * static_cast<size_t>(w));
  std::fseek(fp, 0, SEEK_SET);
  for (size_t i = 0; i < n; i++) {
    int32_t ww;
    if (std::fread(&ww, sizeof(int32_t), 1, fp) != 1 || ww != w) {
      std::fclose(fp);
      throw std::runtime_error("ivecs width mismatch in " + path);
    }
    if (std::fread(out.data() + i * w, sizeof(int32_t), w, fp) != static_cast<size_t>(w)) {
      std::fclose(fp);
      throw std::runtime_error("ivecs short read in " + path);
    }
  }
  std::fclose(fp);
  return {n, static_cast<int>(w)};
}

static void run(int bits, int d, const std::vector<float>& base, size_t nb,
                const std::vector<float>& query, size_t nq,
                const std::vector<int32_t>& gt, int gt_w, size_t k) {
  rq::RotationalQuantizer index(d, bits, rq::Metric::L2);

  auto t0 = std::chrono::high_resolution_clock::now();
  index.add(nb, base.data());
  auto t1 = std::chrono::high_resolution_clock::now();

  std::vector<float>   D(nq * k);
  std::vector<int64_t> I(nq * k);
  index.search(nq, query.data(), k, D.data(), I.data());
  auto t2 = std::chrono::high_resolution_clock::now();

  double add_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double qry_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

  // Recall@k using ground truth top-gt_w (we take the first k truths).
  double recall = 0;
  size_t use = std::min<size_t>(k, static_cast<size_t>(gt_w));
  for (size_t q = 0; q < nq; q++) {
    std::unordered_set<int64_t> tset;
    for (size_t j = 0; j < use; j++) {
      tset.insert(static_cast<int64_t>(gt[q * gt_w + j]));
    }
    size_t hit = 0;
    for (size_t j = 0; j < k; j++) {
      if (tset.count(I[q * k + j])) hit++;
    }
    recall += static_cast<double>(hit) / use;
  }
  recall /= nq;

  double qps = static_cast<double>(nq) / (qry_ms / 1000.0);
  std::printf("bits=%d  code=%4zuB  add=%6.1fms (%.2f us/vec)  "
              "search=%6.1fms  %7.1f QPS  recall@%zu=%.4f\n",
              bits, index.code_size(),
              add_ms, add_ms * 1000.0 / nb,
              qry_ms, qps,
              k, recall);
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr,
        "usage: %s <base.fvecs> <query.fvecs> <groundtruth.ivecs> [k=10]\n",
        argv[0]);
    return 1;
  }
  size_t k = (argc >= 5) ? static_cast<size_t>(std::atoi(argv[4])) : 10;

  std::vector<float>   base, query;
  std::vector<int32_t> gt;

  auto [nb, db] = read_fvecs(argv[1], base);
  auto [nq, dq] = read_fvecs(argv[2], query);
  auto [ng, gw] = read_ivecs(argv[3], gt);
  if (db != dq) {
    std::fprintf(stderr, "dim mismatch: base=%d query=%d\n", db, dq);
    return 1;
  }
  if (ng != nq) {
    std::fprintf(stderr, "row mismatch: queries=%zu groundtruth=%zu\n", nq, ng);
    return 1;
  }
  int d = db;
  std::printf("base=%zu x %d   query=%zu x %d   gt_width=%d   k=%zu\n",
              nb, d, nq, d, gw, k);

  run(8, d, base, nb, query, nq, gt, gw, k);
  run(4, d, base, nb, query, nq, gt, gw, k);
  run(2, d, base, nb, query, nq, gt, gw, k);
  run(1, d, base, nb, query, nq, gt, gw, k);
  return 0;
}
