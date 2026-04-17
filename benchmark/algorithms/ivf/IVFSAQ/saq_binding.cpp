// Pybind11 wrapper for saqlib::IVF from SAQ
// Exposes SAQ/CAQ quantization through a Python-friendly interface

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "index/ivf.hpp"
#include "quantization/config.h"
#include "defines.hpp"
#include "utils/pool.hpp"

#include <omp.h>
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace py = pybind11;
using namespace saqlib;

class PySAQ {
private:
    std::unique_ptr<IVF> ivf_;
    QuantizeConfig cfg_;
    SearcherConfig searcher_cfg_;
    size_t num_data_ = 0;
    size_t num_dim_ = 0;
    size_t num_cen_ = 0;

    // Store references to keep numpy arrays alive
    py::array_t<float> data_ref_;
    py::array_t<float> centroids_ref_;

    // Cluster assignment for each vector (set during build, used for fast MSE)
    std::vector<uint32_t> cluster_ids_;

    // Per-thread query state for graph integration
    struct QueryState {
        Eigen::RowVectorXf query;
        bool valid = false;
    };
    std::vector<QueryState> query_states_;

public:
    PySAQ(float avg_bits, bool enable_segmentation, int caq_adj_rd_lmt,
          bool random_rotation, bool use_fastscan, float caq_adj_eps,
          float vars_bound_m) {
        cfg_.avg_bits = avg_bits;
        cfg_.enable_segmentation = enable_segmentation;
        cfg_.single.quant_type = BaseQuantType::CAQ;
        cfg_.single.random_rotation = random_rotation;
        cfg_.single.use_fastscan = use_fastscan;
        cfg_.single.caq_adj_rd_lmt = caq_adj_rd_lmt;
        cfg_.single.caq_adj_eps = caq_adj_eps;
        searcher_cfg_.searcher_vars_bound_m = vars_bound_m;
        searcher_cfg_.dist_type = DistType::L2Sqr;
    }

    void build(
        py::array_t<float, py::array::c_style | py::array::forcecast> py_data,
        py::array_t<float, py::array::c_style | py::array::forcecast> py_centroids,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> py_cluster_ids,
        py::array_t<float, py::array::c_style | py::array::forcecast> py_variance,
        int num_threads
    ) {
        auto data_buf = py_data.request();
        auto centroids_buf = py_centroids.request();
        auto cluster_ids_buf = py_cluster_ids.request();
        auto variance_buf = py_variance.request();

        if (data_buf.ndim != 2)
            throw std::runtime_error("data must be 2-dimensional");
        if (centroids_buf.ndim != 2)
            throw std::runtime_error("centroids must be 2-dimensional");

        num_data_ = data_buf.shape[0];
        num_dim_ = data_buf.shape[1];
        num_cen_ = centroids_buf.shape[0];

        // Keep references alive
        data_ref_ = py_data;
        centroids_ref_ = py_centroids;

        float* data_ptr = static_cast<float*>(data_buf.ptr);
        float* centroids_ptr = static_cast<float*>(centroids_buf.ptr);
        uint32_t* cluster_ids_ptr = static_cast<uint32_t*>(cluster_ids_buf.ptr);
        float* variance_ptr = static_cast<float*>(variance_buf.ptr);

        // Store cluster assignments for fast MSE computation
        cluster_ids_.assign(cluster_ids_ptr, cluster_ids_ptr + num_data_);

        // Map numpy arrays to Eigen matrices (zero-copy)
        Eigen::Map<FloatRowMat> data_map(data_ptr, num_data_, num_dim_);
        Eigen::Map<FloatRowMat> centroids_map(centroids_ptr, num_cen_, num_dim_);

        // Create variance vector
        FloatVec variance_vec(num_dim_);
        for (size_t i = 0; i < num_dim_; ++i) {
            variance_vec(i) = variance_ptr[i];
        }

        // Create IVF index
        ivf_ = std::make_unique<IVF>(num_data_, num_dim_, num_cen_, cfg_);

        // Set variance for SAQ bit allocation
        ivf_->set_variance(variance_vec);

        // Build the index
        FloatRowMat data_copy = data_map;
        FloatRowMat centroids_copy = centroids_map;
        bool use_1_centroid = (num_cen_ == 1);
        ivf_->construct(data_copy, centroids_copy, cluster_ids_ptr, num_threads, use_1_centroid);

        // Initialize query states for thread safety
        int max_threads = omp_get_max_threads();
        query_states_.resize(max_threads);
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>> search(
        py::array_t<float, py::array::c_style | py::array::forcecast> py_queries,
        uint32_t topk,
        uint32_t nprobe,
        float vars_bound_m,
        int num_threads
    ) {
        if (!ivf_)
            throw std::runtime_error("Index not built. Call build() first.");

        auto queries_buf = py_queries.request();
        uint32_t nq = queries_buf.shape[0];
        uint32_t dim = queries_buf.shape[1];
        float* queries_ptr = static_cast<float*>(queries_buf.ptr);

        py::array_t<int64_t> indices({nq, topk});
        py::array_t<float> distances({nq, topk});
        auto idx_buf = indices.request();
        auto dist_buf = distances.request();
        int64_t* idx_ptr = static_cast<int64_t*>(idx_buf.ptr);
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);

        SearcherConfig scfg;
        scfg.searcher_vars_bound_m = vars_bound_m;
        scfg.dist_type = DistType::L2Sqr;

        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < nq; i++) {
            Eigen::Map<const Eigen::RowVectorXf> query(queries_ptr + i * dim, dim);
            Eigen::RowVectorXf query_copy = query;

            std::vector<PID> results(topk);
            ivf_->search(query_copy, topk, nprobe, scfg, results.data(), nullptr);

            for (uint32_t j = 0; j < topk; j++) {
                idx_ptr[i * topk + j] = static_cast<int64_t>(results[j]);
                dist_ptr[i * topk + j] = 0.0f; // Distances are approximate
            }
        }

        return std::make_pair(indices, distances);
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>> search_with_distances(
        py::array_t<float, py::array::c_style | py::array::forcecast> py_queries,
        uint32_t topk,
        uint32_t nprobe,
        float vars_bound_m,
        int num_threads
    ) {
        if (!ivf_)
            throw std::runtime_error("Index not built. Call build() first.");

        auto queries_buf = py_queries.request();
        uint32_t nq = queries_buf.shape[0];
        uint32_t dim = queries_buf.shape[1];
        float* queries_ptr = static_cast<float*>(queries_buf.ptr);

        py::array_t<int64_t> indices({nq, topk});
        py::array_t<float> distances({nq, topk});
        auto idx_buf = indices.request();
        auto dist_buf = distances.request();
        int64_t* idx_ptr = static_cast<int64_t*>(idx_buf.ptr);
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);

        SearcherConfig scfg;
        scfg.searcher_vars_bound_m = vars_bound_m;
        scfg.dist_type = DistType::L2Sqr;

        #pragma omp parallel for num_threads(num_threads)
        for (uint32_t i = 0; i < nq; i++) {
            Eigen::Map<const Eigen::RowVectorXf> query(queries_ptr + i * dim, dim);
            Eigen::RowVectorXf query_copy = query;

            // Use estimate to get both IDs and distances
            std::vector<std::pair<PID, float>> dist_list;
            ivf_->estimate(query_copy, nprobe, scfg, dist_list, nullptr, nullptr, nullptr);

            // Sort by distance and take topk
            std::partial_sort(dist_list.begin(),
                            dist_list.begin() + std::min((size_t)topk, dist_list.size()),
                            dist_list.end(),
                            [](const auto& a, const auto& b) { return a.second < b.second; });

            for (uint32_t j = 0; j < topk; j++) {
                if (j < dist_list.size()) {
                    idx_ptr[i * topk + j] = static_cast<int64_t>(dist_list[j].first);
                    dist_ptr[i * topk + j] = dist_list[j].second;
                } else {
                    idx_ptr[i * topk + j] = -1;
                    dist_ptr[i * topk + j] = std::numeric_limits<float>::max();
                }
            }
        }

        return std::make_pair(indices, distances);
    }

    // Compute per-element MSE via reconstruction.
    // NOTE: SaqCluEstimator mutates *saq_data and *pcluster (neither is thread-safe),
    // and SaqData has a deleted copy constructor. Runs single-threaded.
    float getMSE(int /* num_threads */) {
        if (!ivf_)
            throw std::runtime_error("Index not built. Call build() first.");
        if (cluster_ids_.empty())
            throw std::runtime_error("Cluster assignments not available. Rebuild the index.");

        auto data_buf = data_ref_.request();
        size_t nd = data_buf.shape[0];
        size_t dim = data_buf.shape[1];
        float* data_ptr = static_cast<float*>(data_buf.ptr);

        const auto& pclusters = ivf_->get_pclusters();
        const SaqData* saq_data = ivf_->get_saq_data();

        SearcherConfig scfg;
        scfg.searcher_vars_bound_m = 1e9f;
        scfg.dist_type = DistType::L2Sqr;

        double total_error = 0.0;

        for (size_t i = 0; i < nd; i++) {
            Eigen::Map<const Eigen::RowVectorXf> query(data_ptr + i * dim, dim);
            Eigen::RowVectorXf query_copy = query;

            uint32_t cid = cluster_ids_[i];
            const auto& pcluster = pclusters[cid];

            SaqCluEstimator<DistType::L2Sqr> estimator(*saq_data, scfg, query_copy);
            estimator.prepare(&pcluster);

            const PID* ids = pcluster.ids();
            for (size_t j = 0; j < pcluster.num_vec_; j++) {
                if (ids[j] == static_cast<PID>(i)) {
                    // containing block, so the LUT's ip_xb_qprime_ state is populated.
                    __m512 cd[2];
                    estimator.compFastDist(j / KFastScanSize, cd);
                    total_error += estimator.compAccurateDist(j);
                    break;
                }
            }
        }

        return static_cast<float>(total_error / (static_cast<double>(nd) * dim));
    }

    void save(const std::string& filename) {
        if (!ivf_)
            throw std::runtime_error("Index not built");
        ivf_->save(filename.c_str());
    }

    void load(const std::string& filename) {
        if (!ivf_) {
            ivf_ = std::make_unique<IVF>(0, 0, 0, cfg_);
        }
        ivf_->load(filename.c_str());
        num_data_ = ivf_->num_data();
        num_dim_ = ivf_->num_dim();
    }

    void set_query(
        py::array_t<float, py::array::c_style | py::array::forcecast> py_query,
        int thread_id
    ) {
        auto query_buf = py_query.request();
        float* query_ptr = static_cast<float*>(query_buf.ptr);
        size_t dim = query_buf.shape[query_buf.ndim - 1];

        if (thread_id >= (int)query_states_.size()) {
            query_states_.resize(thread_id + 1);
        }

        query_states_[thread_id].query = Eigen::Map<const Eigen::RowVectorXf>(query_ptr, dim);
        query_states_[thread_id].valid = true;
    }

    float estimate_distance(uint32_t idx, int thread_id) {
        if (!ivf_)
            throw std::runtime_error("Index not built");
        if (thread_id >= (int)query_states_.size() || !query_states_[thread_id].valid)
            throw std::runtime_error("Query not set. Call set_query() first.");

        SearcherConfig scfg;
        scfg.searcher_vars_bound_m = 1e9f;
        scfg.dist_type = DistType::L2Sqr;

        std::vector<std::pair<PID, float>> dist_list;
        ivf_->estimate(query_states_[thread_id].query, num_cen_, scfg,
                       dist_list, nullptr, nullptr, nullptr);

        for (auto& [pid, dist] : dist_list) {
            if (pid == idx) {
                return dist;
            }
        }
        return std::numeric_limits<float>::max();
    }
};

PYBIND11_MODULE(saq_cpp, m) {
    m.doc() = "SAQ/CAQ C++ bindings (saqlib::IVF wrapper)";

    py::class_<PySAQ>(m, "PySAQ")
        .def(py::init<float, bool, int, bool, bool, float, float>(),
             py::arg("avg_bits"),
             py::arg("enable_segmentation") = true,
             py::arg("caq_adj_rd_lmt") = 6,
             py::arg("random_rotation") = true,
             py::arg("use_fastscan") = true,
             py::arg("caq_adj_eps") = 1e-8f,
             py::arg("vars_bound_m") = 4.0f)
        .def("build", &PySAQ::build,
             py::arg("data"),
             py::arg("centroids"),
             py::arg("cluster_ids"),
             py::arg("variance"),
             py::arg("num_threads") = 16)
        .def("search", &PySAQ::search,
             py::arg("queries"),
             py::arg("topk"),
             py::arg("nprobe") = 1,
             py::arg("vars_bound_m") = 4.0f,
             py::arg("num_threads") = 16)
        .def("search_with_distances", &PySAQ::search_with_distances,
             py::arg("queries"),
             py::arg("topk"),
             py::arg("nprobe") = 1,
             py::arg("vars_bound_m") = 4.0f,
             py::arg("num_threads") = 16)
        .def("getMSE", &PySAQ::getMSE,
             py::arg("num_threads") = 16)
        .def("save", &PySAQ::save,
             py::arg("filename"))
        .def("load", &PySAQ::load,
             py::arg("filename"))
        .def("set_query", &PySAQ::set_query,
             py::arg("query"),
             py::arg("thread_id") = 0)
        .def("estimate_distance", &PySAQ::estimate_distance,
             py::arg("idx"),
             py::arg("thread_id") = 0);
}
