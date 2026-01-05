// Pybind11 wrapper for IVFRN from ivf_rabitq.h
// We use C=1 (single centroid) to effectively do brute-force search with RaBitQ

// #define EIGEN_DONT_PARALLELIZE
// #define FAST_SCAN  // Use fast_scan instead of scan
// #define USE_AVX2

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Include RaBitQ headers
#include "ivf_rabitq.h"
#include <omp.h>
#include <mutex>

namespace py = pybind11;

// Python wrapper for IVFRN with fixed dimensions
// We'll instantiate for common dimensions
template<uint32_t D, uint32_t B>
class PyIVFRN {
private:
    IVFRN<D, B>* index;
    Matrix<float>* X;
    Matrix<float>* centroids;
    Matrix<float>* dist_to_centroid;
    Matrix<float>* x0;
    Matrix<uint32_t>* cluster_id;
    Matrix<uint64_t>* binary;
    static Space<D,B> space;
    uint32_t C, N;
    uint8_t* query_bytes;
    uint64_t* quant_query;
    float *query_vl;
    float *query_vr;
    float *query_width;
    float *query_sqry;
    uint32_t *query_sumq;
    uint32_t *id2pos;
    uint32_t *id2centroid;
public:
    PyIVFRN() : index(nullptr), X(nullptr), centroids(nullptr),
                 dist_to_centroid(nullptr), x0(nullptr),
                 cluster_id(nullptr), binary(nullptr), query_bytes(nullptr),
                 quant_query(nullptr), query_vl(nullptr), query_vr(nullptr), query_width(nullptr),
                 query_sqry(nullptr), query_sumq(nullptr), id2pos(nullptr), id2centroid(nullptr) {}

    ~PyIVFRN() {
        if (X){
            X->data = nullptr;
            delete X;
        }
        if (centroids){
            centroids->data = nullptr;
            delete centroids;
        }
        if (dist_to_centroid){
            dist_to_centroid->data = nullptr;
            delete dist_to_centroid;
        }
        if (x0){
            x0->data = nullptr;
            delete x0;
        }
        if (cluster_id){
            cluster_id->data = nullptr;
            delete cluster_id;
        }
        if (binary){
            binary->data = nullptr;
            delete binary;
        }
        if (index) delete index;
        if (query_bytes) free(query_bytes);
        if (quant_query) free(quant_query);
        if (query_vl) delete[] query_vl;
        if (query_vr) delete[] query_vr;
        if (query_width) delete[] query_width;
        if (query_sqry) delete[] query_sqry;
        if (query_sumq) delete[] query_sumq;
        if (id2pos) delete[] id2pos;
        if (id2centroid) delete[] id2centroid;
    }

    // Build index from Python arrays
    void build(
        py::array_t<float, py::array::c_style | py::array::forcecast> py_data,
        py::array_t<float, py::array::c_style | py::array::forcecast> py_centroids,
        py::array_t<float, py::array::c_style | py::array::forcecast> py_dist_to_centroid,
        py::array_t<float, py::array::c_style | py::array::forcecast> py_x0,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> py_cluster_id,
        py::array_t<uint64_t, py::array::c_style | py::array::forcecast> py_binary
    ) {
        // Request buffers - pybind11 will ensure arrays are C-contiguous
        auto data_buf = py_data.request();
        auto centroids_buf = py_centroids.request();
        auto dist_buf = py_dist_to_centroid.request();
        auto x0_buf = py_x0.request();
        auto cluster_buf = py_cluster_id.request();
        auto binary_buf = py_binary.request();

        // Verify buffer properties
        if (data_buf.ndim != 2) {
            throw std::runtime_error("data must be 2-dimensional");
        }
        if (!data_buf.strides.empty() && data_buf.strides[1] != sizeof(float)) {
            throw std::runtime_error("data must be C-contiguous");
        }

        // Create Matrix wrappers
        X = new Matrix<float>();
        X->n = data_buf.shape[0];
        X->d = data_buf.shape[1];
        X->data = static_cast<float*>(data_buf.ptr);

        centroids = new Matrix<float>();
        centroids->n = centroids_buf.shape[0];
        centroids->d = centroids_buf.shape[1];
        centroids->data = static_cast<float*>(centroids_buf.ptr);

        dist_to_centroid = new Matrix<float>();
        dist_to_centroid->n = dist_buf.shape[0];
        dist_to_centroid->d = 1;
        dist_to_centroid->data = static_cast<float*>(dist_buf.ptr);

        x0 = new Matrix<float>();
        x0->n = x0_buf.shape[0];
        x0->d = 1;
        x0->data = static_cast<float*>(x0_buf.ptr);

        cluster_id = new Matrix<uint32_t>();
        cluster_id->n = cluster_buf.shape[0];
        cluster_id->d = 1;
        cluster_id->data = static_cast<uint32_t*>(cluster_buf.ptr);

        binary = new Matrix<uint64_t>();
        binary->n = binary_buf.shape[0];
        binary->d = binary_buf.shape[1];
        binary->data = static_cast<uint64_t*>(binary_buf.ptr);

        if (index) delete index;
        index = new IVFRN<D, B>(*X, *centroids, *dist_to_centroid, *x0, *cluster_id, *binary);

        C = index->C;
        N = index->N;
        query_bytes = static_cast<uint8_t*>(std::aligned_alloc(64, B * C));
        quant_query = static_cast<uint64_t*>(std::aligned_alloc(32, (B_QUERY * B / 64) * C * sizeof(uint64_t)));
        query_vl = new float[C];
        query_vr = new float[C];
        query_width = new float[C];
        query_sumq = new uint32_t[C];
        query_sqry = new float[C];
        id2pos = new uint32_t[N];
        id2centroid = new uint32_t[N];
        for (int i = 0; i < C; i++){
            uint32_t end = (i == C - 1) ? N : (index->start[i+1]);
            for (int j = index->start[i]; j < end; j++){
                uint32_t id = index->id[j];
                id2pos[id] = j;
                id2centroid[id] = i;
            }
        }
    }

    // Search
    std::pair<py::array_t<int64_t>, py::array_t<float>> search(
        py::array_t<float> py_queries,
        py::array_t<float> py_rd_queries,
        uint32_t k,
        uint32_t nprobe
    ) {

        if (!index) {
            throw std::runtime_error("Index not built");
        }


        auto queries_buf = py_queries.request();
        auto rd_queries_buf = py_rd_queries.request();

        uint32_t nq = queries_buf.shape[0];

        float* queries = static_cast<float*>(queries_buf.ptr);
        float* rd_queries = static_cast<float*>(rd_queries_buf.ptr);

        // Allocate result arrays
        py::array_t<int64_t> indices({nq, k});
        py::array_t<float> distances({nq, k});

        auto idx_buf = indices.request();
        auto dist_buf = distances.request();
        int64_t* idx_ptr = static_cast<int64_t*>(idx_buf.ptr);
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);

        int thread_num = omp_get_max_threads();
        // Search each query
        #pragma omp parallel for num_threads(thread_num)
        for (uint32_t i = 0; i < nq; i++) {

            ResultHeap result = index->search(
                queries + i * D,
                rd_queries + i * B,
                k,
                nprobe
            );

            // Extract results (they come in reverse order from heap)
            std::vector<std::pair<float, uint32_t>> res_vec;
            while (!result.empty()) {
                res_vec.push_back(result.top());
                result.pop();
            }
            std::reverse(res_vec.begin(), res_vec.end());

            // Fill output arrays
            for (uint32_t j = 0; j < k; j++) {
                if (j < res_vec.size()) {
                    idx_ptr[i * k + j] = res_vec[j].second;
                    dist_ptr[i * k + j] = res_vec[j].first;
                } else {
                    idx_ptr[i * k + j] = -1;
                    dist_ptr[i * k + j] = std::numeric_limits<float>::max();
                }
            }
        }

        return std::make_pair(indices, distances);
    }

    void set_query(
        py::array_t<float> py_query,
        py::array_t<float> py_rd_query
    ){
        auto queries_buf = py_query.request();
        auto rd_queries_buf = py_rd_query.request();

        float* query = static_cast<float*>(queries_buf.ptr);
        float* rd_query = static_cast<float*>(rd_queries_buf.ptr);
        
        for (int i = 0; i < C; i++){
            uint8_t* byte_query = query_bytes + i * B;
            query_sqry[i] = sqr_dist<B>(rd_query, index->centroid + i * B);
            float vl, vr;
            index->space.range(rd_query, index->centroid + i * B, vl, vr);
            query_vl[i] = vl;
            query_vr[i] = vr;
            query_width[i] = (vr - vl) / ((1 << B_QUERY) - 1);
            uint32_t sum_q = 0;
            index->space.quantize(byte_query, rd_query, index->centroid + i * B, index->u, vl, query_width[i], sum_q);
            query_sumq[i] = sum_q;
            uint64_t* quant = quant_query + (B_QUERY * B / 64) * i;
            memset(quant, 0, (B_QUERY * B / 64) * sizeof(uint64_t));
            space.transpose_bin(byte_query, quant);
            // std::cout << i << ", " << static_cast<void*>(byte_query) << ", " << static_cast<void*>(quant) << std::endl;
        }
    }

    float estimate_distance_after_set_query(int id){
        uint32_t c = id2centroid[id];
        uint32_t pos = id2pos[id];
        float vl = query_vl[c];
        float vr = query_vr[c];
        float width = query_width[c];
        uint32_t sum_q = query_sumq[c];
        float sqr_y = query_sqry[c];
        uint8_t* byte_query = query_bytes + c * B;
        uint64_t* quant = quant_query + (B_QUERY * B / 64) * c;
        auto ptr_fac = index->fac + pos;
        uint64_t* packed_code = index->binary_code + 1ull * pos * (B / 64);
        float tmp = space.ip_byte_bin(quant, packed_code);
        float dist = ptr_fac->sqr_x + sqr_y + ptr_fac->factor_ppc * vl + (tmp * 2 - sum_q) * (ptr_fac->factor_ip) * width;
        // std::cout << id << ", " << c << ", " << pos << ", " << vl << ", " << vr << ", " << width 
        //     << ", " << sum_q << ", " << sqr_y << ", " << ptr_fac->sqr_x << ", " << ptr_fac->factor_ppc
        //     << tmp << ", " << ptr_fac->factor_ip << ", " << index->id[pos] << std::endl;
        return dist;
    }
    
    std::pair<py::array_t<int64_t>, py::array_t<float>> search_clusters(
        py::array_t<float> py_queries,
        py::array_t<float> py_rd_queries,
        py::array_t<uint32_t> py_assignments,
        uint32_t k
    ) {

        if (!index) {
            throw std::runtime_error("Index not built");
        }


        auto queries_buf = py_queries.request();
        auto rd_queries_buf = py_rd_queries.request();
        auto assignments_buf = py_assignments.request();

        uint32_t nq = queries_buf.shape[0];
        uint32_t nprobe = assignments_buf.shape[1];

        float* queries = static_cast<float*>(queries_buf.ptr);
        float* rd_queries = static_cast<float*>(rd_queries_buf.ptr);
        uint32_t* assignments = static_cast<uint32_t*>(assignments_buf.ptr);

        // Allocate result arrays
        py::array_t<int64_t> indices({nq, k});
        py::array_t<float> distances({nq, k});

        auto idx_buf = indices.request();
        auto dist_buf = distances.request();
        int64_t* idx_ptr = static_cast<int64_t*>(idx_buf.ptr);
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);


        int thread_num = omp_get_max_threads();
        // Search each query
        #pragma omp parallel for num_threads(thread_num)
        for (uint32_t i = 0; i < nq; i++) {

            ResultHeap KNNs;
            float *query = queries + i * D;
            float *rd_query = rd_queries + i * B;
            uint8_t PORTABLE_ALIGN64 byte_query[B];
            float distK = std::numeric_limits<float>::max();
            for (uint32_t j = 0; j < nprobe; j++){
                int cluster_id = assignments[i * nprobe + j];
                float sqr_y = sqr_dist<B>(rd_query, index->centroid + cluster_id * B);
                float vl, vr;
                index->space.range(rd_query, index->centroid + cluster_id * B, vl, vr);
                float width = (vr - vl) / ((1 << B_QUERY) - 1);
                uint32_t sum_q = 0;
                index->space.quantize(byte_query, rd_query, index->centroid + cluster_id * B, index->u, vl, width, sum_q);
                uint8_t PORTABLE_ALIGN32 LUT[B/4*16];
                pack_LUT<B>(byte_query, LUT);
                index->fast_scan(KNNs, distK, k, LUT, index->packed_code + index->packed_start[cluster_id],
                                index->len[cluster_id], index->fac + index->start[cluster_id],
                                sqr_y, vl, width, sum_q, query, index->data + index->start[cluster_id] * D,
                                index->id + index->start[cluster_id]);
            }

            // Extract results (they come in reverse order from heap)
            std::vector<std::pair<float, uint32_t>> res_vec;
            while (!KNNs.empty()) {
                res_vec.push_back(KNNs.top());
                KNNs.pop();
            }
            std::reverse(res_vec.begin(), res_vec.end());

            // Fill output arrays
            for (uint32_t j = 0; j < k; j++) {
                if (j < res_vec.size()) {
                    idx_ptr[i * k + j] = res_vec[j].second;
                    dist_ptr[i * k + j] = res_vec[j].first;
                } else {
                    idx_ptr[i * k + j] = -1;
                    dist_ptr[i * k + j] = std::numeric_limits<float>::max();
                }
            }
        }

        return std::make_pair(indices, distances);
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>> search_and_rerank_clusters(
        py::array_t<float> py_queries,
        py::array_t<float> py_rd_queries,
        py::array_t<uint32_t> py_assignments,
        uint32_t k,
        uint32_t nrerank
    ) {
        if (!index) {
            throw std::runtime_error("Index not built");
        }


        auto queries_buf = py_queries.request();
        auto rd_queries_buf = py_rd_queries.request();
        auto assignments_buf = py_assignments.request();

        uint32_t nq = queries_buf.shape[0];
        uint32_t nprobe = assignments_buf.shape[1];

        float* queries = static_cast<float*>(queries_buf.ptr);
        float* rd_queries = static_cast<float*>(rd_queries_buf.ptr);
        uint32_t* assignments = static_cast<uint32_t*>(assignments_buf.ptr);

        // Allocate result arrays
        py::array_t<int64_t> indices({nq, k});
        py::array_t<float> distances({nq, k});

        auto idx_buf = indices.request();
        auto dist_buf = distances.request();
        int64_t* idx_ptr = static_cast<int64_t*>(idx_buf.ptr);
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);

        uint32_t total_rerank = 0;
        int thread_num = omp_get_max_threads();
        // Search each query
        #pragma omp parallel for num_threads(thread_num) reduction(+:total_rerank)
        for (uint32_t i = 0; i < nq; i++) {

            ResultHeap KNNs;
            float *query = queries + i * D;
            float *rd_query = rd_queries + i * B;
            uint8_t PORTABLE_ALIGN64 byte_query[B];
            float distK = std::numeric_limits<float>::max();
            for (uint32_t j = 0; j < nprobe; j++){
                int cluster_id = assignments[i * nprobe + j];
                float sqr_y = sqr_dist<B>(rd_query, index->centroid + cluster_id * B);
                float vl, vr;
                index->space.range(rd_query, index->centroid + cluster_id * B, vl, vr);
                float width = (vr - vl) / ((1 << B_QUERY) - 1);
                uint32_t sum_q = 0;
                index->space.quantize(byte_query, rd_query, index->centroid + cluster_id * B, index->u, vl, width, sum_q);
                uint8_t PORTABLE_ALIGN32 LUT[B/4*16];
                pack_LUT<B>(byte_query, LUT);
                uint32_t rerank_cnt = index->fast_scan_with_rerank(KNNs, distK, nrerank, LUT, index->packed_code + index->packed_start[cluster_id],
                                index->len[cluster_id], index->fac + index->start[cluster_id],
                                sqr_y, vl, width, sum_q, query, index->data + index->start[cluster_id] * D,
                                index->id + index->start[cluster_id]);
                total_rerank += rerank_cnt;
            }

            // Extract results (they come in reverse order from heap)
            std::vector<std::pair<float, uint32_t>> res_vec;
            while (!KNNs.empty()) {
                res_vec.push_back(KNNs.top());
                KNNs.pop();
            }
            std::reverse(res_vec.begin(), res_vec.end());

            // Fill output arrays
            for (uint32_t j = 0; j < k; j++) {
                if (j < res_vec.size()) {
                    idx_ptr[i * k + j] = res_vec[j].second;
                    dist_ptr[i * k + j] = res_vec[j].first;
                } else {
                    idx_ptr[i * k + j] = -1;
                    dist_ptr[i * k + j] = std::numeric_limits<float>::max();
                }
            }
        }

        std::cout << "number of rerank per vector: " << (float)total_rerank / (float) nq << std::endl;
        return std::make_pair(indices, distances);
    }

    std::pair<py::array_t<int64_t>, py::array_t<float>> search_and_rerank(
        py::array_t<float> py_queries,
        py::array_t<float> py_rd_queries,
        uint32_t k,
        uint32_t nprobe,
        uint32_t nrerank
    ) {

        if (!index) {
            throw std::runtime_error("Index not built");
        }


        auto queries_buf = py_queries.request();
        auto rd_queries_buf = py_rd_queries.request();

        uint32_t nq = queries_buf.shape[0];

        float* queries = static_cast<float*>(queries_buf.ptr);
        float* rd_queries = static_cast<float*>(rd_queries_buf.ptr);

        // Allocate result arrays
        py::array_t<int64_t> indices({nq, k});
        py::array_t<float> distances({nq, k});

        auto idx_buf = indices.request();
        auto dist_buf = distances.request();
        int64_t* idx_ptr = static_cast<int64_t*>(idx_buf.ptr);
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);

        int thread_num = omp_get_max_threads();
        // Search each query
        #pragma omp parallel for num_threads(thread_num)
        for (uint32_t i = 0; i < nq; i++) {

            ResultHeap result = index->search_and_rerank(
                queries + i * D,
                rd_queries + i * B,
                nrerank,
                nprobe
            );

            // Extract results (they come in reverse order from heap)
            std::vector<std::pair<float, uint32_t>> res_vec;
            while (!result.empty()) {
                res_vec.push_back(result.top());
                result.pop();
            }
            std::reverse(res_vec.begin(), res_vec.end());

            // Fill output arrays
            for (uint32_t j = 0; j < k; j++) {
                if (j < res_vec.size()) {
                    idx_ptr[i * k + j] = res_vec[j].second;
                    dist_ptr[i * k + j] = res_vec[j].first;
                } else {
                    idx_ptr[i * k + j] = -1;
                    dist_ptr[i * k + j] = std::numeric_limits<float>::max();
                }
            }
        }

        return std::make_pair(indices, distances);
    }

    // Save index
    void save(const std::string& filename) {
        if (!index) {
            throw std::runtime_error("Index not built");
        }
        index->save(const_cast<char*>(filename.c_str()));
    }

    // Load index
    void load(const std::string& filename) {
        if (!index) {
            index = new IVFRN<D, B>();
        }
        index->load(const_cast<char*>(filename.c_str()));
    }

    float getMSE(
        py::array_t<float> py_queries,
        py::array_t<float> py_rd_queries
    ){
        float mse = 0.0f;
        float ip_diff = 0.0f;
        auto queries_buf = py_queries.request();
        auto rd_queries_buf = py_rd_queries.request();

        uint32_t nq = queries_buf.shape[0];
        uint32_t dim = queries_buf.shape[1];

        float* queries = static_cast<float*>(queries_buf.ptr);
        float* rd_queries = static_cast<float*>(rd_queries_buf.ptr);


        assert(nq == N);
        uint32_t *id2pos = new uint32_t[N];
        uint32_t *cluster_id = new uint32_t[N];
        float *centroid_dist = new float[N];
        float *ip_correct = new float[N];
        for (uint32_t i = 0; i < C; i++){
            uint32_t start = index->start[i];
            uint32_t end = (i == C - 1) ? N : (index->start[i + 1]);
            #pragma omp parallel for
            for (uint32_t j = start; j < end; j++){
                uint32_t id = index->id[j];
                id2pos[id] = j;
                cluster_id[id] = i;
                centroid_dist[id] = sqr_dist<B>(rd_queries + id * B, index->centroid + i * B);
                ip_correct[id] = 0.0f;
                for (uint32_t d = 0; d < dim; d++){
                    ip_correct[id] += queries[id * dim + d] * queries[id * dim + d];
                }
            }
        }

        uint32_t thread_num = omp_get_max_threads();
        uint8_t  PORTABLE_ALIGN64 byte_query[B * thread_num];

        #pragma omp parallel for num_threads(thread_num) reduction(+:mse)
        for (uint32_t i = 0; i < N; i++){
            float* rd_query = rd_queries + i * B;
            uint32_t c = cluster_id[i];
            uint32_t thread_id = omp_get_thread_num();
            float vl, vr;
            space.range(rd_query, index->centroid + c * B, vl, vr);
            float width = (vr - vl) / ((1 << B_QUERY) - 1);
            uint32_t sum_q = 0;
            space.quantize(byte_query + thread_id * B, rd_query, index->centroid + c * B, index->u, vl, width, sum_q);
            uint64_t PORTABLE_ALIGN32 quant_query[B_QUERY * B / 64];
            memset(quant_query, 0, sizeof(quant_query));
            space.transpose_bin(byte_query + thread_id * B, quant_query);
            float sqr_y = centroid_dist[i];
            auto ptr_fac = index->fac + id2pos[i];
            uint64_t* ptr_binary_code = index->binary_code + 1ull * id2pos[i] * (B / 64);
            float ip_byte = space.ip_byte_bin(quant_query, ptr_binary_code);
            float tmp_dist = (ptr_fac -> sqr_x) + sqr_y + ptr_fac -> factor_ppc * vl + (ip_byte * 2 -sum_q) * (ptr_fac -> factor_ip) * width;
            float ip_estimate = (tmp_dist - 2 * ip_correct[i]) / (-2);
            ip_diff += abs(ip_estimate - ip_correct[i]);
            mse += abs(tmp_dist);
        }

        mse /= N;
        ip_diff /= N;
        std::cout << "IP Relative Error: " << ip_diff << std::endl;
        return mse;
    }

    /* This function will estimate the ip between each query and every data vectors*/
    py::array_t<float> estimate_ip(
        py::array_t<float> py_queries,
        py::array_t<float> py_rd_queries
    ){
        std::cout << "[DEBUG]entering estimate ip " << std::endl;
        auto queries_buf = py_queries.request();
        auto rd_queries_buf = py_rd_queries.request();

        uint32_t nq = queries_buf.shape[0];
        uint32_t dim = queries_buf.shape[1];

        float* queries = static_cast<float*>(queries_buf.ptr);
        float* rd_queries = static_cast<float*>(rd_queries_buf.ptr);

        uint32_t C = index->C;

        float* query_norms = new float[nq];
        #pragma omp parallel for
        for (uint32_t i = 0; i < nq; i++){
            query_norms[i] = 0;
            for (uint32_t d = 0; d < dim; d++){
                query_norms[i] += queries[i * dim + d] * queries[i * dim + d];
            }
        }

        float* base_norms = new float[N];
        memset(base_norms, 0, sizeof(float)*N);
        #pragma omp parallel for
        for (uint32_t i = 0; i < N; i++){
            uint32_t id = index->id[i];
            base_norms[id] = 0;
            for (uint32_t d = 0; d < dim; d++){
                base_norms[id] += index->data[i * dim + d] * index->data[i * dim + d];
            }
        }

        // std::cout << "base_norms[4789]: " << base_norms[4789] << std::endl;

        py::array_t<float> distances({nq, N});
        auto dist_buf = distances.request();
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);
        memset(dist_ptr, 0, sizeof(float)*nq*N);
        std:cout << "[DEBUG]N: " << N << ", C: " << C << std::endl;

        #pragma omp parallel for
        for (size_t i = 0; i < nq; i++){
            float* rd_query = rd_queries + i * B;
            uint8_t  PORTABLE_ALIGN64 byte_query[B];   
            for (size_t c = 0; c < C; c++){
                float sqr_y = sqr_dist<B>(rd_query, index->centroid + c * B);
                float vl, vr;
                space.range(rd_query, index->centroid + c * B, vl, vr);
                float width = (vr - vl) / ((1 << B_QUERY) - 1);
                uint32_t sum_q = 0;
                space.quantize(byte_query, rd_query, index->centroid + c * B, index->u, vl, width, sum_q);
                uint8_t PORTABLE_ALIGN32 LUT[B / 4 * 16];
                pack_LUT<B>(byte_query, LUT);
                for(int ii=0;ii<B/4*16;ii++)LUT[ii] *= 2;
                constexpr uint32_t SIZE = 32;
                uint32_t len = index->len[c];
                uint32_t it = len / SIZE;
                uint32_t remain = len - it * SIZE;
                uint32_t nblk_remain = (remain + 31) / 32;
                uint8_t *packed_code = index->packed_code + index->packed_start[c];
                auto ptr_fac = index->fac + index->start[c];
                uint32_t* id = index->id + index->start[c];
                while (it--){
                    uint16_t PORTABLE_ALIGN32 result[SIZE];
                    accumulate<B>((SIZE / 32), packed_code, LUT, result);
                    packed_code += SIZE * B / 8;
                    for(int ii=0;ii<SIZE;ii++){
                        float tmp_dist = (ptr_fac -> sqr_x) + sqr_y + ptr_fac -> factor_ppc * vl + ((float)result[ii]-sum_q) * (ptr_fac -> factor_ip) * width;
                        uint32_t cur_id = *id;
                        // std::cout << i << ", " << cur_id << ", " << tmp_dist << std::endl;
                        // if (cur_id == 4789){
                        //     std::cout << i << ", " << ptr_fac->sqr_x << ", " << sqr_y << ", " << ptr_fac->factor_ppc << ", " << vl << ", " << result[ii] << ", " << sum_q << ", " << ptr_fac -> factor_ip << ", " << width << ", " << std::endl;
                        // }
                        ptr_fac ++;
                        dist_ptr[i * N + cur_id] = (tmp_dist - query_norms[i] - base_norms[cur_id]) / (-2);
                        id++;
                    }
                }
                {
                    uint16_t PORTABLE_ALIGN32 result[SIZE];
                    accumulate<B>(nblk_remain, packed_code, LUT, result);
                    for(int ii=0;ii<remain;ii++){
                        float tmp_dist = (ptr_fac -> sqr_x) + sqr_y + ptr_fac -> factor_ppc * vl + ((float)result[ii] - sum_q) * ptr_fac -> factor_ip * width;
                        uint32_t cur_id = *id;
                        // if (cur_id == 4789){
                        //     std::cout << i << ", " << ptr_fac->sqr_x << ", " << sqr_y << ", " << ptr_fac->factor_ppc << ", " << vl << ", " << result[ii] << ", " << sum_q << ", " << ptr_fac -> factor_ip << ", " << width << ", " << std::endl;
                        // }
                        ptr_fac++;
                        dist_ptr[i * N + cur_id] = (tmp_dist - query_norms[i] - base_norms[cur_id]) / (-2);
                        id++;
                    }
                }
            }
        }
        
        return distances;
    }
};

PYBIND11_MODULE(rabitq_cpp, m) {
    m.doc() = "RaBitQ C++ bindings (IVFRN with C=1)";

    // Instantiate for dimension 128, B=128 (rounded to 128)
    py::class_<PyIVFRN<128, 128>>(m, "PyIVFRN_128_128")
        .def(py::init<>())
        .def("estimate_ip", &PyIVFRN<128, 128>::estimate_ip,
            py::arg("py_queries"),
            py::arg("py_rd_queries"))
        .def("set_query", &PyIVFRN<128,128>::set_query,
            py::arg("py_query"),
            py::arg("py_rd_query"))
        .def("estimate_distance", &PyIVFRN<128,128>::estimate_distance_after_set_query,
            py::arg("id"))
        .def("build", &PyIVFRN<128, 128>::build,
             py::arg("data"),
             py::arg("centroids"),
             py::arg("dist_to_centroid"),
             py::arg("x0"),
             py::arg("cluster_id"),
             py::arg("binary"))
        .def("search", &PyIVFRN<128, 128>::search,
             py::arg("queries"),
             py::arg("rd_queries"),
             py::arg("k"),
             py::arg("nprobe"))
        .def("search_clusters", &PyIVFRN<128,128>::search_clusters,
            py::arg("queries"),
            py::arg("rd_queries"),
            py::arg("assignments"),
            py::arg("k"))
        .def("search_and_rerank_clusters", &PyIVFRN<128,128>::search_and_rerank_clusters,
            py::arg("queries"),
            py::arg("rd_queries"),
            py::arg("assignments"),
            py::arg("k"),
            py::arg("nrerank")
        )
        .def("save", &PyIVFRN<128, 128>::save,
             py::arg("filename"))
        .def("load", &PyIVFRN<128, 128>::load,
             py::arg("filename"))
        .def("getMSE", &PyIVFRN<128, 128>::getMSE,
             py::arg("queries"),
             py::arg("rd_queries"))
        .def("search_and_rerank", &PyIVFRN<128, 128>::search_and_rerank,
             py::arg("queries"),
             py::arg("rd_queries"),
             py::arg("k"),
             py::arg("nprobe"),
             py::arg("nrerank"));


    // Add more dimension instantiations as needed
    // For 960 dimensions (common in embeddings), B would be 960 rounded to 64-multiple = 960
    py::class_<PyIVFRN<960, 960>>(m, "PyIVFRN_960_960")
        .def(py::init<>())
        .def("set_query", &PyIVFRN<960,960>::set_query,
            py::arg("py_query"),
            py::arg("py_rd_query"))
        .def("estimate_distance", &PyIVFRN<960,960>::estimate_distance_after_set_query,
            py::arg("id"))
        .def("build", &PyIVFRN<960, 960>::build,
             py::arg("data"),
             py::arg("centroids"),
             py::arg("dist_to_centroid"),
             py::arg("x0"),
             py::arg("cluster_id"),
             py::arg("binary"))
        .def("search", &PyIVFRN<960, 960>::search,
             py::arg("queries"),
             py::arg("rd_queries"),
             py::arg("k"),
             py::arg("nprobe"))
        .def("search_clusters", &PyIVFRN<960,960>::search_clusters,
            py::arg("queries"),
            py::arg("rd_queries"),
            py::arg("assignments"),
            py::arg("k"))
        .def("search_and_rerank_clusters", &PyIVFRN<960,960>::search_and_rerank_clusters,
            py::arg("queries"),
            py::arg("rd_queries"),
            py::arg("assignments"),
            py::arg("k"),
            py::arg("nrerank")
        )
        .def("save", &PyIVFRN<960, 960>::save,
             py::arg("filename"))
        .def("load", &PyIVFRN<960, 960>::load,
             py::arg("filename"))
        .def("getMSE", &PyIVFRN<960, 960>::getMSE,
             py::arg("queries"),
             py::arg("rd_queries"))
        .def("search_and_rerank", &PyIVFRN<960, 960>::search_and_rerank,
             py::arg("queries"),
             py::arg("rd_queries"),
             py::arg("k"),
             py::arg("nprobe"),
             py::arg("nrerank"));
}