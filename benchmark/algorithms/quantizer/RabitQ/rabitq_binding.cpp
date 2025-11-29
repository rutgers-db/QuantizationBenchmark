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
public:
    PyIVFRN() : index(nullptr) {}

    ~PyIVFRN() {
        if (index) delete index;
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
        cout << "[C++ Search] Number of threads: " << thread_num <<endl;
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
        py::array_t<float> py_rd_queries,
        uint32_t k
    ){
        float mse = 0.0f;
        auto queries_buf = py_queries.request();
        auto rd_queries_buf = py_rd_queries.request();

        uint32_t nq = queries_buf.shape[0];

        float* queries = static_cast<float*>(queries_buf.ptr);
        float* rd_queries = static_cast<float*>(rd_queries_buf.ptr);

        // Allocate result arrays

        int thread_num = omp_get_max_threads();
        std::mutex mtx;
        int found_count = 0;
        int negative_count = 0;
        // Search each query
        #pragma omp parallel for num_threads(thread_num)
        for (uint32_t i = 0; i < nq; i++) {
            
            ResultHeap result = index->search(
                queries + i * D,
                rd_queries + i * B,
                k,
                1
            );

            // Extract results (they come in reverse order from heap)
            bool found = false;

            while (!result.empty()) {
                if (result.top().second == i){
                    mtx.lock();
                    found_count++;
                    mse += abs(result.top().first);
                    if (result.top().first < 0){
                        negative_count++;
                    }
                    mtx.unlock();
                }
                result.pop();
            }


        }
        
        std::cout << "found_count: " << found_count << std::endl;
        std::cout << "negative_count: " << negative_count << std::endl;
        mse /= found_count;
        return mse;
    }
};

PYBIND11_MODULE(rabitq_cpp, m) {
    m.doc() = "RaBitQ C++ bindings (IVFRN with C=1)";

    // Instantiate for dimension 128, B=128 (rounded to 128)
    py::class_<PyIVFRN<128, 128>>(m, "PyIVFRN_128_128")
        .def(py::init<>())
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
        .def("save", &PyIVFRN<128, 128>::save,
             py::arg("filename"))
        .def("load", &PyIVFRN<128, 128>::load,
             py::arg("filename"))
        .def("getMSE", &PyIVFRN<128, 128>::getMSE,
             py::arg("queries"),
             py::arg("rd_queries"),
             py::arg("k"));

    // Add more dimension instantiations as needed
    // For 960 dimensions (common in embeddings), B would be 960 rounded to 64-multiple = 960
    py::class_<PyIVFRN<960, 960>>(m, "PyIVFRN_960_960")
        .def(py::init<>())
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
        .def("save", &PyIVFRN<960, 960>::save,
             py::arg("filename"))
        .def("load", &PyIVFRN<960, 960>::load,
             py::arg("filename"))
        .def("getMSE", &PyIVFRN<960, 960>::getMSE,
             py::arg("queries"),
             py::arg("rd_queries"),
             py::arg("k"));
}
