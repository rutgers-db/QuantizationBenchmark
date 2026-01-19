// Pybind11 wrapper for VAQ (Vector Adaptive Quantization)

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Include VAQ headers
#include "VAQ.hpp"
#include <omp.h>

namespace py = pybind11;

// Python wrapper for VAQ
class PyVAQ {
private:
    VAQ* index;
    RowMatrixXf train_data;
    bool is_trained;
    bool is_encoded;
    LUTType lut;

public:
    PyVAQ() : index(nullptr), is_trained(false), is_encoded(false) {
        index = new VAQ();
    }

    ~PyVAQ() {
        if (index) delete index;
    }

    // Parse method string (e.g., "VAQ256m32min7max13var1,EA")
    void parseMethodString(const std::string& method_string) {
        if (!index) {
            throw std::runtime_error("Index not initialized");
        }
        index->parseMethodString(method_string);
        // Note: lut is initialized in train() after mHighestSubs is computed
    }


    // Train the VAQ index
    void train(
        py::array_t<float, py::array::c_style | py::array::forcecast> py_data,
        bool verbose = false
    ) {
        if (!index) {
            throw std::runtime_error("Index not initialized");
        }

        // Request buffer
        auto data_buf = py_data.request();

        // Verify buffer properties
        if (data_buf.ndim != 2) {
            throw std::runtime_error("data must be 2-dimensional");
        }

        int n_rows = data_buf.shape[0];
        int n_cols = data_buf.shape[1];
        float* data_ptr = static_cast<float*>(data_buf.ptr);

        // Copy data to Eigen matrix (deep copy to avoid lifetime issues)
        train_data = RowMatrixXf(n_rows, n_cols);
        std::memcpy(train_data.data(), data_ptr, n_rows * n_cols * sizeof(float));

        // Train the index
        index->train(train_data, verbose);
        is_trained = true;
        index->encode(train_data);
        is_encoded = true;

        // Initialize LUT now that mHighestSubs is computed
        lut = LUTType(1 << (index->mMaxBitsPerSubs), index->mHighestSubs);
    }

    // Search for k nearest neighbors
    std::pair<py::array_t<int64_t>, py::array_t<float>> search(
        py::array_t<float> py_queries,
        int k,
        bool verbose = false
    ) {
        if (!index) {
            throw std::runtime_error("Index not initialized");
        }
        if (!is_trained || !is_encoded) {
            throw std::runtime_error("Index not trained or encoded. Call train() and encode() first.");
        }
        
        auto queries_buf = py_queries.request();

        if (queries_buf.ndim != 2) {
            throw std::runtime_error("queries must be 2-dimensional");
        }

        int nq = queries_buf.shape[0];
        int dim = queries_buf.shape[1];
        float* queries_ptr = static_cast<float*>(queries_buf.ptr);

        // Create Eigen matrix for queries (deep copy)
        RowMatrixXf queries(nq, dim);
        std::memcpy(queries.data(), queries_ptr, nq * dim * sizeof(float));

        // Search
        int thread_num = omp_get_max_threads();
        if (verbose) {
            std::cout << "[C++ Search] Number of threads: " << thread_num << std::endl;
        }

        LabelDistVecF results = index->search(queries, k, verbose);

        // Allocate result arrays
        py::array_t<int64_t> indices({nq, k});
        py::array_t<float> distances({nq, k});

        auto idx_buf = indices.request();
        auto dist_buf = distances.request();
        int64_t* idx_ptr = static_cast<int64_t*>(idx_buf.ptr);
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);

        // Fill output arrays
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                int result_idx = i * k + j;
                if (result_idx < static_cast<int>(results.labels.size())) {
                    idx_ptr[result_idx] = results.labels[result_idx];
                    dist_ptr[result_idx] = results.distances[result_idx];
                } else {
                    idx_ptr[result_idx] = -1;
                    dist_ptr[result_idx] = std::numeric_limits<float>::max();
                }
            }
        }

        return std::make_pair(indices, distances);
    }

    // Search and rerank (refine) results
    std::pair<py::array_t<int64_t>, py::array_t<float>> search_and_rerank(
        py::array_t<float> py_queries,
        py::array_t<float> py_train_data,
        int k,
        int nrerank,
        bool verbose = false
    ) {
        if (!index) {
            throw std::runtime_error("Index not initialized");
        }
        if (!is_trained || !is_encoded) {
            throw std::runtime_error("Index not trained or encoded. Call train() and encode() first.");
        }

        auto queries_buf = py_queries.request();
        auto train_buf = py_train_data.request();

        if (queries_buf.ndim != 2) {
            throw std::runtime_error("queries must be 2-dimensional");
        }
        if (train_buf.ndim != 2) {
            throw std::runtime_error("train_data must be 2-dimensional");
        }

        int nq = queries_buf.shape[0];
        int dim = queries_buf.shape[1];
        float* queries_ptr = static_cast<float*>(queries_buf.ptr);

        int n_train = train_buf.shape[0];
        int train_dim = train_buf.shape[1];
        float* train_ptr = static_cast<float*>(train_buf.ptr);

        // Create Eigen matrices (deep copies)
        RowMatrixXf queries(nq, dim);
        std::memcpy(queries.data(), queries_ptr, nq * dim * sizeof(float));

        RowMatrixXf train_data_refine(n_train, train_dim);
        std::memcpy(train_data_refine.data(), train_ptr, n_train * train_dim * sizeof(float));

        // First, search with larger k (nrerank)
        int thread_num = omp_get_max_threads();
        if (verbose) {
            std::cout << "[C++ Search and Rerank] Number of threads: " << thread_num << std::endl;
        }

        LabelDistVecF initial_results = index->search(queries, nrerank, verbose);

        // Then refine the results
        LabelDistVecF refined_results = index->refine(queries, initial_results, train_data_refine, k);

        // Allocate result arrays
        py::array_t<int64_t> indices({nq, k});
        py::array_t<float> distances({nq, k});

        auto idx_buf = indices.request();
        auto dist_buf = distances.request();
        int64_t* idx_ptr = static_cast<int64_t*>(idx_buf.ptr);
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);

        // Fill output arrays
        for (int i = 0; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                int result_idx = i * k + j;
                if (result_idx < static_cast<int>(refined_results.labels.size())) {
                    idx_ptr[result_idx] = refined_results.labels[result_idx];
                    dist_ptr[result_idx] = refined_results.distances[result_idx];
                } else {
                    idx_ptr[result_idx] = -1;
                    dist_ptr[result_idx] = std::numeric_limits<float>::max();
                }
            }
        }

        return std::make_pair(indices, distances);
    }

    float getMSE(){
        if (!index) {
            throw std::runtime_error("Index not initialized");
        }
        if (!is_trained || !is_encoded) {
            throw std::runtime_error("Index not trained or encoded. Call train() and encode() first.");
        }

        double total_squared_error = 0.0;
        int nd = train_data.rows();
        std::cout << "[C++ getMSE]: nd = " << train_data.rows() << ", dim = " << train_data.cols() << std::endl;

        // For each subspace
        for (int i = 0; i < index->mHighestSubs; i++) {
            // For each data point
            for (int rowIdx = 0; rowIdx < train_data.rows(); rowIdx++) {
                // Get the assigned centroid code
                uint16_t assigned_code = index->mCodebook(rowIdx, i);

                // Calculate squared distance between data point's subspace and its assigned centroid
                float dist = (train_data.block(rowIdx, i * index->mSubsLen, 1, index->mSubsLen) -
                             index->mCentroidsPerSubs[i].block(assigned_code, 0, 1, index->mSubsLen)).squaredNorm();

                total_squared_error += dist;
            }
        }

        // Return mean squared error
        float mse = static_cast<float>(total_squared_error / nd);

        std::cout << "[C++ getMSE] MSE: " << mse << std::endl;

        return mse;
    }

    void set_query(py::array_t<float> py_query){
        auto queries_buf = py_query.request();

        if (queries_buf.ndim != 2) {
            throw std::runtime_error("queries must be 2-dimensional");
        }

        int nq = queries_buf.shape[0];
        int dim = queries_buf.shape[1];
        float* queries_ptr = static_cast<float*>(queries_buf.ptr);
        assert(nq == 1);

        // Create Eigen matrix for queries (deep copy)
        RowMatrixXf queries(nq, dim);
        std::memcpy(queries.data(), queries_ptr, nq * dim * sizeof(float));
        RowMatrixXf XTestPCA = index->ProjectOnEigenVectors(queries);
        switch (index->mMaxBitsPerSubs) {
            case 9: index->CreateLUT<9>(XTestPCA.row(0), lut); break;
            case 10: index->CreateLUT<10>(XTestPCA.row(0), lut); break;
            case 11: index->CreateLUT<11>(XTestPCA.row(0), lut); break;
            case 12: index->CreateLUT<12>(XTestPCA.row(0), lut); break;
            case 13: index->CreateLUT<13>(XTestPCA.row(0), lut); break;
            case 14: index->CreateLUT<14>(XTestPCA.row(0), lut); break;
            case 15: index->CreateLUT<15>(XTestPCA.row(0), lut); break;
            default:
                index->CreateLUT(XTestPCA.row(0), lut);
                break;
        }
    }

    float estimate_distance(int idx){
        float result = 0.0f;
        auto luts = lut.data();
        auto code = index->mCodebook.data() + index->mCodebook.cols() * idx;
        for (int col=0; col < index->mHighestSubs; col++) {
            result += luts[code[col]];
            luts += lut.rows();
        }
        return result;
    }
};

PYBIND11_MODULE(vaq_cpp, m) {
    m.doc() = "VAQ (Vector Adaptive Quantization) C++ bindings";

    py::class_<PyVAQ>(m, "PyVAQ")
        .def(py::init<>())
        .def("set_query", &PyVAQ::set_query,
            py::arg("query"),
            "set query")
        .def("estimate_distance", &PyVAQ::estimate_distance, py::arg("idx"), "Estimate distance between idx and the setted query")
        .def("parse_method_string", &PyVAQ::parseMethodString,
             py::arg("method_string"),
             "Parse VAQ method string (e.g., 'VAQ256m32min7max13var1,SORT')")
        .def("train", &PyVAQ::train,
             py::arg("data"),
             py::arg("verbose") = false,
             "Train the VAQ index on the given data")
        .def("get_mse", &PyVAQ::getMSE,
             "Get Mean Squared Error between training data and assigned centroids")
        .def("search", &PyVAQ::search,
             py::arg("queries"),
             py::arg("k"),
             py::arg("verbose") = false,
             "Search for k nearest neighbors")
        .def("search_and_rerank", &PyVAQ::search_and_rerank,
             py::arg("queries"),
             py::arg("train_data"),
             py::arg("k"),
             py::arg("nrerank"),
             py::arg("verbose") = false,
             "Search and rerank with exact distances");

    // Expose NNMethod flags as module constants
    // These are anonymous enum values from VAQ::NNMethod
    m.attr("SORT") = py::int_(0x01u);
    m.attr("EA") = py::int_(0x02u);
    m.attr("TI") = py::int_(0x04u);
    m.attr("FAST") = py::int_(0x08u);
    m.attr("FAST2") = py::int_(0x10u);
    m.attr("FAST3") = py::int_(0x20u);
    m.attr("FAST4") = py::int_(0x40u);
    m.attr("HEAP") = py::int_(0x80u);
}
