// PyBind11 wrapper for DiskANN with pluggable quantizer
// This allows using any Python quantizer with DiskANN's disk-based index

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <memory>
#include <iostream>

// Include DiskANN headers
#include "pq_flash_index.h"
#include "disk_utils.h"
#include "utils.h"
#include "linux_aligned_file_reader.h"
#include "partition.h"
#include "index.h"


namespace py = pybind11;

/**
 * @brief Wrapper class for DiskANN index with pluggable quantizer
 *
 * This class wraps DiskANN's build and search functionality, replacing
 * the built-in PQ with any Python quantizer that implements set_query()
 * and estimate_distance().
 */
class DiskANNWithQuantizer {
private:
    std::unique_ptr<diskann::QuantizerAdapter> quantizer;
    std::string index_prefix;
    std::string base_file;
    uint32_t R;  // Max degree
    uint32_t L;  // Build complexity
    uint32_t num_threads;
    uint32_t dimension;
    uint32_t num_points;

    // Metric
    diskann::Metric metric;

public:
    /**
     * @brief Construct a new DiskANN wrapper
     *
     * @param quantizer_obj Python quantizer object
     * @param index_prefix Prefix for index files
     * @param R Max out-degree
     * @param L Build search list size
     * @param num_threads Number of threads
     * @param metric Distance metric ("l2" or "mips")
     */
    DiskANNWithQuantizer(py::object quantizer_obj,
                         const std::string& index_prefix,
                         uint32_t R,
                         uint32_t L,
                         uint32_t num_threads,
                         const std::string& metric_str)
        : index_prefix(index_prefix), R(R), L(L), num_threads(num_threads) {

        quantizer = std::make_unique<diskann::QuantizerAdapter>(quantizer_obj);

        if (metric_str == "l2") {
            metric = diskann::Metric::L2;
        } else {
            throw std::runtime_error("Unsupported metric: " + metric_str);
        }

        std::cout << "DiskANN initialized with R=" << R << ", L=" << L
                  << ", threads=" << num_threads << std::endl;
    }

    /**
     * @brief Build the disk-based index
     *
     * This saves data to disk and builds a Vamana graph using DiskANN's
     * standard build process. The quantizer is trained on the data.
     *
     * @param data Training data (num_points, dimension)
     * @param build_memory_gb Memory budget for building in GB
     */
    void build(py::array_t<float> data, float build_memory_gb) {
        auto buf = data.request();
        num_points = buf.shape[0];
        dimension = buf.shape[1];

        std::cout << "Building DiskANN index for " << num_points
                  << " points, dim=" << dimension << std::endl;

        // 1. Train quantizer
        std::cout << "Training quantizer..." << std::endl;
        quantizer->train(data);

        // 2. Save data to disk in DiskANN format
        base_file = index_prefix + "_data.bin";
        std::cout << "Saving data to " << base_file << std::endl;

        auto data_ptr = buf.ptr;
        std::ofstream writer(base_file, std::ios::binary);

        // Write num_points and dimension
        writer.write(reinterpret_cast<const char*>(&num_points), sizeof(uint32_t));
        writer.write(reinterpret_cast<const char*>(&dimension), sizeof(uint32_t));

        // Write data
        writer.write((char*)data_ptr, sizeof(float)*(size_t)num_points*(size_t)dimension);
        writer.close();

        // 3. Build DiskANN index using disk_utils
        std::cout << "Building Vamana graph..." << std::endl;
        const double p_val = ((double)MAX_PQ_TRAINING_SET_SIZE / (double)num_points);

        std::string data_file_to_use = base_file;
        std::string mem_index_path = index_prefix + "_mem.index";
        std::string merged_index_prefix = mem_index_path + "_tempFiles";
        std::string cur_centroid_filepath = merged_index_prefix + "_centroids.bin";
        std::string disk_index_path = index_prefix + "_disk.index";
        std::string medoids_path = disk_index_path + "_medoids.bin";
        std::string labels_to_medoids_path = disk_index_path + "_labels_to_medoids.txt";
        std::string centroids_path = disk_index_path + "_centroids.bin";
        std::rename(cur_centroid_filepath.c_str(), centroids_path.c_str());
        int num_parts = partition_with_ram_budget<float>(base_file, p_val, build_memory_gb, 2 * R / 3, merged_index_prefix, 2);

        for (int p = 0; p < num_parts; p++){
            std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
            std::string shard_ids_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
            std::string shard_labels_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_labels.txt";
            retrieve_shard_data_from_ids<float>(base_file, shard_ids_file, shard_base_file);
            std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
            diskann::IndexWriteParameters low_degree_params = diskann::IndexWriteParametersBuilder(L, 2 * R / 3)
                                                              .with_filter_list_size(0)
                                                              .with_saturate_graph(false)
                                                              .with_num_threads(num_threads)
                                                              .build();
            uint64_t shard_base_dim, shard_base_pts;
            diskann::get_bin_metadata(shard_base_file, shard_base_pts, shard_base_dim);
            diskann::Index<float> _index(metric, shard_base_dim, shard_base_pts,
                                 std::make_shared<diskann::IndexWriteParameters>(low_degree_params), nullptr,
                                 0, false, false, false, false,
                                 0, false);
            _index.build(shard_base_file.c_str(), shard_base_pts);
            _index.save(shard_index_file.c_str());
            std::remove(shard_base_file.c_str());
        }
    
        diskann::merge_shards(merged_index_prefix + "_subshard-", "_mem.index", merged_index_prefix + "_subshard-",
                          "_ids_uint32.bin", num_parts, R, mem_index_path, medoids_path, false,
                          labels_to_medoids_path);
        for (int p = 0; p < num_parts; p++)
        {
            std::string shard_base_file = merged_index_prefix + "_subshard-" + std::to_string(p) + ".bin";
            std::string shard_id_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_ids_uint32.bin";
            std::string shard_labels_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_labels.txt";
            std::string shard_index_file = merged_index_prefix + "_subshard-" + std::to_string(p) + "_mem.index";
            std::string shard_index_file_data = shard_index_file + ".data";

            std::remove(shard_base_file.c_str());
            std::remove(shard_id_file.c_str());
            std::remove(shard_index_file.c_str());
            std::remove(shard_index_file_data.c_str());
        }

        diskann::create_disk_layout<float>(data_file_to_use.c_str(), mem_index_path, disk_index_path);
    }

    /**
     * @brief Search using the quantizer for approximate distances
     *
     * @param queries Query vectors (nq, dimension)
     * @param k Number of neighbors to return
     * @param search_L Search list size
     * @param beam_width Beam width for search
     * @return Tuple of (indices, distances)
     */
    std::tuple<py::array_t<uint32_t>, py::array_t<float>> search(
        py::array_t<float> queries,
        uint32_t k,
        uint32_t search_L,
        uint32_t beam_width) {
        
        auto queries_buf = queries.request();
        uint32_t nq = queries_buf.shape[0];
        uint32_t dim = queries_buf.shape[1];
        const float* query = static_cast<const float*>(queries_buf.ptr);

        py::array_t<uint64_t> indices({nq, k});
        py::array_t<float> distances({nq, k});
        auto idx_buf = indices.request();
        auto dist_buf = distances.request();
        uint64_t* idx_ptr = static_cast<uint64_t*>(idx_buf.ptr);
        float* dist_ptr = static_cast<float*>(dist_buf.ptr);

        std::shared_ptr<AlignedFileReader> reader = nullptr;
        reader.reset(new LinuxAlignedFileReader());
        std::unique_ptr<diskann::PQFlashIndex<float, uint32_t>> _pFlashIndex(new diskann::PQFlashIndex<float, uint32_t>(reader, metric));
        _pFlashIndex->load(num_threads, index_prefix.c_str());
        std::vector<uint32_t> node_list;
        _pFlashIndex->cache_bfs_levels(100, node_list);
        _pFlashIndex->load_cache_list(node_list);
        node_list.clear();
        node_list.shrink_to_fit();
        auto stats = new diskann::QueryStats[nq];

        int progress_chunk = nq / 100;

        for (int64_t i = 0; i < nq; i++){
            if (i % progress_chunk == 0){
                int progress = i / progress_chunk;
                std::cout << "search progress: " << progress << "%" << std::endl;
            }
            quantizer->set_query(query + i * dim, dim, 0);
            _pFlashIndex->cached_beam_search(quantizer.get(), query + (i * dim), k, search_L,
                                                     idx_ptr + (i * k),
                                                     dist_ptr + (i * k),
                                                     beam_width, false, stats + i, 0);
        }

        std::cout << "search ends" << std::endl;

        delete[] stats;
        return std::make_tuple(indices, distances);
    }

    /**
     * @brief Get index statistics
     */
    py::dict get_stats() {
        py::dict stats;
        stats["num_points"] = num_points;
        stats["dimension"] = dimension;
        stats["R"] = R;
        stats["L"] = L;
        stats["index_prefix"] = index_prefix;
        return stats;
    }
};


PYBIND11_MODULE(diskann_cpp, m) {
    m.doc() = "DiskANN with pluggable quantizer - pybind11 wrapper";

    py::class_<DiskANNWithQuantizer>(m, "DiskANNIndex")
        .def(py::init<py::object, const std::string&, uint32_t, uint32_t, uint32_t, const std::string&>(),
             py::arg("quantizer"),
             py::arg("index_prefix"),
             py::arg("R") = 64,
             py::arg("L") = 100,
             py::arg("num_threads") = 16,
             py::arg("metric") = "l2",
             "Initialize DiskANN index with quantizer")
        .def("build", &DiskANNWithQuantizer::build,
             py::arg("data"),
             py::arg("build_memory_gb") = 8.0f,
             "Build the disk-based index")
        .def("search", &DiskANNWithQuantizer::search,
             py::arg("queries"),
             py::arg("k"),
             py::arg("search_L"),
             py::arg("beam_width") = 4,
             "Search for k nearest neighbors")
        .def("get_stats", &DiskANNWithQuantizer::get_stats,
             "Get index statistics");
}
