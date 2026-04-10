// #define HIGH_ACC_FAST_SCAN
// #define EIGEN_DONT_PARALLELIZE
#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "defines.hpp"
#include "index/IVF.hpp"
#include "utils/IO.hpp"
#include "utils/StopW.hpp"
namespace py = pybind11;
using namespace pybind11::literals;
class Index{

    IVF* ivf = nullptr;
    uint32_t dim;
    uint32_t Nlist = 1;
    uint32_t bit;
    size_t ndata = 0;
    bool constructed = false;

    public:
        Index(uint32_t D, uint32_t B, uint32_t K = 1):dim(D), Nlist(std::max<uint32_t>(1, K)), bit(B)
        {}
        ~Index(){ delete ivf; }
        void train(py::object input, py::object centroids_input, py::object cids, size_t N,  int num_threads = -1){
            py::array_t < float, py::array::c_style | py::array::forcecast > data_items(input);
            py::array_t < float, py::array::c_style | py::array::forcecast > centroids_items(centroids_input);
            py::array_t < uint32_t, py::array::c_style | py::array::forcecast > cids_items(cids);
            ndata = N;

            float* data_ptr = static_cast<float*> (data_items.request().ptr);
            float* centroids_ptr = static_cast<float*> (centroids_items.request().ptr);
            uint32_t* cids_ptr = static_cast<uint32_t*> (cids_items.request().ptr);

            if (ivf) { delete ivf; ivf = nullptr; }
            ivf = new IVF(N, dim, Nlist, bit);
            ivf->construct(data_ptr, centroids_ptr, cids_ptr);
            constructed = true;
        }
        py::object search(py::object input, size_t NQ, size_t TOPK = 1, size_t nprobe = 1, int num_threads = -1){
            py::array_t < float, py::array::c_style | py::array::forcecast > items(input);
            float* raw = static_cast<float*> (items.request().ptr);

            if (!constructed || ivf == nullptr) {
                throw std::runtime_error("Index must be trained before search");
            }
            nprobe = std::max<size_t>(1, std::min(nprobe, ivf->k()));
            Eigen::Map<FloatRowMat> padded_query(raw, NQ, ivf->padded_dim());

            uint32_t * ret_list = new uint32_t[NQ * TOPK]; 
            float* dist_list = new float[NQ * TOPK];
            Rotator& rp = ivf->rotator();
            FloatRowMat rotated_query(NQ, ivf->padded_dim());
            rp.rotate(padded_query, rotated_query);
#pragma omp parallel for if(num_threads > 1)
            for (size_t i = 0; i < NQ; i++) {
                ivf->search(&rotated_query(i, 0), raw + i * dim, TOPK, nprobe, ret_list + i * TOPK, dist_list + i * TOPK);
            }
            
            py::capsule free_when_done_id(ret_list, [](void* f) {
            delete[] f;
            });
            py::capsule free_when_done_dist(dist_list, [](void* f) {
            delete[] f;
            });

        return py::make_tuple(
            py::array_t<uint32_t>(
                { NQ, TOPK },  // shape
                { TOPK * sizeof(uint32_t),
                  sizeof(uint32_t) },  // C-style contiguous strides for each index
                ret_list,  // the data pointer
                free_when_done_id),
            py::array_t<float>(
                { NQ, TOPK },  // shape
                { TOPK * sizeof(float), sizeof(float) },  // C-style contiguous strides for each index
                dist_list,  // the data pointer
                free_when_done_dist));
        }

        double getMSE(py::object input, py::object centroids_input, py::object cids, size_t N,  int num_threads = -1){
            py::array_t < float, py::array::c_style | py::array::forcecast > data_items(input);
            py::array_t < float, py::array::c_style | py::array::forcecast > centroids_items(centroids_input);
            py::array_t < uint32_t, py::array::c_style | py::array::forcecast > cids_items(cids);
            ndata = N;
            if(!constructed){
                throw std::runtime_error("MSE must be used after index constructed");
            }
            float* data_ptr = static_cast<float*> (data_items.request().ptr);
            float* centroids_ptr = static_cast<float*> (centroids_items.request().ptr);
            uint32_t* cids_ptr = static_cast<uint32_t*> (cids_items.request().ptr);
            return ivf->get_mse(data_ptr,centroids_ptr, cids_ptr); 
        }



};


#ifdef HIGH_ACC_FAST_SCAN
#define MODULE_NAME ExtendedRabitQ_HighAcc
struct IndexHA:public Index{using Index::Index;};
#else
#define MODULE_NAME ExtendedRabitQ
struct IndexNHA:public Index{using Index::Index;};
#endif
PYBIND11_MODULE(MODULE_NAME, m) {

#ifdef HIGH_ACC_FAST_SCAN
    py::class_<IndexHA>(m, "Index")
    .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("D"), py::arg("B"), py::arg("K") = 1)
    .def("train", &Index::train, py::arg("input"), py::arg("centroids"), py::arg("cids"), py::arg("N"), py::arg("num_threads")  = -1)
    .def("search", &Index::search, py::arg("input") , py::arg("NQ"), py::arg("TOPK") = 1, py::arg("nprobe") = 1, py::arg("num_threads") = -1)
    .def("getMSE", &Index::getMSE);
#else
    py::class_<IndexNHA>(m, "Index")
    .def(py::init<uint32_t, uint32_t, uint32_t>(), py::arg("D"), py::arg("B"), py::arg("K") = 1)
    .def("train", &Index::train, py::arg("input"), py::arg("centroids"), py::arg("cids"), py::arg("N"), py::arg("num_threads")  = -1)
    .def("search", &Index::search, py::arg("input") , py::arg("NQ"), py::arg("TOPK") = 1, py::arg("nprobe") = 1, py::arg("num_threads") = -1)
    .def("getMSE", &Index::getMSE);
#endif
    // return m.ptr();
}

