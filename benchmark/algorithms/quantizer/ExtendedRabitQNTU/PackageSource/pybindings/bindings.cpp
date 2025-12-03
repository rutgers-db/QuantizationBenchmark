#define HIGH_ACC_FAST_SCAN
#define EIGEN_DONT_PARALLELIZE
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

    IVF* ivf;
    uint32_t dim;
    uint32_t Nlist = 1;
    uint32_t bit;

    float * data;
    size_t ndata;

    public:
        Index(uint32_t D, uint32_t B):dim(D),bit(B)
        {}
        ~Index(){delete ivf;}
        py::object train(py::object input, py::object centroids, py::object cids, size_t N, int num_threads = -1){
            py::array_t < float, py::array::c_style | py::array::forcecast > data_items(input);
            py::array_t < float, py::array::c_style | py::array::forcecast > centroids_items(centroids);
            py::array_t < uint32_t, py::array::c_style | py::array::forcecast > cids_items(cids);
            ndata = N;

            float* data_ptr = static_cast<float*> (data_items.request().ptr);
            float* centroids_ptr = static_cast<float*> (centroids_items.request().ptr);
            uint32_t* cids_ptr = static_cast<uint32_t*> (cids_items.request().ptr);
            data = data_ptr;

            ivf = new IVF(N, dim, Nlist, bit);
            ivf->construct(data_ptr, centroids_ptr, cids_ptr);
        }
        py::object search(py::object input, size_t NQ, size_t TOPK = 1, int num_threads = -1){
            py::array_t < float, py::array::c_style | py::array::forcecast > items(input);
            float* raw = static_cast<float*> (items.request().ptr);

            Eigen::Map<FloatRowMat> padded_query(raw, NQ, ivf->padded_dim());

            uint32_t * ret_list = new uint32_t[NQ * TOPK]; 
            float* dist_list = new float[NQ * TOPK];
            Rotator& rp = ivf->rotator();
            FloatRowMat rotated_query(NQ, ivf->padded_dim());
            rp.rotate(padded_query, rotated_query);
#pragma omp parallel for if(num_threads > 1)
            for (size_t i = 0; i < NQ; i++) {
                ivf->search(&rotated_query(i, 0), raw, TOPK, 1, ret_list + i * TOPK, dist_list + i * TOPK);
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

        float getMSE(){
            nlist =  ivf -> ClusterLst.size();
            return 
        }



};


PYBIND11_PLUGIN(ExtendedRabitQ) {
    py::module m("ExtendedRabitQ");

    py::class_<Index>(m, "Index")
    .def(py::init<uint32_t, uint32_t>(), py::arg("D"), py::arg("B"))
    .def("train", &Index::train, py::arg("input"), py::arg("centroids"), py::arg("cids"), py::arg("N"), py::arg("num_threads")  = -1)
    .def("search", &Index::search, py::arg("input") , py::arg("NQ"), py::arg("TOPK") = 1, py::arg("num_threads") = -1)
    .def("getMSE", &Index::getMSE);
    return m.ptr();
}