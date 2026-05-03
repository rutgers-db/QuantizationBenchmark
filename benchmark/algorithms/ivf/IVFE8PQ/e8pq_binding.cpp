// pybind11 binding for IVFE8PQ, IVFE8PQFastScan and IVFE8PQFastScan4. All three
// classes share the same training pipeline (coarse KMeans via faiss::Clustering,
// PQ training via faiss::ProductQuantizer on the normalized residuals) and the
// same RaBitQ scoring, but differ in the search-time data layout:
//   * IVFE8PQ          (nbit=8): 16-wide tiles, float gather LUT.
//   * IVFE8PQFastScan  (nbit=8): 32-wide tiles, register-resident uint8 LUT
//                                (AVX512VBMI permutex2var; 32-wide gather fallback).
//   * IVFE8PQFastScan4 (nbit=4): 32-wide tiles, nibble-paired packed codes,
//                                pshufb-based 16-entry LUT lookup (PQ4 fastscan).

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

#include "e8pq.hpp"
#include "e8pq_fastscan.hpp"
#include "e8pq_fastscan4.hpp"

namespace py = pybind11;
using e8pqlib::IVFE8PQ;
using e8pqlib::IVFE8PQFastScan;
using e8pqlib::IVFE8PQFastScan4;

class IVFE8PQWrapper {
public:
    IVFE8PQWrapper(size_t n, size_t dim, size_t nlist, size_t nsubvec,
                   size_t nbit, int nthread, const std::string& metric,
                   const std::string& rotator)
        : impl_(n, dim, nlist, nsubvec, nbit, nthread, metric, rotator) {}

    void fit(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        auto d = data.request();
        if (d.ndim != 2) throw std::runtime_error("data must be 2-D");
        size_t nb = static_cast<size_t>(d.shape[0]);
        const float* dptr = static_cast<const float*>(d.ptr);
        py::gil_scoped_release nogil;
        impl_.fit(dptr, nb);
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

        {
            py::gil_scoped_release nogil;
            impl_.search_batch(qptr, nq, k, nprobe, Iptr, Dptr);
        }
        return {I, D};
    }

private:
    IVFE8PQ impl_;
};

class IVFE8PQFastScanWrapper {
public:
    IVFE8PQFastScanWrapper(size_t n, size_t dim, size_t nlist, size_t nsubvec,
                           size_t nbit, int nthread, const std::string& metric,
                           const std::string& rotator)
        : impl_(n, dim, nlist, nsubvec, nbit, nthread, metric, rotator) {}

    void fit(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        auto d = data.request();
        if (d.ndim != 2) throw std::runtime_error("data must be 2-D");
        size_t nb = static_cast<size_t>(d.shape[0]);
        const float* dptr = static_cast<const float*>(d.ptr);
        py::gil_scoped_release nogil;
        impl_.fit(dptr, nb);
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

        {
            py::gil_scoped_release nogil;
            impl_.search_batch(qptr, nq, k, nprobe, Iptr, Dptr);
        }
        return {I, D};
    }

private:
    IVFE8PQFastScan impl_;
};

class IVFE8PQFastScan4Wrapper {
public:
    IVFE8PQFastScan4Wrapper(size_t n, size_t dim, size_t nlist, size_t nsubvec,
                            size_t nbit, int nthread, const std::string& metric,
                            const std::string& rotator)
        : impl_(n, dim, nlist, nsubvec, nbit, nthread, metric, rotator) {}

    void fit(py::array_t<float, py::array::c_style | py::array::forcecast> data) {
        auto d = data.request();
        if (d.ndim != 2) throw std::runtime_error("data must be 2-D");
        size_t nb = static_cast<size_t>(d.shape[0]);
        const float* dptr = static_cast<const float*>(d.ptr);
        py::gil_scoped_release nogil;
        impl_.fit(dptr, nb);
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

        {
            py::gil_scoped_release nogil;
            impl_.search_batch(qptr, nq, k, nprobe, Iptr, Dptr);
        }
        return {I, D};
    }

private:
    IVFE8PQFastScan4 impl_;
};

PYBIND11_MODULE(e8pq_cpp, m) {
    m.doc() = "IVF + learned-PQ codebook on normalized residuals "
              "(RaBitQ-style scoring). Provides IVFE8PQ (gather LUT, nbit=8), "
              "IVFE8PQFastScan (register-resident uint8 LUT, AVX512VBMI, "
              "nbit=8) and IVFE8PQFastScan4 (PQ4 fastscan, pshufb, nbit=4).";

    py::class_<IVFE8PQWrapper>(m, "IVFE8PQ")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nsubvec"),
             py::arg("nbit") = 8,
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("fit", &IVFE8PQWrapper::fit, py::arg("data"))
        .def("search_batch", &IVFE8PQWrapper::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"));

    py::class_<IVFE8PQFastScanWrapper>(m, "IVFE8PQFastScan")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nsubvec"),
             py::arg("nbit") = 8,
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("fit", &IVFE8PQFastScanWrapper::fit, py::arg("data"))
        .def("search_batch", &IVFE8PQFastScanWrapper::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"));

    py::class_<IVFE8PQFastScan4Wrapper>(m, "IVFE8PQFastScan4")
        .def(py::init<size_t, size_t, size_t, size_t, size_t, int,
                      const std::string&, const std::string&>(),
             py::arg("n"),
             py::arg("dim"),
             py::arg("nlist"),
             py::arg("nsubvec"),
             py::arg("nbit") = 4,
             py::arg("nthread") = 1,
             py::arg("metric") = "l2",
             py::arg("rotator") = "fht")
        .def("fit", &IVFE8PQFastScan4Wrapper::fit, py::arg("data"))
        .def("search_batch", &IVFE8PQFastScan4Wrapper::search_batch,
             py::arg("queries"), py::arg("k"), py::arg("nprobe"));
}
