// pybind11 binding for rabitqlib::ivf::IVF

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"

namespace py = pybind11;
using PID = rabitqlib::PID;
using IVF = rabitqlib::ivf::IVF;

class PyIVF {
   private:
    IVF* ivf_ = nullptr;
    size_t n_;
    size_t dim_;
    size_t k_;
    size_t bits_;
    int nthread_;
    rabitqlib::MetricType metric_type_;

   public:
    PyIVF(
        size_t n,
        size_t dim,
        size_t k,
        size_t bits,
        int nthread,
        const std::string& metric = "l2",
        const std::string& rotator = "fht"
    )
        : n_(n), dim_(dim), k_(k), bits_(bits), nthread_(nthread){
        metric_type_ =
            (metric == "ip" || metric == "IP") ? rabitqlib::METRIC_IP : rabitqlib::METRIC_L2;
        rabitqlib::RotatorType rtype = (rotator == "matrix")
            ? rabitqlib::RotatorType::MatrixRotator
            : rabitqlib::RotatorType::FhtKacRotator;
        ivf_ = new IVF(n, dim, k, bits, metric_type_, rtype);
    }

    ~PyIVF() { delete ivf_; }

    void construct(
        py::array_t<float, py::array::c_style | py::array::forcecast> data,
        py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
        py::array_t<uint32_t, py::array::c_style | py::array::forcecast> cluster_ids,
        bool faster = false
    ) {
        auto d = data.request();
        auto c = centroids.request();
        auto ids = cluster_ids.request();

        if (d.ndim != 2) {
            throw std::runtime_error("data must be 2-dimensional");
        }
        if (c.ndim != 2) {
            throw std::runtime_error("centroids must be 2-dimensional");
        }
        if (ids.ndim != 1) {
            throw std::runtime_error("cluster_ids must be 1-dimensional");
        }

        omp_set_num_threads(nthread_);
        ivf_->construct(
            static_cast<const float*>(d.ptr),
            static_cast<const float*>(c.ptr),
            static_cast<const PID*>(ids.ptr),
            faster
        );
    }

    // Search a single query and return top-k result indices
    py::array_t<uint32_t> search(
        py::array_t<float, py::array::c_style | py::array::forcecast> query,
        size_t k,
        size_t nprobe,
        bool use_hacc = true
    ) {
        auto q = query.request();
        if (q.ndim != 1 && !(q.ndim == 2 && q.shape[0] == 1)) {
            throw std::runtime_error("query must be a 1-D array or a (1, dim) 2-D array");
        }

        py::array_t<uint32_t> results(static_cast<py::ssize_t>(k));
        auto r = results.request();

        ivf_->search(
            static_cast<const float*>(q.ptr),
            k,
            nprobe,
            static_cast<PID*>(r.ptr),
            use_hacc
        );
        return results;
    }

    // Search a batch of queries, return (nq, k) index array.
    // Distances are not exposed by IVF::search(); the returned D array is filled with NaN.
    std::pair<py::array_t<int64_t>, py::array_t<float>> search_batch(
        py::array_t<float, py::array::c_style | py::array::forcecast> queries,
        size_t k,
        size_t nprobe,
        bool use_hacc = true
    ) {
#if defined(__AVX512BW__)
        std::cout << "AVX512BW" << std::endl;
#endif
        auto q = queries.request();

        if (q.ndim != 2) {
            throw std::runtime_error("queries must be 2-dimensional");
        }

        size_t nq = static_cast<size_t>(q.shape[0]);
        const float* q_ptr = static_cast<const float*>(q.ptr);
        size_t dim = static_cast<size_t>(q.shape[1]);

        py::array_t<int64_t> I({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
        py::array_t<float> D({static_cast<py::ssize_t>(nq), static_cast<py::ssize_t>(k)});
        auto I_buf = I.request();
        auto D_buf = D.request();

        int64_t* I_ptr = static_cast<int64_t*>(I_buf.ptr);
        float* D_ptr = static_cast<float*>(D_buf.ptr);

        std::fill(D_ptr, D_ptr + nq * k, std::numeric_limits<float>::quiet_NaN());

#pragma omp parallel for num_threads(nthread_)
        for (size_t i = 0; i < nq; ++i) {
            std::vector<PID> tmp(k);
            ivf_->search(q_ptr + i * dim, k, nprobe, tmp.data(), use_hacc);
            for (size_t j = 0; j < k; ++j) {
                I_ptr[i * k + j] = static_cast<int64_t>(tmp[j]);
            }
        }

        return {I, D};
    }


};

PYBIND11_MODULE(rabitqlib_cpp, m) {
    m.doc() = "Python bindings for rabitqlib::ivf::IVF";

    py::class_<PyIVF>(m, "IVF")
        .def(
            py::init<size_t, size_t, size_t, size_t, int, const std::string&, const std::string&>(),
            py::arg("n"),
            py::arg("dim"),
            py::arg("k"),
            py::arg("bits"),
            py::arg("nthread") = 1,
            py::arg("metric") = "l2",
            py::arg("rotator") = "fht"
        )
        .def(
            "construct",
            &PyIVF::construct,
            py::arg("data"),
            py::arg("centroids"),
            py::arg("cluster_ids"),
            py::arg("faster") = false
        )
        .def(
            "search",
            &PyIVF::search,
            py::arg("query"),
            py::arg("k"),
            py::arg("nprobe"),
            py::arg("use_hacc") = true
        )
        .def(
            "search_batch",
            &PyIVF::search_batch,
            py::arg("queries"),
            py::arg("k"),
            py::arg("nprobe"),
            py::arg("use_hacc") = true
        );
}
