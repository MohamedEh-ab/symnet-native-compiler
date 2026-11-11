// bindings.cpp - Python extension module `symnet_native` (pybind11).
//
//   symnet_native.Matrix2x2 / Matrix3x3 / Matrix4x4   stack-allocated, 32-byte aligned matrices
//   symnet_native.eval_word_compiled(k, factors, weight, bias) -> bool
//   symnet_native.CompiledWord(k, factors)             fold once, decide many times
//   symnet_native.gemm_count() / reset_gemm_count()    instrumentation (matmul passes executed)
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <symnet/MatrixEngine.hpp>
#include <symnet/WordEval.hpp>

#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace symnet;

namespace {

template <int N>
void bindMatrix(py::module_& m, const char* name) {
    using M = Mat<N>;
    const std::string cname(name);
    py::class_<M>(m, name, "Fixed-size row-major matrix stored on the stack (32-byte aligned).")
        .def(py::init([]() { return M::identity(); }), "Identity matrix.")
        .def_static("identity", &M::identity)
        .def_static("zero", &M::zero)
        .def_static(
            "from_list",
            [](const std::vector<double>& flat) {
                if (flat.size() != static_cast<std::size_t>(N * N))
                    throw py::value_error("expected " + std::to_string(N * N) + " row-major entries");
                return M::fromData(flat.data());
            },
            py::arg("flat"), "Build from N*N row-major entries.")
        .def_property_readonly_static("dim", [](const py::object&) { return N; })
        .def("to_list", [](const M& a) { return std::vector<double>(a.a, a.a + N * N); },
             "Row-major list of N*N floats.")
        .def("__getitem__",
             [](const M& a, const std::pair<int, int>& ij) {
                 if (ij.first < 0 || ij.first >= N || ij.second < 0 || ij.second >= N)
                     throw py::index_error("index out of range");
                 return a(ij.first, ij.second);
             })
        .def("__setitem__",
             [](M& a, const std::pair<int, int>& ij, double v) {
                 if (ij.first < 0 || ij.first >= N || ij.second < 0 || ij.second >= N)
                     throw py::index_error("index out of range");
                 a(ij.first, ij.second) = v;
             })
        .def("__mul__", [](const M& a, const M& b) { return a * b; }, py::is_operator(), "Matrix product.")
        .def("__matmul__", [](const M& a, const M& b) { return a * b; }, py::is_operator(), "Matrix product.")
        .def(
            "apply",
            [](const M& a, const std::vector<double>& x) {
                if (x.size() != static_cast<std::size_t>(N))
                    throw py::value_error("expected a vector of length " + std::to_string(N));
                alignas(64) double in[N];
                alignas(64) double out[N];
                for (int i = 0; i < N; ++i) in[i] = x[static_cast<std::size_t>(i)];
                gemvFixed<N>(a.a, in, out);
                return std::vector<double>(out, out + N);
            },
            py::arg("x"), "Matrix-vector product.")
        .def("max_abs_diff", &M::maxAbsDiff, py::arg("other"))
        .def("__eq__", [](const M& a, const M& b) { return a.maxAbsDiff(b) == 0.0; }, py::is_operator())
        .def("__repr__", [cname](const M& a) { return cname + "(" + fmtMat(a.a, N, N) + ")"; });
}

}  // namespace

PYBIND11_MODULE(symnet_native, m) {
    m.doc() = "SymNetNative: constant-folded group-word evaluation with fixed-size SIMD-friendly matrices";

    bindMatrix<2>(m, "Matrix2x2");
    bindMatrix<3>(m, "Matrix3x3");
    bindMatrix<4>(m, "Matrix4x4");

    m.def(
        "eval_word_compiled",
        [](int k, const std::vector<double>& factors, const std::vector<double>& weight, double bias) {
            try {
                return evalWordCompiled(k, factors, weight, bias);
            } catch (const std::invalid_argument& e) {
                throw py::value_error(e.what());
            }
        },
        py::arg("k"), py::arg("factors"), py::arg("weight"), py::arg("bias"),
        R"doc(Decision Step(w . vec(W_compiled) + b) for a group word, as a bool.

k       generator dimension (2, 3 or 4)
factors L row-major k x k matrices concatenated in word order (empty list = identity)
weight  k*k hyperplane weights over vec(W_compiled), row-major
bias    hyperplane bias

Inputs are copied into alignas(64) stack arrays and the word is folded right to left.
For repeated decisions on one word use CompiledWord, which folds only once.)doc");

    py::class_<CompiledWord>(m, "CompiledWord", "A word folded once into a single k x k matrix.")
        .def(py::init([](int k, const std::vector<double>& factors) {
                 try {
                     return CompiledWord(k, factors);
                 } catch (const std::invalid_argument& e) {
                     throw py::value_error(e.what());
                 }
             }),
             py::arg("k"), py::arg("factors"))
        .def_property_readonly("dim", &CompiledWord::dim)
        .def_property_readonly("length", &CompiledWord::length, "Number of factors that were folded.")
        .def_property_readonly("matrix", &CompiledWord::matrix, "W_compiled as a row-major list.")
        .def(
            "decide",
            [](const CompiledWord& w, const std::vector<double>& weight, double bias) {
                try {
                    return w.decide(weight, bias);
                } catch (const std::invalid_argument& e) {
                    throw py::value_error(e.what());
                }
            },
            py::arg("weight"), py::arg("bias"));

    m.def("gemm_count", []() { return gemmCounter(); }, "Matrix-multiplication passes executed so far.");
    m.def("reset_gemm_count", []() { gemmCounter() = 0; });
    m.attr("__version__") = "1.0.0";
}
