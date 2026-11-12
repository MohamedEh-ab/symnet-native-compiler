// bind_finite_field.cpp - Python bindings for the exact finite-field passes (symnet::ff).
//
//   symnet_native.finite_field.EllipticCurve(a, b, q)     y^2 = x^3 + a x + b over F_q
//   .with_trace(t) / .resolve(callable) / .valid() / .trace
//   finite_field.lift(curve) -> FrobeniusNode | None      F = [[0, -q], [1, a_q]]
//   finite_field.Mat2                                     exact int64 2x2 matrix
//   finite_field.trace / determinant / characteristic_polynomial / power /
//                  trace_of_power / count_points          Lefschetz passes
//   finite_field.zeta_collapse(F, q) -> ZetaNode | None   Z(T) = num / den
//   finite_field.prove_hasse_weil / prove_functional_equation / certify
//
// Integer results that cannot be certified inside int64 are never wrapped: functions that
// return a matrix/scalar raise OverflowError; those that return Optional return None.
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <symnet/Api.hpp>

#include "bindings.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace api = symnet::api;
namespace ff = symnet::ff;
using i64 = std::int64_t;

namespace {

std::string repr(const ff::Mat2& a) {
    return "Mat2(" + std::to_string(a.m[0]) + ", " + std::to_string(a.m[1]) + ", " + std::to_string(a.m[2]) +
           ", " + std::to_string(a.m[3]) + ")";
}

}  // namespace

void bind_finite_field(py::module_& root) {
    py::module_ m = make_submodule(
        root, "finite_field",
        "Exact integer Frobenius / Lefschetz / zeta / Weil passes for elliptic curves over F_q (no floating point).");

    // ---- Verdict ------------------------------------------------------------------------
    py::enum_<ff::Verdict>(m, "Verdict")
        .value("REFUTED", ff::Verdict::REFUTED)
        .value("PROVED", ff::Verdict::PROVED)
        .def("__bool__", [](ff::Verdict v) { return v == ff::Verdict::PROVED; });

    // ---- Poly2 --------------------------------------------------------------------------
    py::class_<ff::Poly2>(m, "Poly2", "c0 + c1 T + c2 T^2 with exact integer coefficients.")
        .def(py::init([](i64 c0, i64 c1, i64 c2) { return ff::Poly2{{c0, c1, c2}}; }), py::arg("c0"), py::arg("c1"),
             py::arg("c2"))
        .def_property_readonly("c0", [](const ff::Poly2& p) { return p.c[0]; })
        .def_property_readonly("c1", [](const ff::Poly2& p) { return p.c[1]; })
        .def_property_readonly("c2", [](const ff::Poly2& p) { return p.c[2]; })
        .def_property_readonly("coefficients", [](const ff::Poly2& p) { return std::vector<i64>{p.c[0], p.c[1], p.c[2]}; })
        .def("__eq__", [](const ff::Poly2& a, const ff::Poly2& b) { return a == b; }, py::is_operator())
        .def("__str__", [](const ff::Poly2& p) { return ff::to_string(p); })
        .def("__repr__", [](const ff::Poly2& p) { return "Poly2(" + ff::to_string(p) + ")"; });

    // ---- Mat2 ---------------------------------------------------------------------------
    py::class_<ff::Mat2>(m, "Mat2", "Exact int64 2x2 matrix, row-major (m00, m01, m10, m11).")
        .def(py::init([]() { return ff::Mat2::identity(); }), "Identity matrix.")
        .def(py::init([](i64 a, i64 b, i64 c, i64 d) { return ff::Mat2{{a, b, c, d}}; }), py::arg("m00"),
             py::arg("m01"), py::arg("m10"), py::arg("m11"))
        .def_static("identity", []() { return ff::Mat2::identity(); })
        .def_static(
            "from_list",
            [](const std::vector<i64>& v) {
                if (v.size() != 4) throw py::value_error("expected 4 row-major entries");
                return ff::Mat2{{v[0], v[1], v[2], v[3]}};
            },
            py::arg("flat"))
        .def("to_list", [](const ff::Mat2& a) { return std::vector<i64>{a.m[0], a.m[1], a.m[2], a.m[3]}; })
        .def("__getitem__",
             [](const ff::Mat2& a, const std::pair<int, int>& ij) {
                 if (ij.first < 0 || ij.first > 1 || ij.second < 0 || ij.second > 1)
                     throw py::index_error("index out of range");
                 return a.m[ij.first * 2 + ij.second];
             })
        .def("__mul__", [](const ff::Mat2& a, const ff::Mat2& b) { return api::ffMul(a, b); }, py::is_operator(),
             "Matrix product; OverflowError if it does not fit in int64.")
        .def("__matmul__", [](const ff::Mat2& a, const ff::Mat2& b) { return api::ffMul(a, b); }, py::is_operator())
        .def("__sub__", [](const ff::Mat2& a, const ff::Mat2& b) { return api::ffSub(a, b); }, py::is_operator())
        .def("mul_checked",
             [](const ff::Mat2& a, const ff::Mat2& b) {
                 const auto r = ff::mul(a, b);
                 return std::make_pair(r.value, r.overflow);
             },
             py::arg("other"), "Product as (matrix, overflowed); the matrix is wrapped when overflowed is True.")
        .def("det", [](const ff::Mat2& a) { return api::ffDet(a); })
        .def("adjugate", [](const ff::Mat2& a) { return api::ffAdjugate(a); })
        .def("transpose", [](const ff::Mat2& a) { return ff::transpose(a); })
        .def("to_engine", [](const ff::Mat2& a) { return ff::toEngine(a); },
             "As symnet_native.Matrix2x2, or None if any |entry| > 2**53.")
        .def_static("from_engine", [](const symnet::Matrix2x2& e) { return ff::fromEngine(e); }, py::arg("matrix"),
                    "From symnet_native.Matrix2x2, or None unless every entry is an integer with |entry| <= 2**53.")
        .def("__eq__", [](const ff::Mat2& a, const ff::Mat2& b) { return ff::equal(a, b); }, py::is_operator())
        .def("__repr__", [](const ff::Mat2& a) { return repr(a); });

    // ---- curve and Frobenius lift ----------------------------------------------------------
    py::class_<ff::EllipticCurveNode>(m, "EllipticCurve", "y^2 = x^3 + a x + b over F_q (q an odd prime, 3 < q <= 2**62).")
        .def(py::init<i64, i64, i64>(), py::arg("a"), py::arg("b"), py::arg("q"))
        .def_property_readonly("a", [](const ff::EllipticCurveNode& e) { return e.a(); })
        .def_property_readonly("b", [](const ff::EllipticCurveNode& e) { return e.b(); })
        .def_property_readonly("q", [](const ff::EllipticCurveNode& e) { return e.q(); })
        .def_property_readonly("trace", [](const ff::EllipticCurveNode& e) { return e.trace(); },
                               "Frobenius trace a_q, or None if unresolved.")
        .def("valid", [](const ff::EllipticCurveNode& e) { return e.valid(); },
             "3 < q <= 2**62, q odd, q % 3 != 0 and the curve is nonsingular mod q (primality is NOT tested).")
        .def("with_trace", [](const ff::EllipticCurveNode& e, i64 t) { return e.with_trace(t); }, py::arg("a_q"),
             "Copy of the curve with a known trace attached.")
        .def("resolve", [](const ff::EllipticCurveNode& e, const ff::TraceResolver& r) { return e.resolve(r); },
             py::arg("resolver"),
             "Copy with the trace obtained from resolver(curve) -> int | None (Schoof/SEA plugs in here).")
        .def("__repr__", [](const ff::EllipticCurveNode& e) {
            return "EllipticCurve(a=" + std::to_string(e.a()) + ", b=" + std::to_string(e.b()) +
                   ", q=" + std::to_string(e.q()) + ")";
        });

    m.def("schoof_trace_stub", [](const ff::EllipticCurveNode&) { return std::optional<i64>{}; }, py::arg("curve"),
          "Explicit stub resolver: always returns None (no point counting is implemented).");

    py::class_<ff::FrobeniusNode>(m, "FrobeniusNode", "A lifted Frobenius generator: the 2x2 buffer F and the field size q.")
        .def(py::init([](const ff::Mat2& F, i64 q) { return ff::FrobeniusNode{F, q}; }), py::arg("F"), py::arg("q"))
        .def_readonly("F", &ff::FrobeniusNode::F)
        .def_readonly("q", &ff::FrobeniusNode::q)
        .def("__repr__", [](const ff::FrobeniusNode& n) { return "FrobeniusNode(F=" + repr(n.F) + ", q=" + std::to_string(n.q) + ")"; });

    m.def("lift", [](const ff::EllipticCurveNode& e) { return ff::FrobeniusGenerator::lift(e); }, py::arg("curve"),
          "Curve -> FrobeniusNode with F = [[0, -q], [1, a_q]]; None if the curve is invalid or its trace is unresolved.");
    m.def("lift_to_engine", [](const ff::EllipticCurveNode& e) { return ff::FrobeniusGenerator::lift_to_engine(e); },
          py::arg("curve"), "Same lift as symnet_native.Matrix2x2; None if an entry exceeds 2**53.");

    // ---- Lefschetz trace pass ---------------------------------------------------------------
    m.def("trace", [](const ff::Mat2& F) { return ff::LefschetzTracePass::trace(F); }, py::arg("F"), "Tr(F).");
    m.def("determinant", [](const ff::Mat2& F) { return api::ffDet(F); }, py::arg("F"), "det(F); OverflowError if not exact.");
    m.def("characteristic_polynomial", [](const ff::Mat2& F) { return api::ffCharPoly(F); }, py::arg("F"),
          "P(T) = det(I - T F) = 1 - Tr(F) T + det(F) T^2 as a Poly2.");
    m.def("power", [](const ff::Mat2& F, std::uint64_t n) { return api::ffPower(F, n); }, py::arg("F"), py::arg("n"),
          "F^n by binary exponentiation (at most 64 steps); OverflowError if a needed intermediate leaves int64.");
    m.def("trace_of_power", [](const ff::Mat2& F, std::uint64_t n) { return ff::LefschetzTracePass::trace_of_power(F, n); },
          py::arg("F"), py::arg("n"), "Tr(F^n) = alpha^n + beta^n, or None on overflow.");
    m.def("count_points", [](const ff::Mat2& F, std::uint64_t n) { return ff::LefschetzTracePass::count_points(F, n); },
          py::arg("F"), py::arg("n"),
          "#E(F_{q^n}) = det(I - F^n) (Lefschetz); None for n == 0 or when not certifiable inside int64.");

    // ---- zeta function ----------------------------------------------------------------------
    py::class_<ff::ZetaNode>(m, "ZetaNode", "Z(T) = num(T) / den(T).")
        .def_property_readonly("num", [](const ff::ZetaNode& z) { return z.num; }, "1 - tT + qT^2")
        .def_property_readonly("den", [](const ff::ZetaNode& z) { return z.den; }, "(1 - T)(1 - qT)")
        .def("__str__", [](const ff::ZetaNode& z) { return ff::to_string(z); })
        .def("__repr__", [](const ff::ZetaNode& z) { return "ZetaNode(" + ff::to_string(z) + ")"; });

    m.def("zeta_collapse", [](const ff::Mat2& F, i64 q) { return ff::ZetaFunctionPass::collapse(F, q); }, py::arg("F"),
          py::arg("q"), "Closed form of the zeta function; None if det(F) != q or on overflow.");

    // ---- Weil prover --------------------------------------------------------------------------
    py::class_<ff::HasseWeilReport>(m, "HasseWeilReport")
        .def_readonly("verdict", &ff::HasseWeilReport::verdict)
        .def_readonly("trace", &ff::HasseWeilReport::trace)
        .def_readonly("det_is_q", &ff::HasseWeilReport::det_is_q)
        .def_readonly("delta_negative", &ff::HasseWeilReport::delta_negative, "t^2 < 4q")
        .def_readonly("delta_zero", &ff::HasseWeilReport::delta_zero, "t^2 == 4q")
        .def_readonly("overflow", &ff::HasseWeilReport::overflow);

    py::class_<ff::FunctionalEquationReport>(m, "FunctionalEquationReport")
        .def_readonly("verdict", &ff::FunctionalEquationReport::verdict)
        .def_readonly("inverse_collapse", &ff::FunctionalEquationReport::inverse_collapse, "F * F^{-1} == I")
        .def_readonly("weil_pairing", &ff::FunctionalEquationReport::weil_pairing, "F^T J F == q J")
        .def_readonly("numerator_reflects", &ff::FunctionalEquationReport::numerator_reflects)
        .def_readonly("denominator_reflects", &ff::FunctionalEquationReport::denominator_reflects)
        .def_readonly("overflow", &ff::FunctionalEquationReport::overflow);

    py::class_<ff::WeilCertificate>(m, "WeilCertificate")
        .def_readonly("hasse", &ff::WeilCertificate::hasse)
        .def_readonly("functional_equation", &ff::WeilCertificate::functional_equation);

    m.def("prove_hasse_weil", [](const ff::Mat2& F, i64 q) { return ff::WeilProver::prove_hasse_weil(F, q); },
          py::arg("F"), py::arg("q"), "Check det F == q and t^2 <= 4q exactly (128-bit comparison).");
    m.def("prove_functional_equation",
          [](const ff::Mat2& F, i64 q) { return ff::WeilProver::prove_functional_equation(F, q); }, py::arg("F"),
          py::arg("q"), "Check Z(1/(qT)) = Z(T): F F^-1 -> I, F^T J F = q J, and reflection of numerator/denominator.");
    m.def("certify", [](const ff::FrobeniusNode& n) { return ff::WeilProver::certify(n); }, py::arg("node"),
          "Both Weil checks for a FrobeniusNode.");
}
