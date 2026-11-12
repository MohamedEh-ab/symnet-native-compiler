// include/symnet/finite_field/WeilProver.hpp
//
// Certificate checkers for the two Weil statements about E/F_q, run on the
// 2x2 Frobenius buffer.  Integer arithmetic only; kernels are branch-free
// (flags are combined with bitwise &, |); only the final report assembly at
// the API boundary uses ordinary control flow.
//
// What "PROVED" means here (please read):
//   Weil proved RH for curves in 1948 and Hasse did genus 1 in 1933; nothing
//   new is proved.  These passes CHECK, exactly and in integers, that a given
//   (F, q) satisfies the statements.  If the trace came from a correct
//   point-count (Schoof) the Hasse check must pass; a failure indicates a
//   wrong trace or a non-Frobenius matrix, not a counterexample to Weil.
//
// ---------------------------------------------------------------------------
// Theorem 1 (Hasse-Weil / RH for E/F_q)
// ---------------------------------------------------------------------------
//   P(T) = 1 - tT + qT^2 = (1 - aT)(1 - bT),   a + b = t,  a b = q.
//   Delta = t^2 - 4q.
//     Delta < 0 : a, b non-real conjugates,  |a|^2 = ab = q.
//     Delta = 0 : a = b = t/2 real, t^2 = 4q, so |a| = sqrt(q).
//     Delta > 0 : a, b real, distinct, ab = q  =>  one has |.| > sqrt(q).
//   So  Delta <= 0  <=>  |a| = |b| = sqrt(q)  <=>  the zeros of Z(T) lie on
//   |T| = q^{-1/2}  <=>  |t| <= 2 sqrt(q).
//   t^2 and 4q are compared as 128-bit integers (no overflow, no sqrt).
//   The determinant check det(F) == q is part of the verdict: the inequality
//   says nothing about a matrix that is not a Frobenius.
//
// ---------------------------------------------------------------------------
// Theorem 2 (functional equation) -- CORRECTED statement
// ---------------------------------------------------------------------------
//   For a curve of genus g:  Z(1/(qT)) = q^{1-g} T^{2-2g} Z(T).
//   For an elliptic curve (g = 1) the factor is 1:
//
//                       Z(E, 1/(qT)) = Z(E, T).
//
//   The relation "Z(1/(qT)) = q T^2 Z(T)" is NOT true for Z (it would force
//   qT^2 = 1 wherever Z(T) != 0).  The factor qT^2 belongs to the NUMERATOR
//   and DENOMINATOR separately:
//
//       q T^2 * P(1/(qT)) = P(T)          q T^2 * D(1/(qT)) = D(T),
//       D(T) = (1 - T)(1 - qT),
//
//   whose quotient is Z(1/(qT)) = Z(T).  With p(T) = c0 + c1 T + c2 T^2 the
//   reflected polynomial is  R[p](T) = q T^2 p(1/(qT)) = (c2/q) + c1 T + (q c0) T^2,
//   so R[p] = p  <=>  c2 = q c0.  For P and D this reads  det(F) = q.
//
//   Where it comes from (Poincare duality): the cup product pairing is
//   preserved by Frobenius up to the factor q, i.e.  F^T J F = q J  with
//   J = [[0,1],[-1,0]].  Hence q F^{-1} = adj(F) is conjugate to F, which is
//   what makes P self-reflective.
//
//   Checks performed:
//     (a) Mechanism-B style cancellation: F * F^{-1} collapses to the identity
//         tensor I_2.  F^{-1} is carried as the integral pair (adj(F), det F).
//         Honest remark: this step alone is a tautology for any invertible
//         matrix (it certifies invertibility and that q F^{-1} = adj F is the
//         right integral modifier); the arithmetic content of Theorem 2 is
//         in (b) and (c).
//     (b) Weil pairing:  F^T J F == q J.
//     (c) Reflection identities for numerator and denominator (c2 == q c0).
#ifndef SYMNET_FINITE_FIELD_WEIL_PROVER_HPP
#define SYMNET_FINITE_FIELD_WEIL_PROVER_HPP

#include <cstdint>

#include "symnet/finite_field/FrobeniusGenerator.hpp"
#include "symnet/finite_field/LefschetzTracePass.hpp"
#include "symnet/Mat2.hpp"
#include "symnet/finite_field/ZetaFunctionPass.hpp"

namespace symnet::ff {

enum class Verdict : std::uint8_t { REFUTED = 0, PROVED = 1 };

inline const char* to_string(Verdict v) noexcept { return v == Verdict::PROVED ? "PROVED" : "REFUTED"; }

struct HasseWeilReport {
  Verdict verdict;
  std::int64_t trace;
  bool det_is_q;
  bool delta_negative;  // t^2 <  4q
  bool delta_zero;      // t^2 == 4q  (supersingular, q a square)
  bool overflow;
};

struct FunctionalEquationReport {
  Verdict verdict;
  bool inverse_collapse;      // F * F^{-1} == I_2
  bool weil_pairing;          // F^T J F == q J
  bool numerator_reflects;    // q T^2 P(1/(qT)) == P(T)
  bool denominator_reflects;  // q T^2 D(1/(qT)) == D(T)
  bool overflow;
};

struct WeilCertificate {
  HasseWeilReport hasse;
  FunctionalEquationReport functional_equation;
};

class WeilProver {
 public:
  // Theorem 1.
  static HasseWeilReport prove_hasse_weil(const Mat2& F, std::int64_t q) noexcept {
    const std::int64_t t = LefschetzTracePass::trace(F);
    const Checked<std::int64_t> d = det(F);

    const bool q_ok = (q > 3) & (q <= EllipticCurveNode::kMaxModulus);
    const bool det_ok = (d.value == q) & !d.overflow;

    const U128 t2 = mul64(magnitude(t), magnitude(t));
    const std::uint64_t uq = static_cast<std::uint64_t>(q);
    const U128 four_q{uq >> 62, uq << 2};

    const bool lt = !gt(t2, four_q) & !eq(t2, four_q);
    const bool zero = eq(t2, four_q);
    const bool holds = lt | zero;                          // Delta <= 0
    const bool ok = holds & det_ok & q_ok;

    return HasseWeilReport{static_cast<Verdict>(ok), t, det_ok, lt, zero, d.overflow};
  }

  // Theorem 2 (g = 1 functional equation Z(1/(qT)) = Z(T)).
  static FunctionalEquationReport prove_functional_equation(const Mat2& F, std::int64_t q) noexcept {
    bool o = false;

    // (a) F * F^{-1} -> I_2, with F^{-1} = adj(F) / det(F).
    const Checked<std::int64_t> d = det(F);
    const Checked<Mat2> adj = adjugate(F);
    o |= d.overflow | adj.overflow;
    const bool collapse =
        collapses_to_identity(ScaledMat2{F, 1}, ScaledMat2{adj.value, d.value}, o);

    // (b) Weil pairing  F^T J F == q J.
    std::int64_t nq;
    o |= sub_ovf(0, q, nq);
    const Mat2 J{{0, 1, -1, 0}};
    const Mat2 qJ{{0, q, nq, 0}};
    const Checked<Mat2> tj = mul(transpose(F), J);
    const Checked<Mat2> tjf = mul(tj.value, F);
    o |= tj.overflow | tjf.overflow;
    const bool pairing = equal(tjf.value, qJ);

    // (c) Reflection identities on the collapsed closed form.
    const std::optional<ZetaNode> z = ZetaFunctionPass::collapse(F, q);
    const Poly2 zero_poly{{0, 0, 1}};   // c2 != q*c0 for any q > 0: reflects == false if absent
    const ZetaNode zn = z.value_or(ZetaNode{zero_poly, zero_poly});
    const bool have = z.has_value();
    const bool num_r = reflects(zn.num, q, o) & have;
    const bool den_r = reflects(zn.den, q, o) & have;

    const bool ok = collapse & pairing & num_r & den_r & !o;
    return FunctionalEquationReport{static_cast<Verdict>(ok), collapse, pairing, num_r, den_r, o};
  }

  static WeilCertificate certify(const FrobeniusNode& n) noexcept {
    return WeilCertificate{prove_hasse_weil(n.F, n.q), prove_functional_equation(n.F, n.q)};
  }

 private:
  // A * B == I  with A = a.num/a.den, B = b.num/b.den:  a.num*b.num == (a.den*b.den) I.
  static bool collapses_to_identity(const ScaledMat2& a, const ScaledMat2& b, bool& o) noexcept {
    const Checked<Mat2> p = mul(a.num, b.num);
    std::int64_t den;
    o |= mul_ovf(a.den, b.den, den);
    o |= p.overflow;
    const Mat2 target{{den, 0, 0, den}};
    return equal(p.value, target) & (den != 0);
  }

  // R[p] = p  <=>  c2 == q * c0.
  static bool reflects(const Poly2& p, std::int64_t q, bool& o) noexcept {
    std::int64_t qc0;
    o |= mul_ovf(q, p.c[0], qc0);
    return p.c[2] == qc0;
  }
};

}  // namespace symnet::ff

#endif  // SYMNET_FINITE_FIELD_WEIL_PROVER_HPP
