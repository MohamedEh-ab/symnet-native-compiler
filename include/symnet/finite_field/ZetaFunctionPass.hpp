// include/symnet/finite_field/ZetaFunctionPass.hpp
//
// Infinite-series collapse of the local zeta function.
//
//   Z(E/F_q, T) = exp( sum_{n>=1} N_n T^n / n ),     N_n = #E(F_{q^n}).
//
// Lefschetz gives N_n = 1 - Tr(F^n) + q^n, with Tr(F^n) = alpha^n + beta^n.
// Because  sum_{n>=1} c^n T^n / n = -log(1 - cT),  the series splits term by term:
//
//   sum N_n T^n / n =  -log(1 - T)  +  log(1 - alpha T) + log(1 - beta T)  -  log(1 - qT)
//
//   =>  Z(T) = (1 - alpha T)(1 - beta T) / ((1 - T)(1 - qT))
//            = det(I - T F) / ((1 - T)(1 - qT)).
//
// The compiler never sums the series.  It reads Tr(F) and det(F) from the
// buffer and emits the quotient of two integer quadratics:
//
//   numerator   = 1 - Tr(F) T + det(F) T^2
//   denominator = (1 - T)(1 - qT) = 1 - (q + 1) T + q T^2
//
// Fixed work, no loops, integer arithmetic only.  A buffer with det(F) != q
// is not the Frobenius of a curve over F_q and is rejected.
#ifndef SYMNET_FINITE_FIELD_ZETA_FUNCTION_PASS_HPP
#define SYMNET_FINITE_FIELD_ZETA_FUNCTION_PASS_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "symnet/finite_field/LefschetzTracePass.hpp"
#include "symnet/Mat2.hpp"

namespace symnet::ff {

// AST node:  Z(T) = num(T) / den(T).
struct ZetaNode {
  Poly2 num;
  Poly2 den;
};

class ZetaFunctionPass {
 public:
  static std::optional<ZetaNode> collapse(const Mat2& F, std::int64_t q) noexcept {
    const Checked<Poly2> cp = LefschetzTracePass::characteristic_polynomial(F);

    std::int64_t qp1, neg_qp1;
    bool o = add_ovf(1, q, qp1);
    o |= sub_ovf(0, qp1, neg_qp1);

    const bool det_is_q = cp.value.c[2] == q;
    if (cp.overflow | o | !det_is_q) return std::nullopt;
    return ZetaNode{cp.value, Poly2{{1, neg_qp1, q}}};
  }

  // Heap-allocated AST form (null on failure).
  static std::unique_ptr<ZetaNode> collapse_ast(const Mat2& F, std::int64_t q) {
    const std::optional<ZetaNode> z = collapse(F, q);
    return z ? std::make_unique<ZetaNode>(*z) : nullptr;
  }
};

// Pretty printers (diagnostics only; not on the compile path).
inline std::string to_string(const Poly2& p) {
  std::ostringstream os;
  os << p.c[0];
  for (int k = 1; k < 3; ++k) {
    os << (p.c[k] < 0 ? " - " : " + ") << magnitude(p.c[k]) << 'T';
    if (k == 2) os << "^2";
  }
  return os.str();
}

inline std::string to_string(const ZetaNode& z) {
  return "(" + to_string(z.num) + ") / (" + to_string(z.den) + ")";
}

}  // namespace symnet::ff

#endif  // SYMNET_FINITE_FIELD_ZETA_FUNCTION_PASS_HPP
