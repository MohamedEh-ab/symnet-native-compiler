// include/symnet/finite_field/FrobeniusGenerator.hpp
//
// AST node for  E : y^2 = x^3 + a x + b  over F_q  and its lift to a 2x2
// Frobenius generator.
//
// ---------------------------------------------------------------------------
// Étale-cohomology dictionary (what the 2x2 buffer means)
// ---------------------------------------------------------------------------
// For a prime l != char(F_q), H^1_et(E_{Fbar_q}, Q_l) is 2-dimensional (it is
// the dual of the Tate module V_l(E)).  The q-power Frobenius pi_q acts on it
// by a 2x2 matrix over Q_l.  Facts used by the rest of this directory:
//
//   * Lefschetz fixed-point formula:   #E(F_{q^n}) = det(I - F^n)
//                                              = 1 - Tr(F^n) + q^n
//     (H^0 contributes 1, H^2 = Q_l(-1) contributes q^n, H^1 contributes -Tr).
//   * det(F) = q  (Frobenius multiplies the cup-product pairing
//     H^1 x H^1 -> H^2 = Q_l(-1) by q: the Weil pairing / Poincare duality).
//   * Tr(F) = a_q = q + 1 - #E(F_q) is an INTEGER independent of l.
//
// What this header does NOT claim:
//   * The actual Frobenius matrix is only defined up to GL_2(Q_l)-conjugacy.
//     We emit the companion matrix of x^2 - a_q x + q,
//
//                      F_q = [ 0  -q  ]
//                            [ 1  a_q ],
//
//     one representative of that class.  Its ENTRIES carry no meaning; its
//     conjugation-invariants (trace, determinant, characteristic polynomial,
//     traces of powers, hence the zeta function) are exactly the right ones.
//   * Computing a_q is NOT O(1).  Schoof/SEA is polynomial in log q.  The AST
//     therefore carries an optional, externally resolved trace (see
//     TraceResolver).  `schoof_trace_stub` is an explicit stub that returns
//     nullopt; plug a real implementation in.  There is no point-counting
//     loop in this directory.
//   * Preconditions: q is an odd PRIME, 3 < q <= 2^62.  Primality is the
//     caller's guarantee (it is not re-tested here); a, b are reduced mod q.
#ifndef SYMNET_FINITE_FIELD_FROBENIUS_GENERATOR_HPP
#define SYMNET_FINITE_FIELD_FROBENIUS_GENERATOR_HPP

#include <cstdint>
#include <functional>
#include <optional>
#include <variant>

#include "symnet/Mat2.hpp"

namespace symnet::ff {

namespace detail {

// (a + b) mod q for a, b < q < 2^62; branch-free.
inline std::uint64_t addmod(std::uint64_t a, std::uint64_t b, std::uint64_t q) noexcept {
  const std::uint64_t s = a + b;
  return s - (q & (0 - static_cast<std::uint64_t>(s >= q)));
}

// (a * b) mod q by fixed-length (64 step) double-and-add: no 128-bit
// division needed, so it is portable.  Used only for the one-off
// nonsingularity test at AST construction time.
inline std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t q) noexcept {
  a %= q;
  std::uint64_t r = 0;
  for (int i = 63; i >= 0; --i) {
    r = addmod(r, r, q);
    r = addmod(r, a & (0 - ((b >> i) & 1u)), q);
  }
  return r;
}

}  // namespace detail

class EllipticCurveNode;

// Resolves the Frobenius trace a_q for a curve (Schoof / SEA, a lookup table,
// a certificate from an external tool, ...).
using TraceResolver = std::function<std::optional<std::int64_t>(const EllipticCurveNode&)>;

class EllipticCurveNode {
 public:
  static constexpr std::int64_t kMaxModulus = std::int64_t{1} << 62;

  EllipticCurveNode(std::int64_t a, std::int64_t b, std::int64_t q) noexcept
      : q_(q), a_(reduce(a, q)), b_(reduce(b, q)) {}

  std::int64_t a() const noexcept { return a_; }
  std::int64_t b() const noexcept { return b_; }
  std::int64_t q() const noexcept { return q_; }
  std::optional<std::int64_t> trace() const noexcept { return trace_; }

  // Attach a trace that has been established elsewhere.
  EllipticCurveNode with_trace(std::int64_t a_q) const noexcept {
    EllipticCurveNode c = *this;
    c.trace_ = a_q;
    return c;
  }

  // Ask a resolver for the trace; unresolved if it declines.
  EllipticCurveNode resolve(const TraceResolver& r) const {
    EllipticCurveNode c = *this;
    c.trace_ = r ? r(*this) : std::nullopt;
    return c;
  }

  // Well-formedness: 3 < q <= 2^62, q odd, q % 3 != 0 (cheap necessary
  // conditions for a prime > 3) and 4a^3 + 27b^2 != 0 (mod q).
  bool valid() const noexcept {
    return q_ > 3 && q_ <= kMaxModulus && (q_ & 1) == 1 && (q_ % 3) != 0 && nonsingular();
  }

 private:
  static std::int64_t reduce(std::int64_t x, std::int64_t q) noexcept {
    return q > 0 ? ((x % q) + q) % q : 0;
  }

  bool nonsingular() const noexcept {
    const std::uint64_t q = static_cast<std::uint64_t>(q_);
    const std::uint64_t a = static_cast<std::uint64_t>(a_);
    const std::uint64_t b = static_cast<std::uint64_t>(b_);
    const std::uint64_t a3 = detail::mulmod(detail::mulmod(a, a, q), a, q);
    const std::uint64_t b2 = detail::mulmod(b, b, q);
    const std::uint64_t disc =
        detail::addmod(detail::mulmod(4, a3, q), detail::mulmod(27, b2, q), q);
    return disc != 0;
  }

  std::int64_t q_;
  std::int64_t a_;
  std::int64_t b_;
  std::optional<std::int64_t> trace_;
};

// Explicit stub: a real Schoof/SEA implementation goes here.
inline std::optional<std::int64_t> schoof_trace_stub(const EllipticCurveNode&) noexcept {
  return std::nullopt;
}

// The lifted generator: the 2x2 buffer together with its field size.
struct FrobeniusNode {
  Mat2 F;
  std::int64_t q;
};

using FiniteFieldNode = std::variant<EllipticCurveNode, FrobeniusNode>;

class FrobeniusGenerator {
 public:
  // Curve -> companion-matrix Frobenius generator.  nullopt if the curve is
  // ill-formed or its trace has not been resolved.
  static std::optional<FrobeniusNode> lift(const EllipticCurveNode& e) noexcept {
    const std::optional<std::int64_t> t = e.trace();
    if (!e.valid() || !t.has_value()) return std::nullopt;
    const std::int64_t q = e.q();
    return FrobeniusNode{Mat2{{0, -q, 1, *t}}, q};
  }

  // Same lift, delivered as the engine's symnet::Matrix2x2 (double).  nullopt
  // when the curve is not liftable OR an entry (-q, a_q) is not exactly
  // representable as a double (|entry| > 2^53); use the int64 Mat2 then.
  static std::optional<symnet::Matrix2x2> lift_to_engine(const EllipticCurveNode& e) noexcept {
    const std::optional<FrobeniusNode> n = lift(e);
    if (!n) return std::nullopt;
    return toEngine(n->F);
  }

  // Variant entry point: a FrobeniusNode is already lifted.
  static std::optional<FrobeniusNode> lift(const FiniteFieldNode& n) {
    struct Visitor {
      std::optional<FrobeniusNode> operator()(const EllipticCurveNode& e) const noexcept {
        return FrobeniusGenerator::lift(e);
      }
      std::optional<FrobeniusNode> operator()(const FrobeniusNode& f) const noexcept { return f; }
    };
    return std::visit(Visitor{}, n);
  }
};

}  // namespace symnet::ff

#endif  // SYMNET_FINITE_FIELD_FROBENIUS_GENERATOR_HPP
