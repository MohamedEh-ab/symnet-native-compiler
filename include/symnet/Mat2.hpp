// include/symnet/finite_field/Mat2.hpp
//
// Exact 2x2 integer tensor buffer for the finite_field passes, integrated with
// the project's MatrixEngine.hpp.
//
// Why a separate integer type instead of symnet::Matrix2x2 (= Mat<2>, double)?
//   Every invariant here (Tr F, det F = q, Tr F^n, Delta = t^2 - 4q) must be an
//   EXACT integer.  A double carries 53 mantissa bits, but q goes up to 2^62 and
//   F^n entries grow like q^(n/2), so the engine's double kernels cannot be used
//   for these invariants.  Mat2 keeps the engine's conventions instead:
//     * the same 4-lane row-major layout  a[i*2+j]  (so toEngine/fromEngine are a
//       lane-wise copy),
//     * the same instrumentation: every ff::mul counts as one k x k pass in
//       symnet::gemmCounter() (define SYMNET_FF_NO_GEMM_COUNT to turn that off),
//     * the same discipline as the engine's element mode: words act on the 2x2
//       state directly (X <- W X, O(k^3)); no Kronecker product is ever built.
//   toEngine()/fromEngine() bridge to symnet::Matrix2x2 and refuse (nullopt)
//   whenever the conversion would not be exact.
//
// Design rules
//   * No floating point in any invariant.
//   * Kernels are branch-free: overflow is OR-accumulated into a flag that
//     travels with the value (Checked<T>); selects use masks.
//   * An `overflow` flag means "could not be certified inside int64", never a
//     silently wrong answer.
#ifndef SYMNET_FINITE_FIELD_MAT2_HPP
#define SYMNET_FINITE_FIELD_MAT2_HPP

#include <cmath>
#include <cstdint>
#include <optional>

#include "MatrixEngine.hpp"  // symnet::Mat<N>, symnet::Matrix2x2, symnet::gemmCounter()

namespace symnet::ff {

// ---------------------------------------------------------------------------
// 128-bit helper (portable; used for the overflow-free Hasse comparison)
// ---------------------------------------------------------------------------
struct U128 {
  std::uint64_t hi;
  std::uint64_t lo;
};

inline U128 mul64(std::uint64_t a, std::uint64_t b) noexcept {
  const std::uint64_t a0 = a & 0xFFFFFFFFull, a1 = a >> 32;
  const std::uint64_t b0 = b & 0xFFFFFFFFull, b1 = b >> 32;
  const std::uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  const std::uint64_t mid = (p00 >> 32) + (p01 & 0xFFFFFFFFull) + (p10 & 0xFFFFFFFFull);
  return U128{p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32),
              (mid << 32) | (p00 & 0xFFFFFFFFull)};
}

inline bool gt(const U128& a, const U128& b) noexcept {
  return (a.hi > b.hi) | ((a.hi == b.hi) & (a.lo > b.lo));
}
inline bool eq(const U128& a, const U128& b) noexcept {
  return (a.hi == b.hi) & (a.lo == b.lo);
}

// |x| as an unsigned value, branch-free and well defined for INT64_MIN.
inline std::uint64_t magnitude(std::int64_t x) noexcept {
  const std::uint64_t m = 0 - static_cast<std::uint64_t>(x < 0);
  return (static_cast<std::uint64_t>(x) ^ m) - m;
}

// ---------------------------------------------------------------------------
// Overflow-reporting scalar arithmetic.  Each returns true on overflow and
// writes the wrapped result to r.  Evaluation never branches on the data.
// ---------------------------------------------------------------------------
inline bool add_ovf(std::int64_t a, std::int64_t b, std::int64_t& r) noexcept {
  const std::uint64_t ua = static_cast<std::uint64_t>(a);
  const std::uint64_t ub = static_cast<std::uint64_t>(b);
  const std::uint64_t ur = ua + ub;
  r = static_cast<std::int64_t>(ur);
  return (((ua ^ ur) & (ub ^ ur)) >> 63) != 0;
}

inline bool sub_ovf(std::int64_t a, std::int64_t b, std::int64_t& r) noexcept {
  const std::uint64_t ua = static_cast<std::uint64_t>(a);
  const std::uint64_t ub = static_cast<std::uint64_t>(b);
  const std::uint64_t ur = ua - ub;
  r = static_cast<std::int64_t>(ur);
  return (((ua ^ ub) & (ua ^ ur)) >> 63) != 0;
}

inline bool mul_ovf(std::int64_t a, std::int64_t b, std::int64_t& r) noexcept {
#if (defined(__GNUC__) || defined(__clang__)) && !defined(SYMNET_FF_PORTABLE_ARITH)
  return __builtin_mul_overflow(a, b, &r);
#else
  // Portable path: 64x64 -> 128 magnitude product, then range-check.
  const U128 p = mul64(magnitude(a), magnitude(b));
  const std::uint64_t neg = static_cast<std::uint64_t>((a < 0) != (b < 0));
  const std::uint64_t mask = 0 - neg;
  const std::uint64_t limit = 0x7FFFFFFFFFFFFFFFull + neg;   // |INT64_MIN| = 2^63
  r = static_cast<std::int64_t>((p.lo ^ mask) - mask);
  return (p.hi != 0) | (p.lo > limit);
#endif
}

// ---------------------------------------------------------------------------
// Checked<T>: a value plus a sticky "may have overflowed" flag.
// ---------------------------------------------------------------------------
template <class T>
struct Checked {
  T value;
  bool overflow;
};

// ---------------------------------------------------------------------------
// Poly2: a polynomial c[0] + c[1] T + c[2] T^2 with exact integer coefficients.
// ---------------------------------------------------------------------------
struct Poly2 {
  std::int64_t c[3];
};
inline bool operator==(const Poly2& a, const Poly2& b) noexcept {
  return ((a.c[0] ^ b.c[0]) | (a.c[1] ^ b.c[1]) | (a.c[2] ^ b.c[2])) == 0;
}

// ---------------------------------------------------------------------------
// Mat2: the 2x2 tensor buffer, 64-byte aligned (one cache line, four int64
// lanes = one 256-bit vector; the upper 32 bytes are padding).
//   m = { m00, m01, m10, m11 }  (row-major)
// ---------------------------------------------------------------------------
struct alignas(64) Mat2 {
  std::int64_t m[4];

  static Mat2 identity() noexcept { return Mat2{{1, 0, 0, 1}}; }
};
static_assert(alignof(Mat2) == 64 && sizeof(Mat2) == 64, "Mat2 must occupy one 64-byte cache line");

inline bool equal(const Mat2& a, const Mat2& b) noexcept {
  return ((a.m[0] ^ b.m[0]) | (a.m[1] ^ b.m[1]) | (a.m[2] ^ b.m[2]) | (a.m[3] ^ b.m[3])) == 0;
}

// Branch-free lane select: c ? a : b.
inline Mat2 select(bool c, const Mat2& a, const Mat2& b) noexcept {
  const std::uint64_t k = 0 - static_cast<std::uint64_t>(c);
  Mat2 r;
  for (int i = 0; i < 4; ++i) {   // fixed trip count, fully unrolled by the optimiser
    r.m[i] = static_cast<std::int64_t>((static_cast<std::uint64_t>(a.m[i]) & k) |
                                       (static_cast<std::uint64_t>(b.m[i]) & ~k));
  }
  return r;
}

// C = A * B with sticky overflow.  (AVX2 has no 64-bit low multiply, so the
// product stays scalar; the compiler schedules the 8 multiplies in parallel.)
inline Checked<Mat2> mul(const Mat2& A, const Mat2& B) noexcept {
#if !defined(SYMNET_FF_NO_GEMM_COUNT)
  ++symnet::gemmCounter();   // one k x k pass, same accounting as symnet::gemm
#endif
  bool o = false;
  auto dot = [&o](std::int64_t a0, std::int64_t b0, std::int64_t a1, std::int64_t b1) noexcept {
    std::int64_t x, y, r;
    o |= mul_ovf(a0, b0, x);
    o |= mul_ovf(a1, b1, y);
    o |= add_ovf(x, y, r);
    return r;
  };
  const std::int64_t c00 = dot(A.m[0], B.m[0], A.m[1], B.m[2]);
  const std::int64_t c01 = dot(A.m[0], B.m[1], A.m[1], B.m[3]);
  const std::int64_t c10 = dot(A.m[2], B.m[0], A.m[3], B.m[2]);
  const std::int64_t c11 = dot(A.m[2], B.m[1], A.m[3], B.m[3]);
  return {Mat2{{c00, c01, c10, c11}}, o};
}

inline Checked<Mat2> sub(const Mat2& A, const Mat2& B) noexcept {
  bool o = false;
  std::int64_t r[4];
  for (int i = 0; i < 4; ++i) o |= sub_ovf(A.m[i], B.m[i], r[i]);
  return {Mat2{{r[0], r[1], r[2], r[3]}}, o};
}

inline Mat2 transpose(const Mat2& A) noexcept { return Mat2{{A.m[0], A.m[2], A.m[1], A.m[3]}}; }

inline Checked<std::int64_t> det(const Mat2& A) noexcept {
  std::int64_t x, y, r;
  bool o = mul_ovf(A.m[0], A.m[3], x);
  o |= mul_ovf(A.m[1], A.m[2], y);
  o |= sub_ovf(x, y, r);
  return {r, o};
}

// adj(A) = det(A) * A^{-1}; integral even when A^{-1} is not.
inline Checked<Mat2> adjugate(const Mat2& A) noexcept {
  std::int64_t n1, n2;
  bool o = sub_ovf(0, A.m[1], n1);
  o |= sub_ovf(0, A.m[2], n2);
  return {Mat2{{A.m[3], n1, n2, A.m[0]}}, o};
}

// A rational matrix num/den (den != 0) so that inverses stay integral.
struct ScaledMat2 {
  Mat2 num;
  std::int64_t den;
};

// ---------------------------------------------------------------------------
// Bridge to the double-based engine (symnet::Matrix2x2).  Both conversions are
// exact-or-nullopt:
//   toEngine   : every |entry| <= 2^53 (all such integers are exact doubles)
//   fromEngine : every entry is an integer value with |entry| <= 2^53
// ---------------------------------------------------------------------------
inline constexpr std::uint64_t kExactDoubleLimit = std::uint64_t{1} << 53;

inline std::optional<symnet::Matrix2x2> toEngine(const Mat2& A) noexcept {
  symnet::Matrix2x2 E;
  bool ok = true;
  for (int i = 0; i < 4; ++i) {
    ok &= magnitude(A.m[i]) <= kExactDoubleLimit;
    E.a[i] = static_cast<double>(A.m[i]);
  }
  if (!ok) return std::nullopt;
  return E;
}

inline std::optional<Mat2> fromEngine(const symnet::Matrix2x2& E) noexcept {
  Mat2 A;
  bool ok = true;
  for (int i = 0; i < 4; ++i) {
    const double v = E.a[i];
    const bool good = (v == std::nearbyint(v)) & (std::fabs(v) <= 9007199254740992.0);  // NaN fails
    ok &= good;
    A.m[i] = static_cast<std::int64_t>(good ? v : 0.0);
  }
  if (!ok) return std::nullopt;
  return A;
}

}  // namespace symnet::ff

#endif  // SYMNET_FINITE_FIELD_MAT2_HPP
