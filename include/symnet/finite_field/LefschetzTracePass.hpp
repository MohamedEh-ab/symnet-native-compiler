// include/symnet/finite_field/LefschetzTracePass.hpp
//
// Lefschetz trace pass on the 2x2 Frobenius buffer.
//
//   P(T)       = det(I - T F) = 1 - Tr(F) T + det(F) T^2      (fixed cost)
//   Tr(F^n)    = alpha^n + beta^n,    alpha, beta = eigenvalues of F
//   #E(F_{q^n}) = det(I - F^n) = q^n + 1 - Tr(F^n)             (Lefschetz)
//
// About "O(1) eigenvalue diagonalisation" -- what is and is not possible:
//   The eigenvalues alpha, beta are the roots of x^2 - a_q x + q.  When the
//   discriminant a_q^2 - 4q is negative (the Hasse-Weil situation) they are
//   non-real conjugates in Q(sqrt(a_q^2 - 4q)); they are never available as
//   exact integers, and floating point is excluded by design.  The
//   diagonalisation is therefore performed EXACTLY inside the ring
//   Z[x]/(x^2 - a_q x + q), which is what binary exponentiation of the
//   integer matrix computes.  Cost: at most 64 iterations (bit length of n),
//   each a constant number of 2x2 products with a branch-free mask select.
//   Truly O(1) in n is impossible for exact output: Tr(F^n) has Theta(n log q)
//   bits.  Results that do not fit in int64 are reported, never wrapped.
#ifndef SYMNET_FINITE_FIELD_LEFSCHETZ_TRACE_PASS_HPP
#define SYMNET_FINITE_FIELD_LEFSCHETZ_TRACE_PASS_HPP

#include <cstdint>
#include <optional>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include "symnet/Mat2.hpp"

namespace symnet::ff {

class LefschetzTracePass {
 public:
  // Tr(F) = m00 + m11 by SIMD diagonal summation.
  //   AVX2:   load the aligned 4 x int64 row-major buffer, broadcast lane 3
  //           (m11) over all lanes, add, read lane 0.
  //   Fallback: the identical scalar expression (wrapping add).
  // The wrap is harmless for Frobenius buffers (entries are 0, -q, 1, a_q);
  // powers are screened by `power`'s overflow flag before being traced.
  static std::int64_t trace(const Mat2& F) noexcept {
#if defined(__AVX2__)
    const __m256i v = _mm256_load_si256(reinterpret_cast<const __m256i*>(F.m));  // m00 m01 m10 m11
    const __m256i d = _mm256_permute4x64_epi64(v, 0xFF);                          // m11 m11 m11 m11
    const __m256i s = _mm256_add_epi64(v, d);                                     // lane0 = m00 + m11
    return static_cast<std::int64_t>(_mm_cvtsi128_si64(_mm256_castsi256_si128(s)));
#else
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(F.m[0]) +
                                     static_cast<std::uint64_t>(F.m[3]));
#endif
  }

  static Checked<std::int64_t> determinant(const Mat2& F) noexcept { return det(F); }

  // P(T) = det(I - T F) = 1 - Tr(F) T + det(F) T^2.
  static Checked<Poly2> characteristic_polynomial(const Mat2& F) noexcept {
    std::int64_t neg_tr;
    bool o = sub_ovf(0, trace(F), neg_tr);
    const Checked<std::int64_t> d = det(F);
    o |= d.overflow;
    return {Poly2{{1, neg_tr, d.value}}, o};
  }

  // F^n by binary exponentiation, branch-free per step (mask select).
  // overflow == true means some needed intermediate left int64.
  static Checked<Mat2> power(const Mat2& F, std::uint64_t n) noexcept {
    int len = 0;
    for (std::uint64_t x = n; x != 0; x >>= 1) ++len;

    Mat2 result = Mat2::identity();
    Mat2 base = F;
    bool o = false;
    for (int i = 0; i < len; ++i) {
      const bool bit = ((n >> i) & 1u) != 0;
      const Checked<Mat2> prod = mul(result, base);
      o |= (prod.overflow & bit);                 // only counts if the product is used
      result = select(bit, prod.value, result);
      const Checked<Mat2> sq = mul(base, base);
      o |= (sq.overflow & (i + 1 < len));         // the last squaring is never used
      base = sq.value;
    }
    return {result, o};
  }

  // Tr(F^n) = alpha^n + beta^n.
  static std::optional<std::int64_t> trace_of_power(const Mat2& F, std::uint64_t n) noexcept {
    const Checked<Mat2> p = power(F, n);
    if (p.overflow) return std::nullopt;
    return trace(p.value);
  }

  // Lefschetz: #E(F_{q^n}) = det(I - F^n), n >= 1.  nullopt on n == 0 or if
  // the count cannot be certified inside int64.
  static std::optional<std::int64_t> count_points(const Mat2& F, std::uint64_t n) noexcept {
    if (n == 0) return std::nullopt;
    const Checked<Mat2> p = power(F, n);
    const Checked<Mat2> imf = sub(Mat2::identity(), p.value);
    const Checked<std::int64_t> d = det(imf.value);
    if (p.overflow | imf.overflow | d.overflow) return std::nullopt;
    return d.value;
  }
};

}  // namespace symnet::ff

#endif  // SYMNET_FINITE_FIELD_LEFSCHETZ_TRACE_PASS_HPP
