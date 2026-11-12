// tests/finite_field_test.cpp
//
// Validation harness.  The brute-force point counters below are TEST-ONLY
// oracles (naive loops over F_p and F_{p^2}); the headers under test contain
// no point-counting loop.  They give independent ground truth for:
//   * a_q = q + 1 - #E(F_q)  ->  lift  ->  Hasse-Weil and functional equation
//   * #E(F_{p^2}) predicted by det(I - F^2)  vs  a direct count over F_{p^2}
//   * the collapsed zeta function vs the defining relation  k Z_k = sum N_j Z_{k-j}
//   * the corrected functional equation, and the failure of the "q T^2 Z(T)" form
#include <cstdint>
#include <cstdio>
#include <vector>

#include "symnet/finite_field/WeilProver.hpp"

using namespace symnet::ff;

static int g_checks = 0, g_fail = 0;
#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::printf("  FAILED: %s  (line %d)\n", #cond, __LINE__);             \
    }                                                                        \
  } while (0)

// ---- test-only oracles ------------------------------------------------------
static long long brute_count_Fp(long long a, long long b, long long p) {
  std::vector<int> sq(p, 0);
  for (long long y = 0; y < p; ++y) sq[y * y % p]++;
  long long n = 1;  // point at infinity
  for (long long x = 0; x < p; ++x) n += sq[((x * x % p) * x + a * x + b) % p];
  return n;
}

static long long brute_count_Fp2(long long a, long long b, long long p) {
  long long nu = 2;  // quadratic non-residue
  for (;; ++nu) {
    bool is_sq = false;
    for (long long y = 0; y < p; ++y) is_sq |= (y * y % p == nu % p);
    if (!is_sq) break;
  }
  struct E2 { long long x0, x1; };
  auto mulE = [&](E2 u, E2 v) {
    return E2{(u.x0 * v.x0 + nu * (u.x1 * v.x1 % p)) % p, (u.x0 * v.x1 + u.x1 * v.x0) % p};
  };
  std::vector<int> sq(p * p, 0);
  for (long long y0 = 0; y0 < p; ++y0)
    for (long long y1 = 0; y1 < p; ++y1) {
      const E2 s = mulE({y0, y1}, {y0, y1});
      sq[s.x0 * p + s.x1]++;
    }
  long long n = 1;
  for (long long x0 = 0; x0 < p; ++x0)
    for (long long x1 = 0; x1 < p; ++x1) {
      const E2 x{x0, x1};
      const E2 x3 = mulE(mulE(x, x), x);
      n += sq[((x3.x0 + a * x0 + b) % p) * p + (x3.x1 + a * x1) % p];
    }
  return n;
}

// ---- exact rationals for the functional-equation demo -----------------------
using i128 = __int128;
static i128 gcd128(i128 a, i128 b) { if (a < 0) a = -a; if (b < 0) b = -b; while (b) { i128 t = a % b; a = b; b = t; } return a; }
struct Frac {
  i128 n, d;
  static Frac make(i128 n, i128 d) { if (d < 0) { n = -n; d = -d; } i128 g = gcd128(n, d); return {n / g, d / g}; }
};
static Frac operator+(Frac a, Frac b) { return Frac::make(a.n * b.d + b.n * a.d, a.d * b.d); }
static Frac operator*(Frac a, Frac b) { return Frac::make(a.n * b.n, a.d * b.d); }
static Frac operator/(Frac a, Frac b) { return Frac::make(a.n * b.d, a.d * b.n); }
static bool operator==(Frac a, Frac b) { return a.n == b.n && a.d == b.d; }
static Frac evalPoly(const Poly2& p, Frac u) { return Frac::make(p.c[0], 1) + Frac::make(p.c[1], 1) * u + Frac::make(p.c[2], 1) * u * u; }
static Frac evalZ(const ZetaNode& z, Frac u) { return evalPoly(z.num, u) / evalPoly(z.den, u); }

int main() {
  // ---- 1. Sweep: every nonsingular curve over small primes ----------------------
  std::printf("[1] sweep over all nonsingular curves y^2 = x^3 + ax + b over F_p\n");
  const long long primes[] = {5, 7, 11, 13, 17, 19, 23, 29, 31};
  long long curves = 0, supersingular = 0;
  for (long long p : primes) {
    for (long long a = 0; a < p; ++a)
      for (long long b = 0; b < p; ++b) {
        EllipticCurveNode e(a, b, p);
        if (!e.valid()) continue;
        ++curves;

        const long long N1 = brute_count_Fp(a, b, p);        // oracle
        const long long t = p + 1 - N1;
        const auto node = FrobeniusGenerator::lift(e.with_trace(t));
        CHECK(node.has_value());
        const Mat2& F = node->F;

        // Lefschetz at n = 1, trace, determinant, characteristic polynomial.
        CHECK(LefschetzTracePass::trace(F) == t);
        CHECK(LefschetzTracePass::determinant(F).value == p);
        const auto cp = LefschetzTracePass::characteristic_polynomial(F);
        CHECK((cp.value == Poly2{{1, -t, p}}) && !cp.overflow);
        CHECK(LefschetzTracePass::count_points(F, 1).value() == N1);

        // Both theorems.
        const WeilCertificate cert = WeilProver::certify(*node);
        CHECK(cert.hasse.verdict == Verdict::PROVED);
        CHECK(cert.functional_equation.verdict == Verdict::PROVED);
        supersingular += cert.hasse.delta_zero;

        // Zeta closed form.
        const auto z = ZetaFunctionPass::collapse(F, p);
        CHECK(z.has_value());
        CHECK((z->den == Poly2{{1, -(p + 1), p}}));

        // Series relation  k Z_k = sum_{j<=k} N_j Z_{k-j}  (T Z'/Z = sum N_j T^j),
        // with Z_k from long division of num/den and N_j from the Lefschetz pass.
        i128 Zk[7];
        Zk[0] = 1;
        for (int k = 1; k <= 6; ++k) {
          i128 v = (k <= 2) ? z->num.c[k] : 0;
          for (int i = 1; i <= 2 && i <= k; ++i) v -= (i128)z->den.c[i] * Zk[k - i];
          Zk[k] = v;
        }
        for (int k = 1; k <= 6; ++k) {
          i128 rhs = 0;
          for (int j = 1; j <= k; ++j) rhs += (i128)LefschetzTracePass::count_points(F, j).value() * Zk[k - j];
          CHECK((i128)k * Zk[k] == rhs);
        }
      }
  }
  std::printf("    %lld curves, %lld supersingular-with-q-square cases\n", curves, supersingular);

  // ---- 2. Extension fields: det(I - F^2) vs direct count over F_{p^2} ------------
  std::printf("[2] #E(F_{p^2}) from F^2 vs direct count over F_{p^2}\n");
  for (long long p : {5LL, 7LL, 11LL}) {
    for (long long a = 0; a < p; ++a)
      for (long long b = 0; b < p; ++b) {
        EllipticCurveNode e(a, b, p);
        if (!e.valid()) continue;
        const long long t = p + 1 - brute_count_Fp(a, b, p);
        const Mat2 F = FrobeniusGenerator::lift(e.with_trace(t))->F;
        CHECK(LefschetzTracePass::count_points(F, 2).value() == brute_count_Fp2(a, b, p));
        // Tr(F^2) = t^2 - 2q  (Newton identity)
        CHECK(LefschetzTracePass::trace_of_power(F, 2).value() == t * t - 2 * p);
      }
  }

  // ---- 3. Worked example: y^2 = x^3 + 2x + 3 over F_97 ---------------------------
  std::printf("[3] worked example y^2 = x^3 + 2x + 3 over F_97\n");
  {
    EllipticCurveNode e(2, 3, 97);
    const long long N1 = brute_count_Fp(2, 3, 97);
    const long long t = 97 + 1 - N1;
    const FrobeniusNode node = *FrobeniusGenerator::lift(e.with_trace(t));
    const WeilCertificate c = WeilProver::certify(node);
    const ZetaNode z = *ZetaFunctionPass::collapse(node.F, 97);
    std::printf("    #E(F_97)=%lld  a_q=%lld  F=[[%lld,%lld],[%lld,%lld]]\n", N1, t,
                (long long)node.F.m[0], (long long)node.F.m[1], (long long)node.F.m[2], (long long)node.F.m[3]);
    std::printf("    Delta = t^2-4q = %lld  ->  Hasse-Weil %s\n", t * t - 4 * 97, to_string(c.hasse.verdict));
    std::printf("    Functional equation: collapse=%d pairing=%d num=%d den=%d -> %s\n",
                c.functional_equation.inverse_collapse, c.functional_equation.weil_pairing,
                c.functional_equation.numerator_reflects, c.functional_equation.denominator_reflects,
                to_string(c.functional_equation.verdict));
    std::printf("    Z(T) = %s\n", to_string(z).c_str());
    for (int n : {1, 2, 3, 5}) {
      std::printf("    #E(F_97^%d) = %lld\n", n, (long long)LefschetzTracePass::count_points(node.F, n).value());
    }

    // Exact rational check: Z(1/(qT)) == Z(T), and the "q T^2 Z(T)" form is false.
    const Frac T0 = Frac::make(3, 1), u = Frac::make(1, 97 * 3);
    const Frac lhs = evalZ(z, u), rhs = evalZ(z, T0);
    const Frac literal = Frac::make(97, 1) * T0 * T0 * rhs;
    CHECK(lhs == rhs);
    CHECK(!(lhs == literal));
    std::printf("    Z(1/(qT)) == Z(T) at T=3: %s;  q T^2 Z(T) == Z(1/(qT)): %s\n",
                lhs == rhs ? "true" : "false", lhs == literal ? "true" : "false");
  }

  // ---- 4. Negative tests: the checkers must be able to say no --------------------
  std::printf("[4] negative tests\n");
  {
    // Trace violating Hasse: q = 5, t = 100.
    const Mat2 bad{{0, -5, 1, 100}};
    CHECK(WeilProver::prove_hasse_weil(bad, 5).verdict == Verdict::REFUTED);
    // t = 5, q = 5: Delta = 25 - 20 > 0.
    CHECK(WeilProver::prove_hasse_weil(Mat2{{0, -5, 1, 5}}, 5).verdict == Verdict::REFUTED);
    // Boundary t^2 = 4q is accepted (q = 4 is not prime; checks the comparison only, q must be > 3).
    CHECK(WeilProver::prove_hasse_weil(Mat2{{0, -4, 1, 4}}, 4).delta_zero);
    // det(F) != q: not a Frobenius, both theorems refuse.
    const Mat2 notfrob{{0, -7, 1, 2}};
    CHECK(WeilProver::prove_hasse_weil(notfrob, 5).verdict == Verdict::REFUTED);
    CHECK(WeilProver::prove_functional_equation(notfrob, 5).verdict == Verdict::REFUTED);
    // Unresolved trace -> no lift; singular curve -> invalid; stub resolver -> unresolved.
    CHECK(!FrobeniusGenerator::lift(EllipticCurveNode(2, 3, 97)).has_value());
    CHECK(!EllipticCurveNode(0, 0, 7).valid());
    CHECK(!FrobeniusGenerator::lift(EllipticCurveNode(2, 3, 97).resolve(schoof_trace_stub)).has_value());
    // Huge q near 2^62 with |t| near 2 sqrt(q): 128-bit comparison, no overflow.
    const std::int64_t bigq = (std::int64_t{1} << 62) - 57;   // value only used arithmetically
    const std::int64_t tmax = 2147483647LL * 2;               // ~2 * 2^31 ~ 2 sqrt(2^62)
    CHECK(WeilProver::prove_hasse_weil(Mat2{{0, -bigq, 1, tmax}}, bigq).delta_negative);
    CHECK(WeilProver::prove_hasse_weil(Mat2{{0, -bigq, 1, tmax + 100}}, bigq).verdict == Verdict::REFUTED);
    // Powers whose count leaves int64 are reported, not wrapped.
    const Mat2 F97 = FrobeniusGenerator::lift(EllipticCurveNode(2, 3, 97).with_trace(-2))->F;
    CHECK(!LefschetzTracePass::count_points(F97, 40).has_value());
    CHECK(!LefschetzTracePass::count_points(F97, 0).has_value());
    CHECK(LefschetzTracePass::count_points(F97, 9).has_value());
  }

  std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
