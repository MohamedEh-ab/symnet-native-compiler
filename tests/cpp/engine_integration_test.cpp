// tests/engine_integration_test.cpp
//
// Cross-checks the exact integer passes (symnet::ff) against the project's own
// double-based MatrixEngine.hpp kernels, on inputs where doubles are exact.
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>

#include "symnet/finite_field/WeilProver.hpp"

using namespace symnet::ff;

static int g_checks = 0, g_fail = 0;
#define CHECK(cond)                                                    \
  do {                                                                 \
    ++g_checks;                                                        \
    if (!(cond)) {                                                     \
      ++g_fail;                                                        \
      std::printf("  FAILED: %s  (line %d)\n", #cond, __LINE__);       \
    }                                                                  \
  } while (0)

int main() {
  std::mt19937_64 rng(12345);
  std::uniform_int_distribution<int> small(-50, 50);

  // 1. ff::mul == engine operator* (exact on small integers); counter accounting.
  std::printf("[1] ff::mul vs symnet::Matrix2x2 operator*, gemmCounter accounting\n");
  for (int it = 0; it < 5000; ++it) {
    const Mat2 A{{small(rng), small(rng), small(rng), small(rng)}};
    const Mat2 B{{small(rng), small(rng), small(rng), small(rng)}};
    const auto EA = toEngine(A), EB = toEngine(B);
    CHECK(EA.has_value() && EB.has_value());

    const std::size_t c0 = symnet::gemmCounter();
    const Checked<Mat2> P = mul(A, B);
    const std::size_t c1 = symnet::gemmCounter();
    const symnet::Matrix2x2 E = *EA * *EB;                 // engine kernel (also counts 1)
    const std::size_t c2 = symnet::gemmCounter();
    CHECK(c1 - c0 == 1 && c2 - c1 == 1);
    CHECK(!P.overflow);
    CHECK(equal(P.value, *fromEngine(E)));
  }

  // 2. F^n: ff::power vs repeated engine multiplication (exact while < 2^53).
  std::printf("[2] F^n: ff::power vs repeated engine products\n");
  const long long qs[] = {5, 7, 97, 101, 1009};
  for (long long q : qs) {
    for (long long t = -static_cast<long long>(2 * std::sqrt(double(q))); t <= static_cast<long long>(2 * std::sqrt(double(q))); ++t) {
      const Mat2 F{{0, -q, 1, t}};
      symnet::Matrix2x2 X = symnet::Matrix2x2::identity();
      const symnet::Matrix2x2 EF = *toEngine(F);
      for (int n = 1; n <= 5; ++n) {
        X = EF * X;
        const Checked<Mat2> P = LefschetzTracePass::power(F, static_cast<std::uint64_t>(n));
        CHECK(!P.overflow);
        CHECK(equal(P.value, *fromEngine(X)));
      }
    }
  }

  // 3. Exact F^{-1} = adj(F)/det(F)  vs  the engine's Gauss-Jordan inverse (DMat).
  std::printf("[3] adj(F)/q vs symnet::inverse(DMat)\n");
  for (long long q : qs) {
    for (long long t = -9; t <= 9; ++t) {
      const Mat2 F{{0, -q, 1, t}};
      symnet::DMat M(2, 2);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) M(i, j) = static_cast<double>(F.m[i * 2 + j]);
      const symnet::DMat Inv = symnet::inverse(M);
      const Checked<Mat2> adj = adjugate(F);
      const double den = static_cast<double>(det(F).value);
      double err = 0.0;
      for (int i = 0; i < 4; ++i) err = std::max(err, std::fabs(Inv.a[i] - static_cast<double>(adj.value.m[i]) / den));
      CHECK(err < 1e-12);
      // and the exact collapse used by the prover agrees:
      CHECK(WeilProver::prove_functional_equation(F, q).inverse_collapse);
    }
  }

  // 4. Bridges refuse inexact conversions.
  std::printf("[4] bridge refusals\n");
  const std::int64_t bigq = (std::int64_t{1} << 62) - 57;
  CHECK(!toEngine(Mat2{{0, -bigq, 1, 3}}).has_value());                      // > 2^53
  CHECK(toEngine(Mat2{{0, -(std::int64_t{1} << 53), 1, 3}}).has_value());     // == 2^53 exact
  CHECK(!toEngine(Mat2{{0, -((std::int64_t{1} << 53) + 1), 1, 3}}).has_value());
  symnet::Matrix2x2 H = symnet::Matrix2x2::identity();
  H.a[1] = 0.5;
  CHECK(!fromEngine(H).has_value());                                          // not an integer
  H.a[1] = std::numeric_limits<double>::quiet_NaN();
  CHECK(!fromEngine(H).has_value());
  H.a[1] = 1e300;
  CHECK(!fromEngine(H).has_value());
  CHECK(!FrobeniusGenerator::lift_to_engine(EllipticCurveNode(2, 3, 97)).has_value());  // no trace yet
  const auto EF = FrobeniusGenerator::lift_to_engine(EllipticCurveNode(2, 3, 97).with_trace(-2));
  CHECK(EF.has_value() && EF->a[0] == 0.0 && EF->a[1] == -97.0 && EF->a[2] == 1.0 && EF->a[3] == -2.0);

  // 5. Why the exact type exists: near q ~ 2^62 the double engine cannot hold F, ff can decide.
  std::printf("[5] large q: double engine refuses, exact path still certifies\n");
  const Mat2 Fbig{{0, -bigq, 1, 4294967294LL}};   // |t| <= 2 sqrt(q) (approx 2^32 - 2)
  CHECK(!toEngine(Fbig).has_value());
  CHECK(WeilProver::prove_hasse_weil(Fbig, bigq).verdict == Verdict::PROVED);

  std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
