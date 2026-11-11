// test_engine.cpp - C++ unit tests and benchmark for the SymNet engine (target: unit_tests).
//
// Asserts: XOR / Klein-4 automated lift synthesis, D4 constant folding (one matrix pass),
// SO(2) phasor composition to double precision, angular sector decisions over dense sweeps,
// folded-vs-unfolded equivalence, and prints nanosecond metrics for folded runtime passes.
//
//   unit_tests             full suite + benchmark
//   unit_tests --verbose   also print the full DSL transcripts
//   unit_tests --quick     fewer sweep angles / benchmark iterations (used by ctest)
//
// Exit status: 0 when every check passes.
#include <symnet/Examples.hpp>
#include <symnet/Interpreter.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace symnet;
using namespace symnet::examples;

// =====================================================================================
// Generated DSL programs
// =====================================================================================

static std::string denseSweepSource(int steps) {
    return std::string(R"(
group DenseSweep {
    gen R(theta) = phasor(theta)
    sector Wedge  = [pi/4, 3*pi/4]
    sector Reflex = [pi/2, 2*pi]
)") + "    sweep Wedge  over R from 0 to 2*pi steps " + std::to_string(steps) +
           "\n    sweep Reflex over R from 0 to 2*pi steps " + std::to_string(steps) + "\n}\n";
}

// D4 word r*s*r*s*... of length L, classified against H = {e, s}.
static std::string benchSource(int L) {
    std::string word;
    for (int i = 0; i < L; ++i) word += (i ? " * " : "") + std::string(i % 2 ? "s" : "r");
    return "group Bench {\n"
           "  gen r = rotation(90)\n"
           "  gen s = matrix([[-1, 0], [0, 1]])\n"
           "  subgroup D = span(r, s)\n"
           "  subgroup H = span(s)\n"
           "  separate InH { weight = [0, 0, 0, 1], bias = -0.5 }\n"
           "  classify InH of " + word + "\n}\n";
}

// =====================================================================================
// Test harness
// =====================================================================================

namespace {

using Clock = std::chrono::steady_clock;

struct Suite {
    int passed = 0, failed = 0;
    void check(bool ok, const std::string& name, const std::string& detail = "") {
        (ok ? passed : failed)++;
        std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << name;
        if (!detail.empty()) std::cout << "  (" << detail << ")";
        std::cout << "\n";
    }
};

template <class T>
inline void doNotOptimize(T& v) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : "+r,m"(v) : : "memory");
#else
    static volatile char sink;
    sink = static_cast<char>(reinterpret_cast<std::uintptr_t>(&v));
#endif
}

void banner(const std::string& text) {
    std::cout << "\n" << std::string(78, '=') << "\n" << text << "\n" << std::string(78, '=') << "\n";
}

const Job* findJob(const std::vector<CompiledGroup>& groups, const std::string& needle) {
    for (const auto& g : groups)
        for (const Task& t : g.tasks) {
            if (const auto* j = std::get_if<Job>(&t)) {
                if (j->label.find(needle) != std::string::npos) return j;
            } else if (const auto* v = std::get_if<VerifyJob>(&t)) {
                for (const Job& j2 : v->jobs)
                    if (j2.label.find(needle) != std::string::npos) return &j2;
            }
        }
    return nullptr;
}

const VerifyJob* findVerify(const std::vector<CompiledGroup>& groups, const std::string& needle) {
    for (const auto& g : groups)
        for (const Task& t : g.tasks)
            if (const auto* v = std::get_if<VerifyJob>(&t))
                if (v->label.find(needle) != std::string::npos) return v;
    return nullptr;
}

// Fraction of jobs of a VerifyJob classified as expected by the folded runtime.
double separationRate(const SymNetRuntime& rt, const VerifyJob& vj) {
    SymNetRuntime::Scratch s;
    int good = 0;
    for (const Job& j : vj.jobs) good += (rt.decideFast(j.folded, j.inputVec.data(), s) == *j.expected) ? 1 : 0;
    return static_cast<double>(good) / static_cast<double>(vj.jobs.size());
}

// Minimum-of-repeats ns/pass for a callable executing one pass.
template <class Fn>
double nsPerPass(Fn&& fn, long iters, int repeats = 5) {
    double best = 1e300;
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        for (long i = 0; i < iters; ++i) fn(i);
        const auto t1 = Clock::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(iters));
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    bool verbose = false, quick = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--verbose") || !std::strcmp(argv[i], "-v")) verbose = true;
        else if (!std::strcmp(argv[i], "--quick")) quick = true;
        else { std::cerr << "usage: " << argv[0] << " [--verbose] [--quick]\n"; return 2; }
    }

    SymNetInterpreter interp(10.0, verbose ? &std::cout : nullptr);
    const SymNetRuntime& rt = interp.runtime();
    Suite suite;
    Report dsl;
    int exitCode = 0;

    try {
        auto runProgram = [&](const char* title, const std::string& src, int expectedPasses) {
            if (verbose) banner(title);
            const Report r = interp.run(src);
            dsl += r;
            suite.check(r.failed == 0 && r.passed == expectedPasses,
                        std::string(title) + ": DSL checks",
                        sfmt("%d/%d passed, folded==reference on %d jobs", r.passed, expectedPasses, r.xChecked - r.xFailed));
            suite.check(r.xFailed == 0, std::string(title) + ": folded graph == unfolded Kronecker reference");
        };

        // ------------------------------------------------------------------------------
        banner("PHASE 1 REGRESSION (hand-written lift, D4 word and subgroup)");
        runProgram("Klein4 hand-written lift", KLEIN_XOR_SOURCE, 4);
        runProgram("D4 word r*s*r and subgroup H", D4_SOURCE, 9);

        // ------------------------------------------------------------------------------
        banner("TEST 1: XOR / Klein 4-group automated kernel lift synthesis");
        {
            runProgram("XOR point set", XOR_SPACE_SOURCE, 6);
            const auto groups = interp.compile(XOR_SPACE_SOURCE);
            const auto& clf = groups[0].classifiers.at("XorAuto");
            suite.check(clf.lifts && clf.lifts->size() == 1, "XOR: exactly one LiftLayer injected");
            suite.check(clf.lifts && fmtExpr(*clf.lifts->front().expr) == "abs(x - y)", "XOR: discovered lift is |x - y|",
                        clf.lifts ? clf.lifts->front().name + " = " + fmtExpr(*clf.lifts->front().expr) : "");
            suite.check(clf.weight == std::vector<double>({0.0, 0.0, 1.0}) && clf.bias == -0.5,
                        "XOR: hyperplane w = [0, 0, 1], b = -0.5", "w = " + fmtVec(clf.weight) + ", b = " + fmtG(clf.bias));
            const VerifyJob* vj = findVerify(groups, "XorAuto");
            suite.check(vj && separationRate(rt, *vj) == 1.0, "XOR: 100% hyperplane separation (4/4 points)");

            runProgram("Klein 4-group matrices", KLEIN_GROUP_SOURCE, 12);
            const auto kg = interp.compile(KLEIN_GROUP_SOURCE);
            for (const char* nm : {"InH", "InCosetA"}) {
                const auto& c = kg[0].classifiers.at(nm);
                suite.check(c.lifts && c.lifts->size() == 1, std::string("Klein4: ") + nm + " needs a synthesized lift",
                            c.lifts ? c.lifts->front().name + " = " + fmtExpr(*c.lifts->front().expr) : "");
                const VerifyJob* v = findVerify(kg, std::string("classifier ") + nm);
                suite.check(v && separationRate(rt, *v) == 1.0, std::string("Klein4: ") + nm + " 100% hyperplane separation (4/4)");
            }

            runProgram("D4 centre Z(D4)", D4_CENTER_SOURCE, 11);
            const auto dc = interp.compile(D4_CENTER_SOURCE);
            const auto& zc = dc[0].classifiers.at("InCenter");
            suite.check(zc.lifts && zc.lifts->size() == 1, "D4 centre: minimal single-lift feature map found");
            const VerifyJob* zv = findVerify(dc, "InCenter");
            suite.check(zv && separationRate(rt, *zv) == 1.0, "D4 centre: 100% hyperplane separation (8/8 elements)");
        }

        // ------------------------------------------------------------------------------
        banner("TEST 2: D4 constant folding (r * s * r -> ONE matrix multiplication pass)");
        {
            const auto groups = interp.compile(D4_SOURCE);
            const Job* ev = findJob(groups, "eval r * s * r");
            const Job* cl = findJob(groups, "classify InH of r * s * r");
            suite.check(ev && cl, "D4: eval and classify jobs compiled");
            suite.check(ev->network.layers.size() == 3, "D4: unfolded IR has 3 Linear layers (r, s, r)");
            suite.check(ev->folded.foldedFactors == 3, "D4: constant folding collapsed 3 factors into W_compiled");

            const DMat& r = groups[0].generators.at("r");
            const DMat& s = groups[0].generators.at("s");
            const Matrix2x2 R = Matrix2x2::fromData(r.data()), S = Matrix2x2::fromData(s.data());
            const Matrix2x2 direct = R * S * R;  // what the interpreter would multiply one by one
            Matrix2x2 W = Matrix2x2::fromData(ev->folded.Wc.data());
            suite.check(W.maxAbsDiff(direct) == 0.0, "D4: W_compiled == r*s*r (exact)", "W_compiled = " + fmtMat(W.a, 2, 2));
            suite.check(W.maxAbsDiff(Matrix2x2::fromData(s.data())) == 0.0, "D4: r*s*r == s (so it lies in H = {e, s})");

            gemmCounter() = 0;
            const ForwardResult fr = rt.forward(ev->folded, ev->inputVec, false);
            suite.check(gemmCounter() == 1, "D4: folded eval executes exactly 1 matrix multiplication pass",
                        sfmt("gemm passes = %zu", gemmCounter()));
            suite.check(maxAbsDiff(fr.finalVector, {-1, 0, 0, 1}) == 0.0, "D4: eval result == [[-1, 0], [0, 1]]", fmtVec(fr.finalVector));

            SymNetRuntime::Scratch scr;
            gemmCounter() = 0;
            const bool dec = rt.decideFast(cl->folded, cl->inputVec.data(), scr);
            suite.check(dec && gemmCounter() == 1, "D4: classify InH of r*s*r -> 1 in a single pass", sfmt("gemm passes = %zu", gemmCounter()));

            const ForwardResult ref = rt.forwardReference(cl->network, cl->inputVec);
            suite.check(ref.decision == dec && maxAbsDiff(ref.finalVector, rt.forward(cl->folded, cl->inputVec, false).finalVector) <= 1e-15,
                        "D4: folded result == Kronecker-expanded reference");
        }

        // ------------------------------------------------------------------------------
        banner("TEST 3: continuous SO(2) phasor composition (double-precision tolerance)");
        {
            runProgram("Continuous SO(2)", SO2_SOURCE, 20);
            const auto groups = interp.compile(SO2_SOURCE);
            const CheckJob* chk = nullptr;
            for (const Task& t : groups[0].tasks)
                if (const auto* c = std::get_if<CheckJob>(&t)) chk = c;
            suite.check(chk != nullptr, "SO(2): check R(pi/4) * R(pi/4) == R(pi/2) compiled");
            const ForwardResult a = rt.forward(chk->lhs.folded, chk->lhs.inputVec, false);
            const ForwardResult b = rt.forward(chk->rhs.folded, chk->rhs.inputVec, false);
            const double d = maxAbsDiff(a.finalVector, b.finalVector);
            suite.check(d < 1e-15, "SO(2): folded R(pi/4)*R(pi/4) == R(pi/2) with |diff| < 1e-15", sfmt("max |lhs - rhs| = %.3e", d));
            suite.check(chk->lhs.folded.foldedFactors == 2, "SO(2): two phasor factors folded into one 2x2 matrix");

            const Matrix2x2 q = ContinuousSO2Group::rotation(kPi / 4), h = ContinuousSO2Group::rotation(kPi / 2);
            const double dm = (q * q).maxAbsDiff(h);
            suite.check(dm < 1e-15, "SO(2): Matrix2x2 composition R(pi/4)^2 == R(pi/2)", sfmt("max diff = %.3e", dm));
            const auto z = ContinuousSO2Group::phasor(kPi / 4) * ContinuousSO2Group::phasor(kPi / 4);
            const double dz = std::abs(z - ContinuousSO2Group::phasor(kPi / 2));
            suite.check(dz < 1e-15, "SO(2): complex phasor e^(i pi/4) * e^(i pi/4) == e^(i pi/2)", sfmt("|diff| = %.3e", dz));
            const DMat Mz = ContinuousSO2Group::matrixFromPhasor(z);
            suite.check(maxAbsDiff(Mz.a, {0, -1, 1, 0}) == 0.0, "SO(2): phasor -> rotation matrix [[0,-1],[1,0]]", fmtMat(Mz.data(), 2, 2));

            double worst = 0.0;  // homomorphism e^(ia) e^(ib) = e^(i(a+b)) over a grid of angles
            for (int i = 0; i < 64; ++i)
                for (int j = 0; j < 64; ++j) {
                    const double A = kPi * i / 64.0, B = kPi * j / 64.0;
                    worst = std::max(worst, (ContinuousSO2Group::rotation(A) * ContinuousSO2Group::rotation(B)).maxAbsDiff(ContinuousSO2Group::rotation(A + B)));
                }
            suite.check(worst < 1e-14, "SO(2): homomorphism R(a)R(b) == R(a+b) over 4096 angle pairs", sfmt("worst diff = %.3e", worst));
        }

        // ------------------------------------------------------------------------------
        banner("TEST 4: continuous angular sector decisions across swept angles");
        {
            const int steps = quick ? 720 : 3600;
            const Report r = interp.run(denseSweepSource(steps));
            dsl += r;
            suite.check(r.failed == 0 && r.passed == 2 * steps && r.xFailed == 0,
                        sfmt("dense sweep: Wedge (AND) + Reflex (OR) over %d angles each vs analytic arc test", steps),
                        sfmt("%d/%d passed", r.passed, 2 * steps));

            const auto groups = interp.compile(SO2_SOURCE);
            const Job* inside = findJob(groups, "classify Wedge of R(pi / 4) * R(pi / 4) on [1, 0]");
            const Job* outside = findJob(groups, "classify Wedge of R(pi / 8) on [1, 0]");
            SymNetRuntime::Scratch s;
            suite.check(inside && rt.decideFast(inside->folded, inside->inputVec.data(), s), "sector: 90 deg is inside Wedge [45, 135]");
            suite.check(outside && !rt.decideFast(outside->folded, outside->inputVec.data(), s), "sector: 22.5 deg is outside Wedge [45, 135]");
            suite.check(ContinuousSO2Group::sectorPerceptrons(kPi / 4, 3 * kPi / 4).mode == "AND" &&
                            ContinuousSO2Group::sectorPerceptrons(kPi / 2, 2 * kPi).mode == "OR",
                        "sector: width <= pi compiles to AND, width > pi compiles to OR");
        }

        // ------------------------------------------------------------------------------
        banner("TEST 5: folded runtime metrics (nanoseconds per pass)");
        {
            const long iters = quick ? 200000 : 2000000;
            std::cout << "\n  Representative DSL jobs (folded graph, allocation-free hot path):\n";
            std::cout << sfmt("    %-52s %10s %8s\n", "job", "ns / pass", "gemm");
            struct Probe { const char* src; const char* needle; };
            const Probe probes[] = {
                {D4_SOURCE, "classify InH of r * s * r"},
                {D4_CENTER_SOURCE, "classify InCenter of r * r"},
                {KLEIN_GROUP_SOURCE, "classify InH of a * b"},
                {XOR_SPACE_SOURCE, "classify XorAuto on [1, 0]"},
                {SO2_SOURCE, "classify Wedge of R(pi / 4) * R(pi / 4) on [1, 0]"},
                {SO2_SOURCE, "classify Reflex of R(pi / 4) on [1, 0]"},
            };
            std::vector<std::pair<bool, std::string>> pendingChecks;
            for (const Probe& p : probes) {
                const auto groups = interp.compile(p.src);
                const Job* j = findJob(groups, p.needle);
                SymNetRuntime::Scratch s;
                std::vector<double> x = j->inputVec;
                gemmCounter() = 0;
                bool d0 = rt.decideFast(j->folded, x.data(), s);
                const std::size_t passes = gemmCounter();
                int acc = 0;
                const double ns = nsPerPass([&](long i) {
                    x[0] = j->inputVec[0] + static_cast<double>(i & 7) * 0.0;  // keep the compiler honest
                    const bool d = rt.decideFast(j->folded, x.data(), s);
                    acc += d;
                    doNotOptimize(acc);
                }, iters);
                std::cout << sfmt("    %-52s %10.2f %8zu\n", p.needle, ns, passes);
                pendingChecks.emplace_back(!j->expected || d0 == *j->expected, std::string("metric job decision correct: ") + p.needle);
            }

            for (const auto& [ok, name] : pendingChecks) suite.check(ok, name);
            std::cout << "\n  Word-length scaling, D4 word (r*s)^(L/2), classified against H:\n";
            std::cout << sfmt("    %6s %14s %16s %18s %12s %10s\n", "L", "folded ns", "unfolded O(Lk^3)", "dense-kron O(Lk^4)", "compile us", "gemm/pass");
            double foldedFirst = 0, foldedLast = 0, unfoldedLast = 0;
            bool allOnePass = true, allAgree = true;
            const int lengths[] = {3, 16, 128, 1024, 8192};
            for (int L : lengths) {
                const auto c0 = Clock::now();
                const auto groups = interp.compile(benchSource(L));
                const auto c1 = Clock::now();
                const Job& j = std::get<Job>(groups[0].tasks[0]);
                const KronProgram kp = SymNetRuntime::expandKron(j.network);
                SymNetRuntime::Scratch s;
                std::vector<double> x = j.inputVec;

                gemmCounter() = 0;
                const bool dF = rt.decideFast(j.folded, x.data(), s);
                const std::size_t passes = gemmCounter();
                const bool dU = rt.decideUnfolded(j.network, j.folded, x.data(), s);
                const bool dK = rt.decideKron(kp, j.folded, x.data(), s);
                const ForwardResult ref = rt.forwardReference(j.network, j.inputVec);
                allOnePass = allOnePass && passes == 1;
                allAgree = allAgree && dF == dU && dF == dK && ref.decision == dF &&
                           maxAbsDiff(ref.finalVector, rt.forward(j.folded, j.inputVec, false).finalVector) <= 1e-12;

                const long slowIters = std::max<long>(20, (quick ? 400000L : 4000000L) / L);
                int acc = 0;
                const double nF = nsPerPass([&](long) { acc += rt.decideFast(j.folded, x.data(), s); doNotOptimize(acc); }, iters);
                const double nU = nsPerPass([&](long) { acc += rt.decideUnfolded(j.network, j.folded, x.data(), s); doNotOptimize(acc); }, slowIters, 3);
                const double nK = nsPerPass([&](long) { acc += rt.decideKron(kp, j.folded, x.data(), s); doNotOptimize(acc); }, slowIters, 3);
                if (L == lengths[0]) foldedFirst = nF;
                foldedLast = nF;
                unfoldedLast = nU;
                std::cout << sfmt("    %6d %14.2f %16.2f %18.2f %12.1f %10zu\n", L, nF, nU, nK,
                                  std::chrono::duration<double, std::micro>(c1 - c0).count(), passes);
            }
            suite.check(allOnePass, "folded word evaluation performs exactly 1 matrix multiplication pass for every L (O(1) in L)");
            suite.check(allAgree, "folded == per-factor tensor path == dense Kronecker path == reference for every L");
            suite.check(foldedLast < unfoldedLast, "folded pass beats the unfolded O(L k^3) path at L = 8192",
                        sfmt("%.1f ns vs %.1f ns, %.0fx faster; folded L=8192 / L=3 = %.2fx", foldedLast, unfoldedLast,
                             unfoldedLast / foldedLast, foldedLast / foldedFirst));
        }

        // ------------------------------------------------------------------------------
        const int total = suite.passed + dsl.passed;
        const int failed = suite.failed + dsl.failed + dsl.xFailed;
        banner(sfmt("SUMMARY: %d checks passed, %d failed   (%d native assertions + %d DSL checks; %d folded-vs-reference equivalence checks)",
                    total, failed, suite.passed, dsl.passed, dsl.xChecked));
        exitCode = failed == 0 ? 0 : 1;
    } catch (const SymNetError& e) {
        std::cerr << "SymNet error: " << e.what() << "\n";
        return 2;
    }
    return exitCode;
}
