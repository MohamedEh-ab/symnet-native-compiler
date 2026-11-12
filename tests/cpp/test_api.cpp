// test_api.cpp - native tests for the binding facade (include/symnet/Api.hpp).
//
// The Python bindings are thin wrappers over symnet::api, so these assertions pin down the
// behaviour that tests/python/test_bindings.py re-checks from Python.
#include <symnet/Api.hpp>
#include <symnet/Examples.hpp>

#include <cstdio>
#include <limits>

using namespace symnet;
namespace ex = symnet::examples;

static int g_checks = 0, g_fail = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        ++g_checks;                                                       \
        if (!(cond)) {                                                    \
            ++g_fail;                                                     \
            std::printf("  FAILED: %s  (line %d)\n", #cond, __LINE__);    \
        }                                                                 \
    } while (0)

template <class E, class Fn>
static bool throws(Fn&& fn) {
    try {
        fn();
    } catch (const E&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

int main() {
    // ---- tokenizer ------------------------------------------------------------------
    std::printf("[1] tokenize\n");
    {
        const auto toks = api::tokenizeSource("gen r = rotation(90)\n");
        CHECK(toks.size() == 8);  // gen r = rotation ( 90 ) + eof
        CHECK(std::get<0>(toks[0]) == "ident" && std::get<1>(toks[0]) == "gen" && std::get<2>(toks[0]) == 1);
        CHECK(std::get<0>(toks[2]) == "symbol" && std::get<0>(toks[5]) == "number");
        CHECK(std::get<0>(toks.back()) == "eof");
    }

    // ---- interpreter.run ---------------------------------------------------------------
    std::printf("[2] Interpreter.run\n");
    const api::Interpreter interp;
    {
        const api::RunResult r = interp.run(ex::D4_SOURCE);
        CHECK(r.passed == 9 && r.failed == 0 && r.xFailed == 0 && r.ok());
        CHECK(r.transcript.empty());
        const api::RunResult v = interp.run(ex::D4_SOURCE, true);
        CHECK(v.passed == 9 && v.transcript.find("[runtime]") != std::string::npos);
        int total = 0;
        for (const auto& e : ex::kExamples) total += interp.run(e.source).passed;
        CHECK(total == 62);  // same total as symnet.py --all
    }

    // ---- errors map to the SymNet hierarchy ---------------------------------------------
    std::printf("[3] errors\n");
    CHECK(throws<LexError>([&] { interp.run("group G { @ }"); }));
    CHECK(throws<ParseError>([&] { interp.run("group G {"); }));
    CHECK(throws<CompileError>([&] { interp.run("group G { classify Nope on (0, 0) }"); }));
    CHECK(throws<SymNetError>([&] { interp.compile("group G { gen r = rotation(90) classify Nope on (0, 0) }"); }));

    // ---- compile: synthesis result is inspectable ----------------------------------------
    std::printf("[3b] compile + inspect\n");
    {
        const api::Program p = interp.compile(ex::XOR_SPACE_SOURCE);
        CHECK(p.groups.size() == 1 && p.groups[0].name == "XORSpace");
        const auto& cl = p.groups[0].classifiers;
        CHECK(cl.size() == 1 && cl[0].name == "XorAuto" && !cl[0].isSector);
        CHECK(cl[0].lifts == std::vector<std::string>({"k1 = abs(x - y)"}));
        CHECK(cl[0].weight == std::vector<double>({0.0, 0.0, 1.0}) && cl[0].bias == -0.5);
        bool synthLine = false;
        for (const auto& d : p.groups[0].declarations) synthLine |= d.rfind("[synth]", 0) == 0;
        CHECK(synthLine);
        // tasks: one verify batch of 4 points, then two classify jobs
        const auto& tasks = p.groups[0].tasks;
        CHECK(tasks.size() == 3 && tasks[0].kind == "verify" && tasks[0].jobs.size() == 4 && tasks[1].kind == "job");
        int ok = 0;
        for (const auto& j : tasks[0].jobs) ok += (j.decide() == *j.expected());
        CHECK(ok == 4);
    }

    // ---- jobs: folding is visible and evaluation is one pass ------------------------------
    std::printf("[4] JobHandle: constant folding\n");
    {
        const api::Program p = interp.compile(ex::D4_SOURCE);
        const api::JobHandle* ev = nullptr;
        const api::JobHandle* cl = nullptr;
        for (const auto& t : p.groups[0].tasks)
            if (t.kind == "job") (t.jobs[0].kind() == "eval" ? ev : cl) = &t.jobs[0];
        CHECK(ev && cl);
        CHECK(ev->foldedFactors() == 3 && ev->elementMode() && ev->dim() == 2);
        CHECK(ev->foldedMatrix() == std::vector<double>({-1, 0, 0, 1}));  // r*s*r == s
        CHECK(ev->architecture().find("L3:r") != std::string::npos);
        CHECK(ev->foldedArchitecture().find("W_compiled (3 factors folded)") != std::string::npos);
        CHECK(ev->matrixShape().has_value() && *ev->matrixShape() == std::make_pair(2, 2));

        gemmCounter() = 0;
        const ForwardResult fr = ev->evaluate();
        CHECK(gemmCounter() == 1 && fr.finalVector == std::vector<double>({-1, 0, 0, 1}) && !fr.decision.has_value());
        CHECK(fr.trace.empty());
        CHECK(ev->evaluate(true).trace.size() == 2);       // input + W_compiled
        CHECK(ev->evaluate(true, true).trace.size() == 4); // reference: input + 3 layers

        gemmCounter() = 0;
        CHECK(cl->decide() && gemmCounter() == 1);
        const ForwardResult a = cl->evaluate(), b = cl->evaluate(false, true);
        CHECK(a.decision == b.decision && a.decision == std::optional<bool>(true));
        CHECK(a.score && b.score && std::fabs(*a.score - *b.score) < 1e-12 && a.softScore.has_value());
        CHECK(cl->expected() == std::optional<bool>(true));
        CHECK(throws<SymNetRuntimeError>([&] { ev->decide(); }));  // eval jobs have no hyperplane
    }
    {  // verify / check task shapes (SO(2) program)
        const api::Program p = interp.compile(ex::SO2_SOURCE);
        int checks = 0, verifies = 0;
        for (const auto& t : p.groups[0].tasks) {
            checks += t.kind == "check" && t.jobs.size() == 2;
            verifies += t.kind == "verify" && t.jobs.size() == 8;
        }
        CHECK(checks == 1 && verifies == 2);
        bool sector = false;
        for (const auto& c : p.groups[0].classifiers) sector |= c.isSector;
        CHECK(sector);
    }

    // ---- finite field -------------------------------------------------------------------
    std::printf("[5] finite-field facade\n");
    using namespace symnet::ff;
    {
        const EllipticCurveNode e(2, 3, 97);
        CHECK(e.valid() && !e.trace().has_value() && !FrobeniusGenerator::lift(e).has_value());
        const auto node = FrobeniusGenerator::lift(e.with_trace(-2));
        CHECK(node.has_value() && equal(node->F, Mat2{{0, -97, 1, -2}}));
        CHECK(api::ffDet(node->F) == 97);
        CHECK(api::ffCharPoly(node->F) == Poly2({{1, 2, 97}}));
        CHECK(LefschetzTracePass::count_points(node->F, 3).value() == 912100);
        CHECK(to_string(ZetaFunctionPass::collapse(node->F, 97).value()) == "(1 + 2T + 97T^2) / (1 - 98T + 97T^2)");
        const WeilCertificate c = WeilProver::certify(*node);
        CHECK(c.hasse.verdict == Verdict::PROVED && c.functional_equation.verdict == Verdict::PROVED);

        // resolver: returns a trace / declines
        const TraceResolver good = [](const EllipticCurveNode&) { return std::optional<std::int64_t>(-2); };
        CHECK(FrobeniusGenerator::lift(e.resolve(good)).has_value());
        CHECK(!FrobeniusGenerator::lift(e.resolve(schoof_trace_stub)).has_value());

        // Checked<T> -> overflow_error
        CHECK(equal(api::ffMul(node->F, node->F), Mat2{{-97, 194, -2, -93}}));
        const Mat2 big{{std::numeric_limits<std::int64_t>::max(), 0, 0, 1}};
        CHECK(throws<std::overflow_error>([&] { api::ffMul(big, big); }));
        CHECK(throws<std::overflow_error>([&] { api::ffPower(node->F, 40); }));
        CHECK(throws<std::overflow_error>([&] { api::ffDet(Mat2{{big.m[0], 0, 0, big.m[0]}}); }));
        CHECK(equal(api::ffPower(node->F, 2), api::ffMul(node->F, node->F)));
        CHECK(equal(api::ffSub(Mat2::identity(), Mat2::identity()), Mat2{{0, 0, 0, 0}}));
        CHECK(equal(api::ffAdjugate(node->F), Mat2{{-2, 97, -1, 0}}));
        const auto eng = toEngine(node->F);
        CHECK(eng.has_value() && equal(*fromEngine(*eng), node->F));
    }

    std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
