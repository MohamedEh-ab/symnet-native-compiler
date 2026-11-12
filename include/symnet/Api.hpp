// Api.hpp - binding-friendly facade over the DSL compiler/runtime and the finite-field passes.
//
// Everything Python-visible that involves logic lives HERE, as ordinary C++ with no pybind11
// types, so it can be unit-tested natively (tests/cpp/test_api.cpp).  src/bind_*.cpp only
// registers these types and functions with pybind11.
//
//   DSL      : Interpreter -> Program -> GroupHandle -> TaskHandle -> JobHandle
//              (snapshots of the compiler output; each JobHandle owns a copy of its Job, so
//              handles stay valid independently of one another)
//   Finite   : ffMul / ffDet / ffAdjugate / ffCharPoly / ffPower turn the sticky overflow flag
//   field      of ff::Checked<T> into std::overflow_error (Python: OverflowError)
#pragma once

#include "Interpreter.hpp"
#include "WordEval.hpp"
#include "finite_field/WeilProver.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace symnet::api {

// =====================================================================================
// DSL facade
// =====================================================================================

// Token stream as (kind, text, line); kind is "number" | "ident" | "symbol" | "eof".
inline std::vector<std::tuple<std::string, std::string, int>> tokenizeSource(const std::string& source) {
    static const char* const kinds[] = {"number", "ident", "symbol", "eof"};
    std::vector<std::tuple<std::string, std::string, int>> out;
    for (const Token& t : tokenize(source)) out.emplace_back(kinds[static_cast<int>(t.kind)], t.text, t.line);
    return out;
}

// Result of Interpreter::run.
struct RunResult {
    int passed = 0, failed = 0;     // DSL checks (same accounting as symnet.py)
    int xChecked = 0, xFailed = 0;  // folded-vs-reference equivalence checks
    std::string transcript;         // empty unless requested
    bool ok() const { return failed == 0 && xFailed == 0; }
};

// One compiled query. Owns a private copy of the Job (network, folded graph, input vector).
class JobHandle {
public:
    JobHandle(std::shared_ptr<const Job> job, double sharpness) : job_(std::move(job)), rt_(sharpness) {}

    const std::string& kind() const { return job_->kind; }
    const std::string& label() const { return job_->label; }
    const std::string& shortLabel() const { return job_->shortLabel; }
    std::optional<bool> expected() const { return job_->expected; }
    const std::vector<double>& input() const { return job_->inputVec; }
    bool elementMode() const { return job_->folded.elementMode; }
    int dim() const { return job_->folded.N; }
    std::optional<std::pair<int, int>> matrixShape() const { return job_->matrixShape; }

    std::string architecture() const { return job_->network.architecture(); }        // unfolded IR
    std::string foldedArchitecture() const { return job_->folded.architecture(); }   // after folding
    int foldedFactors() const { return job_->folded.foldedFactors; }
    std::vector<double> foldedMatrix() const { return job_->folded.Wc.a; }           // W_compiled, row-major

    // Run the job. Default: the constant-folded graph (one k x k pass). reference=true runs the
    // unfolded, Kronecker-expanded oracle (the Python-faithful per-layer trace).
    ForwardResult evaluate(bool trace = false, bool reference = false) const {
        return reference ? rt_.forwardReference(job_->network, job_->inputVec)
                         : rt_.forward(job_->folded, job_->inputVec, trace);
    }

    // Allocation-light decision on the folded graph (throws if the job has no hyperplane).
    bool decide() const {
        SymNetRuntime::Scratch scratch;
        return rt_.decideFast(job_->folded, job_->inputVec.data(), scratch);
    }

private:
    std::shared_ptr<const Job> job_;
    SymNetRuntime rt_;
};

// One statement-level task: a single job, a verify/sweep batch, or a check (lhs, rhs).
struct TaskHandle {
    std::string kind;  // "job" | "verify" | "check"
    std::string label;
    std::vector<JobHandle> jobs;
};

struct ClassifierInfo {
    std::string name;
    std::vector<double> weight;
    double bias = 0.0;
    std::vector<std::string> lifts;  // "k1 = abs(x - y)": user lifts followed by synthesized ones
    bool isSector = false;
};

struct GroupHandle {
    std::string name;
    std::vector<std::string> declarations;  // the compiler's declaration log (incl. [synth] lines)
    std::vector<ClassifierInfo> classifiers;  // sorted by name
    std::vector<TaskHandle> tasks;
};

struct Program {
    std::vector<GroupHandle> groups;
};

inline Program buildProgram(const std::vector<CompiledGroup>& groups, double sharpness) {
    Program p;
    for (const CompiledGroup& g : groups) {
        GroupHandle gh;
        gh.name = g.name;
        gh.declarations = g.declarations;
        for (const auto& [name, clf] : g.classifiers) {
            ClassifierInfo ci;
            ci.name = name;
            ci.weight = clf.weight;
            ci.bias = clf.bias;
            ci.isSector = clf.sector.has_value();
            for (const LiftDecl& ld : clf.lifts ? *clf.lifts : g.liftDecls)
                ci.lifts.push_back(ld.name + " = " + fmtExpr(*ld.expr));
            gh.classifiers.push_back(std::move(ci));
        }
        for (const Task& t : g.tasks) {
            TaskHandle th;
            auto add = [&](const Job& j) { th.jobs.emplace_back(std::make_shared<const Job>(j), sharpness); };
            if (const auto* j = std::get_if<Job>(&t)) {
                th.kind = "job";
                th.label = j->label;
                add(*j);
            } else if (const auto* v = std::get_if<VerifyJob>(&t)) {
                th.kind = "verify";
                th.label = v->label;
                for (const Job& jj : v->jobs) add(jj);
            } else {
                const auto& c = std::get<CheckJob>(t);
                th.kind = "check";
                th.label = c.label;
                add(c.lhs);
                add(c.rhs);
            }
            gh.tasks.push_back(std::move(th));
        }
        p.groups.push_back(std::move(gh));
    }
    return p;
}

// Entry point exposed to Python as dsl.Interpreter.
class Interpreter {
public:
    explicit Interpreter(double sharpness = 10.0) : sharpness_(sharpness) {}
    double sharpness() const { return sharpness_; }

    // Lex + parse + compile + run every query. With transcript=true the full Python-style
    // transcript is captured in RunResult::transcript.
    RunResult run(const std::string& source, bool transcript = false) const {
        std::ostringstream os;
        SymNetInterpreter interp(sharpness_, transcript ? &os : nullptr);
        const Report r = interp.run(source);
        RunResult out;
        out.passed = r.passed;
        out.failed = r.failed;
        out.xChecked = r.xChecked;
        out.xFailed = r.xFailed;
        out.transcript = os.str();
        return out;
    }

    // Compile without running; inspect or evaluate jobs individually.
    Program compile(const std::string& source) const {
        SymNetInterpreter interp(sharpness_);
        return buildProgram(interp.compile(source), sharpness_);
    }

private:
    double sharpness_;
};

// =====================================================================================
// Finite-field facade: Checked<T> -> exceptions
// =====================================================================================

inline ff::Mat2 ffMul(const ff::Mat2& a, const ff::Mat2& b) {
    const auto r = ff::mul(a, b);
    if (r.overflow) throw std::overflow_error("Mat2 product does not fit in int64");
    return r.value;
}

inline ff::Mat2 ffSub(const ff::Mat2& a, const ff::Mat2& b) {
    const auto r = ff::sub(a, b);
    if (r.overflow) throw std::overflow_error("Mat2 difference does not fit in int64");
    return r.value;
}

inline std::int64_t ffDet(const ff::Mat2& a) {
    const auto r = ff::det(a);
    if (r.overflow) throw std::overflow_error("determinant does not fit in int64");
    return r.value;
}

inline ff::Mat2 ffAdjugate(const ff::Mat2& a) {
    const auto r = ff::adjugate(a);
    if (r.overflow) throw std::overflow_error("adjugate does not fit in int64");
    return r.value;
}

inline ff::Poly2 ffCharPoly(const ff::Mat2& f) {
    const auto r = ff::LefschetzTracePass::characteristic_polynomial(f);
    if (r.overflow) throw std::overflow_error("characteristic polynomial does not fit in int64");
    return r.value;
}

inline ff::Mat2 ffPower(const ff::Mat2& f, std::uint64_t n) {
    const auto r = ff::LefschetzTracePass::power(f, n);
    if (r.overflow) throw std::overflow_error("matrix power does not fit in int64");
    return r.value;
}

}  // namespace symnet::api
