// Interpreter.hpp - runs DSL text end to end (lex -> parse -> compile -> fold -> run -> report).
//
// For every job the folded graph produces the decision; the unfolded, Kronecker-expanded
// reference path (a literal port of the Python forward pass) produces the per-layer trace and
// is compared against the folded result ("xcheck") to certify semantic equivalence.
#pragma once

#include "Parser.hpp"
#include "Runtime.hpp"

#include <iostream>
#include <sstream>

namespace symnet {

struct Report {
    int passed = 0, failed = 0;    // DSL checks (identical accounting to symnet.py)
    int xChecked = 0, xFailed = 0; // folded-vs-reference equivalence checks

    void add(bool ok) { passed += ok ? 1 : 0; failed += ok ? 0 : 1; }
    void addX(bool ok) { ++xChecked; xFailed += ok ? 0 : 1; }
    Report& operator+=(const Report& o) {
        passed += o.passed; failed += o.failed; xChecked += o.xChecked; xFailed += o.xFailed;
        return *this;
    }
};

// Describe a 2x2 matrix as a complex unit e^(i*theta) when it is a rotation.
inline std::string phasorStr(const DMat& M) {
    if (!ContinuousSO2Group::isRotation(M)) return "";
    const auto z = ContinuousSO2Group::phasorOf(M);
    const double re = nz(roundTo(z.real(), 9)), im = nz(roundTo(z.imag(), 9));
    const double th = ContinuousSO2Group::angleOf(M);
    return sfmt("phasor e^(i*theta) = %g%+gi, theta = %.4g*pi rad", re, im, th / kPi);
}

inline const char* pyBool(bool b) { return b ? "True" : "False"; }

class SymNetInterpreter {
public:
    explicit SymNetInterpreter(double sharpness = 10.0, std::ostream* out = nullptr)
        : runtime_(sharpness), null_(nullptr), out_(out ? *out : null_) {}

    const SymNetRuntime& runtime() const { return runtime_; }

    // Lex + parse + compile without running anything.
    std::vector<CompiledGroup> compile(const std::string& source) const {
        return compiler_.compileProgram(Parser(tokenize(source)).parseProgram());
    }

    Report run(const std::string& source) {
        out_ << "--- DSL source " << std::string(60, '-') << "\n";
        {
            std::size_t b = source.find_first_not_of('\n');
            std::size_t e = source.find_last_not_of('\n');
            std::string body = (b == std::string::npos) ? "" : source.substr(b, e - b + 1);
            std::istringstream is(body);
            std::string line;
            int n = 1;
            while (std::getline(is, line)) out_ << sfmt("%3d | %s", n++, line.c_str()) << "\n";
        }
        const auto tokens = tokenize(source);
        const auto program = Parser(tokens).parseProgram();
        std::size_t nStmts = 0;
        for (const auto& g : program) nStmts += g.statements.size();
        out_ << sfmt("\n[lexer/parser] %zu tokens -> %zu group(s), %zu statement(s)\n", tokens.size() - 1, program.size(), nStmts);
        const auto groups = compiler_.compileProgram(program);

        Report report;
        for (const CompiledGroup& g : groups) {
            out_ << "\n[compiler] group " << g.name << "\n";
            for (const auto& d : g.declarations) out_ << "   " << d << "\n";
            for (const Task& task : g.tasks) {
                if (const auto* v = std::get_if<VerifyJob>(&task)) runVerify(*v, report);
                else if (const auto* c = std::get_if<CheckJob>(&task)) runCheck(*c, report);
                else runJob(std::get<Job>(task), report);
            }
        }
        return report;
    }

private:
    SymNetCompiler compiler_;
    SymNetRuntime runtime_;
    std::ostream null_;
    std::ostream& out_;

    // Folded result vs literal Python-semantics result.
    void crossCheck(const ForwardResult& fast, const ForwardResult& ref, Report& rep) const {
        bool ok = fast.decision == ref.decision && fast.finalVector.size() == ref.finalVector.size() &&
                  maxAbsDiff(fast.finalVector, ref.finalVector) <= 1e-9;
        if (ok && fast.score && ref.score) ok = std::fabs(*fast.score - *ref.score) <= 1e-9;
        rep.addX(ok);
    }

    void runJob(const Job& job, Report& report) {
        const ForwardResult ref = runtime_.forwardReference(job.network, job.inputVec);
        const ForwardResult res = runtime_.forward(job.folded, job.inputVec, false);
        crossCheck(res, ref, report);

        out_ << "\n[runtime] " << job.label << "\n";
        out_ << "   network: " << job.network.architecture() << "\n";
        out_ << "   folded:  " << job.folded.architecture() << "\n";
        for (const auto& [label, vec] : ref.trace) {
            std::string line = sfmt("   %-40s %s", label.c_str(), fmtVec(vec).c_str());
            if (job.matrixShape && (label == "input" || label.rfind("L", 0) == 0) &&
                static_cast<int>(vec.size()) == job.matrixShape->first * job.matrixShape->second)
                line += "   = " + fmtMat(vec.data(), job.matrixShape->first, job.matrixShape->second);
            out_ << line << "\n";
        }
        out_ << "   final point in feature space: " << fmtVec(res.finalVector) << "\n";
        if (job.matrixShape && !res.decision) {
            DMat M(job.matrixShape->first, job.matrixShape->second);
            M.a = res.finalVector;
            out_ << "   resulting group element (matrix): " << fmtMat(M.data(), M.rows, M.cols) << "\n";
            const std::string ph = phasorStr(M);
            if (!ph.empty()) out_ << "   " << ph << "\n";
        }
        if (res.decision) {
            out_ << sfmt("   Step(%+.3f) = %d  ->  decision = %s   (soft sigmoid score %.4f)", *res.score,
                         static_cast<int>(*res.decision), pyBool(*res.decision), *res.softScore) << "\n";
            if (job.expected) {
                const bool ok = *res.decision == *job.expected;
                report.add(ok);
                out_ << "   expected " << pyBool(*job.expected) << " -> " << (ok ? "PASS" : "FAIL") << "\n";
            }
        }
    }

    void runCheck(const CheckJob& cj, Report& report) {
        const ForwardResult a = runtime_.forward(cj.lhs.folded, cj.lhs.inputVec, false);
        const ForwardResult b = runtime_.forward(cj.rhs.folded, cj.rhs.inputVec, false);
        const ForwardResult ra = runtime_.forwardReference(cj.lhs.network, cj.lhs.inputVec);
        const ForwardResult rb = runtime_.forwardReference(cj.rhs.network, cj.rhs.inputVec);
        crossCheck(a, ra, report);
        crossCheck(b, rb, report);

        const auto [rows, cols] = *cj.lhs.matrixShape;
        // np.allclose(Ma, Mb, atol=1e-9): |a - b| <= atol + rtol * |b|, rtol = 1e-5
        bool ok = true;
        double maxDiff = 0.0;
        for (std::size_t i = 0; i < a.finalVector.size(); ++i) {
            const double d = std::fabs(a.finalVector[i] - b.finalVector[i]);
            maxDiff = std::max(maxDiff, d);
            if (!(d <= 1e-9 + 1e-5 * std::fabs(b.finalVector[i]))) ok = false;
        }
        report.add(ok);
        out_ << "\n[runtime] " << cj.label << "\n";
        const struct { const char* tag; const Job* job; const ForwardResult* ref; const ForwardResult* res; } sides[] = {
            {"lhs", &cj.lhs, &ra, &a}, {"rhs", &cj.rhs, &rb, &b}};
        for (const auto& sd : sides) {
            out_ << "   " << sd.tag << ": " << sd.job->network.architecture() << "\n";
            out_ << "   " << sd.tag << ": " << sd.job->folded.architecture() << "\n";
            for (std::size_t i = 1; i < sd.ref->trace.size(); ++i)
                out_ << sfmt("        %-24s %s", sd.ref->trace[i].first.c_str(), fmtMat(sd.ref->trace[i].second.data(), rows, cols).c_str()) << "\n";
            DMat M(rows, cols);
            M.a = sd.res->finalVector;
            const std::string ph = phasorStr(M);
            out_ << "        => " << fmtMat(M.data(), rows, cols) << (ph.empty() ? "" : "   [" + ph + "]") << "\n";
        }
        out_ << sfmt("   max |lhs - rhs| = %.2e  ->  %s", maxDiff, ok ? "PASS" : "FAIL") << "\n";
    }

    void runVerify(const VerifyJob& vj, Report& report) {
        out_ << "\n[runtime] " << vj.label << "\n";
        out_ << sfmt("   %-20s%-36s%7s  %-9s%-9s%s", "item", "feature point (before decision)", "score", "decision", "expected", "result") << "\n";
        int good = 0;
        for (const Job& job : vj.jobs) {
            const ForwardResult res = runtime_.forward(job.folded, job.inputVec, false);
            crossCheck(res, runtime_.forwardReference(job.network, job.inputVec), report);
            const bool ok = res.decision == job.expected;
            report.add(ok);
            good += ok ? 1 : 0;
            out_ << sfmt("   %-20s%-36s%+7.2f  %-9s%-9s%s", job.shortLabel.c_str(), fmtVec(res.finalVector).c_str(), *res.score,
                         pyBool(*res.decision), job.expected ? pyBool(*job.expected) : "None", ok ? "PASS" : "FAIL") << "\n";
        }
        out_ << sfmt("   => %d/%zu correct", good, vj.jobs.size()) << (good == static_cast<int>(vj.jobs.size()) ? "  (100% separation)" : "") << "\n";
    }
};

}  // namespace symnet
