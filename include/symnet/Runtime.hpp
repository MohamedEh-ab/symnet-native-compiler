// Runtime.hpp - SymNetRuntime: executes the constant-folded execution graph.
//
//   forward()          : folded graph -> feature vector, Step decision, soft score, trace.
//                        A word of ANY length is one k x k pass (state-matrix product or
//                        mat-vec), so the run-time cost is O(1) in the word length L.
//   decideFast()       : allocation-free hot path used for nanosecond benchmarking.
//   forwardReference() : literal port of the Python forward pass over the UNFOLDED network,
//                        expanding kron(M, I) per layer.  Used for the per-layer trace and as
//                        the oracle in the folded-vs-reference equivalence checks.
//   decideUnfolded()   : O(L k^3) per-factor direct tensor path (benchmark baseline).
//   decideKron()       : O(L k^4) dense Kronecker path (benchmark baseline).
#pragma once

#include "Compiler.hpp"

#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace symnet {

struct ForwardResult {
    std::vector<double> inputVector;
    std::vector<double> finalVector;  // state fed into the first decision layer
    std::optional<bool> decision;     // nullopt when the network has no hyperplane
    std::optional<double> score;      // w . v + b of the output perceptron
    std::optional<double> softScore;  // sigmoid(sharpness * score): differentiable surrogate
    std::vector<std::pair<std::string, std::vector<double>>> trace;
};

// Dense expansion of the Linear layers (kron(M, I) in element mode), as the Python IR stores them.
struct KronProgram {
    std::vector<std::pair<std::string, DMat>> layers;
};

class SymNetRuntime {
public:
    struct Scratch {
        std::vector<double> a, b, c;
    };

    explicit SymNetRuntime(double sharpness = 10.0) : sharpness_(sharpness) {}

    // ---- folded graph ------------------------------------------------------------
    ForwardResult forward(const FoldedNetwork& net, const std::vector<double>& x, bool withTrace = true) const {
        checkInput(net.name, net.inputDim, x.size());
        ForwardResult r;
        r.inputVector = x;
        std::vector<double> v = x;
        if (withTrace) r.trace.emplace_back("input", v);

        if (net.foldedFactors > 0) {  // the single folded pass
            std::vector<double> out(v.size());
            if (net.elementMode) gemm(net.Wc.data(), v.data(), out.data(), net.N);  // X <- W_compiled X
            else gemvSquare(net.Wc.data(), v.data(), out.data(), net.N);            // v <- W_compiled v
            v = std::move(out);
            if (withTrace)
                r.trace.emplace_back(sfmt("W_compiled (%d factor%s folded)", net.foldedFactors, net.foldedFactors == 1 ? "" : "s"), v);
        }
        for (const FoldedLift& l : net.lifts) {
            const double val = l.prog.run(v.data());
            v.push_back(val);
            if (withTrace) r.trace.emplace_back("lift:" + l.name, v);
        }
        std::optional<std::vector<double>> final;
        if (net.hasBank) {
            if (net.bank.W.cols != static_cast<int>(v.size()))
                throw SymNetRuntimeError(sfmt("%s: W is (%d, %d), input is (%zu,)", net.bank.name.c_str(), net.bank.W.rows, net.bank.W.cols, v.size()));
            final = v;
            std::vector<double> h(static_cast<std::size_t>(net.bank.W.rows));
            gemv(net.bank.W.data(), v.data(), h.data(), net.bank.W.rows, net.bank.W.cols);
            for (std::size_t i = 0; i < h.size(); ++i) h[i] = (h[i] + net.bank.b[i] > 0.0) ? 1.0 : 0.0;
            v = std::move(h);
            if (withTrace) r.trace.emplace_back(net.bank.name + " [hidden step units]", v);
        }
        if (net.hasHyper) {
            if (net.hyper.weight.size() != v.size())
                throw SymNetRuntimeError(sfmt("%s: %zu weights vs feature dim %zu", net.hyper.name.c_str(), net.hyper.weight.size(), v.size()));
            if (!final) final = v;
            const double score = dot(net.hyper.weight.data(), v.data(), v.size()) + net.hyper.bias;
            r.score = score;
            r.decision = score > 0.0;  // Step(s) = 1 if s > 0 else 0
            r.softScore = sigmoid(sharpness_ * score);
            if (withTrace) r.trace.emplace_back(net.hyper.name + " [w.v+b]", std::vector<double>{score});
        }
        r.finalVector = final ? *final : v;
        return r;
    }

    // Allocation-free decision pass. Returns the Step(w.v+b) decision.
    bool decideFast(const FoldedNetwork& net, const double* x, Scratch& s, double* scoreOut = nullptr) const {
        prepare(net, s);
        double* v = s.a.data();
        if (net.foldedFactors > 0) {
            if (net.elementMode) gemm(net.Wc.data(), x, v, net.N);
            else gemvSquare(net.Wc.data(), x, v, net.N);
        } else {
            std::copy(x, x + net.inputDim, v);
        }
        return tail(net, s, scoreOut);
    }

    // ---- baselines -----------------------------------------------------------------
    // Per-factor direct tensor path: L passes of O(k^3) (no kron, but no folding).
    bool decideUnfolded(const Network& net, const FoldedNetwork& tailNet, const double* x, Scratch& s) const {
        prepare(tailNet, s);
        if (s.c.size() < s.a.size()) s.c.resize(s.a.size());
        double* cur = s.a.data();
        double* nxt = s.c.data();
        std::copy(x, x + net.inputDim, cur);
        for (const Layer& L : net.layers) {
            const auto* lin = std::get_if<LinearLayer>(&L);
            if (!lin) break;
            if (net.elementMode) gemm(lin->M.data(), cur, nxt, net.N);
            else gemvSquare(lin->M.data(), cur, nxt, net.N);
            std::swap(cur, nxt);
        }
        if (cur != s.a.data()) std::copy(cur, cur + net.inputDim, s.a.data());
        return tail(tailNet, s, nullptr);
    }

    static KronProgram expandKron(const Network& net) {
        KronProgram p;
        for (const Layer& L : net.layers) {
            const auto* lin = std::get_if<LinearLayer>(&L);
            if (!lin) break;
            p.layers.emplace_back(lin->name, net.elementMode ? kron(lin->M, DMat::identity(net.N)) : lin->M);
        }
        return p;
    }

    // Dense Kronecker path: L passes of O(k^4) mat-vecs, exactly what the Python IR executes.
    bool decideKron(const KronProgram& prog, const FoldedNetwork& tailNet, const double* x, Scratch& s) const {
        prepare(tailNet, s);
        if (s.c.size() < s.a.size()) s.c.resize(s.a.size());
        double* cur = s.a.data();
        double* nxt = s.c.data();
        std::copy(x, x + tailNet.inputDim, cur);
        for (const auto& [name, W] : prog.layers) {
            gemv(W.data(), cur, nxt, W.rows, W.cols);
            std::swap(cur, nxt);
        }
        if (cur != s.a.data()) std::copy(cur, cur + tailNet.inputDim, s.a.data());
        return tail(tailNet, s, nullptr);
    }

    // ---- literal Python-semantics forward pass over the unfolded network -------------
    ForwardResult forwardReference(const Network& net, const std::vector<double>& x) const {
        checkInput(net.name, net.inputDim, x.size());
        ForwardResult r;
        r.inputVector = x;
        std::vector<double> v = x;
        r.trace.emplace_back("input", v);
        std::optional<std::vector<double>> final;

        for (const Layer& L : net.layers) {
            if (const auto* lin = std::get_if<LinearLayer>(&L)) {
                const DMat W = net.elementMode ? kron(lin->M, DMat::identity(net.N)) : lin->M;
                if (W.cols != static_cast<int>(v.size()))
                    throw SymNetRuntimeError(sfmt("%s: W is (%d, %d), input is (%zu,)", lin->name.c_str(), W.rows, W.cols, v.size()));
                std::vector<double> out(static_cast<std::size_t>(W.rows));
                for (int i = 0; i < W.rows; ++i) out[static_cast<std::size_t>(i)] = std::inner_product(W.row(i), W.row(i) + W.cols, v.begin(), 0.0);
                v = std::move(out);
                r.trace.emplace_back(lin->name, v);
            } else if (const auto* lf = std::get_if<LiftLayer>(&L)) {
                if (lf->varNames.size() != v.size())
                    throw SymNetRuntimeError(sfmt("%s: expected %zu components, got %zu", lf->name.c_str(), lf->varNames.size(), v.size()));
                Env env;
                for (std::size_t i = 0; i < v.size(); ++i) env[lf->varNames[i]] = v[i];
                v.push_back(evalExpr(*lf->expr, env));
                r.trace.emplace_back(lf->name, v);
            } else if (const auto* bk = std::get_if<PerceptronBankLayer>(&L)) {
                if (bk->W.cols != static_cast<int>(v.size()))
                    throw SymNetRuntimeError(sfmt("%s: W is (%d, %d), input is (%zu,)", bk->name.c_str(), bk->W.rows, bk->W.cols, v.size()));
                if (!final) final = v;
                std::vector<double> h(static_cast<std::size_t>(bk->W.rows));
                for (int i = 0; i < bk->W.rows; ++i) {
                    const double s = std::inner_product(bk->W.row(i), bk->W.row(i) + bk->W.cols, v.begin(), 0.0);
                    h[static_cast<std::size_t>(i)] = (s + bk->b[static_cast<std::size_t>(i)] > 0.0) ? 1.0 : 0.0;
                }
                v = std::move(h);
                r.trace.emplace_back(bk->name + " [hidden step units]", v);
            } else {
                const auto& hp = std::get<HyperplaneLayer>(L);
                if (hp.weight.size() != v.size())
                    throw SymNetRuntimeError(sfmt("%s: %zu weights vs feature dim %zu", hp.name.c_str(), hp.weight.size(), v.size()));
                if (!final) final = v;
                const double score = std::inner_product(hp.weight.begin(), hp.weight.end(), v.begin(), 0.0) + hp.bias;
                r.score = score;
                r.decision = score > 0.0;
                r.softScore = sigmoid(sharpness_ * score);
                r.trace.emplace_back(hp.name + " [w.v+b]", std::vector<double>{score});
                break;
            }
        }
        r.finalVector = final ? *final : v;
        return r;
    }

private:
    double sharpness_;

    static double sigmoid(double t) {
        return 1.0 / (1.0 + std::exp(-std::max(-60.0, std::min(60.0, t))));
    }

    static void checkInput(const std::string& name, int expected, std::size_t got) {
        if (static_cast<int>(got) != expected)
            throw SymNetRuntimeError(sfmt("network '%s' expects input of shape (%d,), got (%zu,)", name.c_str(), expected, got));
    }

    static void prepare(const FoldedNetwork& net, Scratch& s) {
        const std::size_t total = static_cast<std::size_t>(net.featureDim());
        if (s.a.size() < total) s.a.resize(total);
        if (net.hasBank && s.b.size() < static_cast<std::size_t>(net.bank.W.rows)) s.b.resize(static_cast<std::size_t>(net.bank.W.rows));
    }

    // lifts -> (bank) -> hyperplane, on s.a which already holds the post-Linear state.
    static bool tail(const FoldedNetwork& net, Scratch& s, double* scoreOut) {
        double* v = s.a.data();
        const std::size_t n0 = static_cast<std::size_t>(net.inputDim);
        for (std::size_t i = 0; i < net.lifts.size(); ++i) v[n0 + i] = net.lifts[i].prog.run(v);
        if (!net.hasHyper) throw SymNetRuntimeError("network has no hyperplane layer");
        const double* feat = v;
        if (net.hasBank) {
            gemv(net.bank.W.data(), v, s.b.data(), net.bank.W.rows, net.bank.W.cols);
            for (int i = 0; i < net.bank.W.rows; ++i)
                s.b[static_cast<std::size_t>(i)] = (s.b[static_cast<std::size_t>(i)] + net.bank.b[static_cast<std::size_t>(i)] > 0.0) ? 1.0 : 0.0;
            feat = s.b.data();
        }
        const double score = dot(net.hyper.weight.data(), feat, net.hyper.weight.size()) + net.hyper.bias;
        if (scoreOut) *scoreOut = score;
        return score > 0.0;
    }
};

}  // namespace symnet
