// Compiler.hpp - SymNetCompiler: AST -> Network IR -> constant-folded execution graph.
//
//   Network (IR)        : one LinearLayer per generator factor (application order), then
//                         LiftLayers, an optional PerceptronBank and a Hyperplane.
//   foldNetwork()       : algebraic constant folding.  The chain of generator matrices is
//                         pre-multiplied at compile time, W_compiled = F1 * F2 * ... * Fn, so a
//                         word of length L costs ONE k x k pass at run time (O(1) in L).
//   KernelSynthesizer   : Novikoff perceptron cycle detection decides linear separability and
//                         searches non-linear basis lifts (|u-v|, u*v, ...) when it fails.
//   ContinuousSO2Group  : e^(i*theta) <-> 2x2 rotation matrices and sector perceptrons.
#pragma once

#include "AST.hpp"
#include "MatrixEngine.hpp"

#include <complex>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <unordered_set>
#include <variant>

namespace symnet {

// =====================================================================================
// 4. Network IR
// =====================================================================================

// v -> W v (no bias). `M` is the k x k generator. In element mode the layer acts on a state
// matrix X (X <- M X), which is exactly kron(M, I) acting on vec(X) without ever building it.
struct LinearLayer { std::string name; DMat M; };

// v -> concat(v, [f(v)])
struct LiftLayer {
    std::string name;
    std::vector<std::string> varNames;  // name bound to each current component
    ExprPtr expr;
    std::string target;
};

// v -> Step(W v + b) in {0,1}^m
struct PerceptronBankLayer {
    std::string name;
    DMat W;                // (m, N)
    std::vector<double> b;  // (m)
};

// v -> Step(w . v + b)
struct HyperplaneLayer {
    std::string name;
    std::vector<double> weight;
    double bias = 0.0;
};

using Layer = std::variant<LinearLayer, LiftLayer, PerceptronBankLayer, HyperplaneLayer>;

struct Network {
    std::string name;
    int inputDim = 0;
    int N = 0;                 // generator dimension k
    bool elementMode = false;  // state is vec(X) with X in R^(k x k)
    std::vector<Layer> layers;

    // One-line description such as: input(2) -> Lift[k1] -> R^3 -> Step(w.v+b) hyperplane:X
    std::string architecture() const {
        int dim = inputDim;
        std::string s = sfmt("input(%d)", dim);
        for (const Layer& L : layers) {
            if (const auto* lin = std::get_if<LinearLayer>(&L)) {
                const int sh = elementMode ? N * N : N;  // kron(M, I) shape in element mode
                s += sfmt(" -> Linear[%dx%d] %s", sh, sh, lin->name.c_str());
            } else if (const auto* lf = std::get_if<LiftLayer>(&L)) {
                ++dim;
                s += sfmt(" -> Lift[%s] -> R^%d", lf->target.c_str(), dim);
            } else if (const auto* bk = std::get_if<PerceptronBankLayer>(&L)) {
                s += sfmt(" -> Perceptron bank[%dx%d] %s", bk->W.rows, bk->W.cols, bk->name.c_str());
            } else {
                s += " -> Step(w.v+b) " + std::get<HyperplaneLayer>(L).name;
            }
        }
        return s;
    }
};

// ---- constant-folded execution graph ------------------------------------------------
struct FoldedLift {
    std::string name, target;
    ExprProgram prog;
};

struct FoldedNetwork {
    std::string name;
    int inputDim = 0;
    int N = 0;
    bool elementMode = false;

    DMat Wc;                  // W_compiled: product of every generator factor (k x k)
    int foldedFactors = 0;    // how many LinearLayers were collapsed into Wc
    std::vector<FoldedLift> lifts;
    bool hasBank = false;
    PerceptronBankLayer bank;
    bool hasHyper = false;
    HyperplaneLayer hyper;

    int featureDim() const { return inputDim + static_cast<int>(lifts.size()); }

    std::string architecture() const {
        int dim = inputDim;
        std::string s = sfmt("input(%d)", dim);
        if (foldedFactors > 0)
            s += sfmt(" -> Linear[%dx%d %s] W_compiled (%d factor%s folded)", N, N,
                      elementMode ? "state-matrix op" : "point op", foldedFactors, foldedFactors == 1 ? "" : "s");
        for (const FoldedLift& l : lifts) {
            ++dim;
            s += sfmt(" -> Lift[%s] -> R^%d", l.target.c_str(), dim);
        }
        if (hasBank) s += sfmt(" -> Perceptron bank[%dx%d] %s", bank.W.rows, bank.W.cols, bank.name.c_str());
        if (hasHyper) s += " -> Step(w.v+b) " + hyper.name;
        return s;
    }
};

// Constant folding pass: collapse the leading run of LinearLayers into one matrix and compile
// the remaining lift expressions into constant-folded RPN tapes.
inline FoldedNetwork foldNetwork(const Network& net) {
    FoldedNetwork f;
    f.name = net.name;
    f.inputDim = net.inputDim;
    f.N = net.N;
    f.elementMode = net.elementMode;
    f.Wc = DMat::identity(net.N);

    std::size_t i = 0;
    for (; i < net.layers.size(); ++i) {
        const auto* lin = std::get_if<LinearLayer>(&net.layers[i]);
        if (!lin) break;
        if (lin->M.rows != net.N || lin->M.cols != net.N)
            throw CompileError(sfmt("%s: generator is %dx%d but the network works in dimension %d",
                                    lin->name.c_str(), lin->M.rows, lin->M.cols, net.N));
        // layers are in application order (rightmost factor first):  Wc <- M_layer * Wc
        f.Wc = (f.foldedFactors == 0) ? lin->M : matmul(lin->M, f.Wc);
        ++f.foldedFactors;
    }
    for (; i < net.layers.size(); ++i) {
        const Layer& L = net.layers[i];
        if (const auto* lf = std::get_if<LiftLayer>(&L)) {
            if (static_cast<int>(lf->varNames.size()) != f.featureDim())
                throw CompileError(sfmt("%s: expected %zu components, got %d", lf->name.c_str(), lf->varNames.size(), f.featureDim()));
            f.lifts.push_back({lf->name, lf->target, ExprProgram::compile(*lf->expr, lf->varNames)});
        } else if (const auto* bk = std::get_if<PerceptronBankLayer>(&L)) {
            f.hasBank = true;
            f.bank = *bk;
        } else if (const auto* hp = std::get_if<HyperplaneLayer>(&L)) {
            f.hasHyper = true;
            f.hyper = *hp;
            break;  // the hyperplane terminates the network
        } else {
            throw CompileError("a Linear layer after a non-linear layer cannot be constant-folded");
        }
    }
    return f;
}

// =====================================================================================
// 4b. Continuous Lie group SO(2) (complex phasors)
// =====================================================================================

struct SectorSpec {
    DMat W;                    // 2 x 2 hidden half-plane normals
    std::vector<double> b;     // 2
    std::vector<double> outW;  // 2
    double outB = 0.0;
    std::string mode;          // "AND" (width <= pi) or "OR" (width > pi)
};

struct ContinuousSO2Group {
    static constexpr double TOL = 1e-9;

    static std::complex<double> phasor(double theta) { return {std::cos(theta), std::sin(theta)}; }

    // Raw (uncleaned) fixed-size rotation matrix, used for machine-precision composition checks.
    static Matrix2x2 rotation(double theta) {
        const double c = std::cos(theta), s = std::sin(theta);
        Matrix2x2 m;
        m(0, 0) = c; m(0, 1) = -s; m(1, 0) = s; m(1, 1) = c;
        return m;
    }

    static DMat matrix(double theta) {
        const double c = std::cos(theta), s = std::sin(theta);
        DMat m(2, 2);
        m(0, 0) = c; m(0, 1) = -s; m(1, 0) = s; m(1, 1) = c;
        return cleanMatrix(m);
    }

    static DMat matrixFromPhasor(std::complex<double> z) {
        DMat m(2, 2);
        m(0, 0) = z.real(); m(0, 1) = -z.imag(); m(1, 0) = z.imag(); m(1, 1) = z.real();
        return cleanMatrix(m);
    }

    static bool isRotation(const DMat& M) {
        return M.rows == 2 && M.cols == 2 && std::fabs(M(0, 0) - M(1, 1)) < TOL &&
               std::fabs(M(0, 1) + M(1, 0)) < TOL &&
               std::fabs(M(0, 0) * M(0, 0) + M(1, 0) * M(1, 0) - 1.0) < TOL;
    }
    static std::complex<double> phasorOf(const DMat& M) { return {M(0, 0), M(1, 0)}; }
    static double angleOf(const DMat& M) { return std::atan2(M(1, 0), M(0, 0)); }

    // Analytic ground truth: is theta in the closed counter-clockwise arc [lo, hi]?
    static bool inSector(double theta, double lo, double hi) {
        const double twoPi = 2.0 * kPi;
        const double rel = pyMod(theta - lo, twoPi);
        return rel <= (hi - lo) + TOL || rel >= twoPi - TOL;
    }

    // Closed sector [lo, hi] -> two half-plane units + AND/OR output perceptron.
    static SectorSpec sectorPerceptrons(double lo, double hi, double eps = 1e-9) {
        const double width = hi - lo;
        if (!(0.0 < width && width < 2.0 * kPi - TOL))
            throw CompileError(sfmt("sector width %.4f rad must lie in (0, 2*pi)", width));
        SectorSpec sp;
        sp.W = DMat(2, 2);
        sp.W(0, 0) = -std::sin(lo); sp.W(0, 1) = std::cos(lo);   // n1
        sp.W(1, 0) = std::sin(hi);  sp.W(1, 1) = -std::cos(hi);  // n2
        sp.b = {eps, eps};
        sp.outW = {1.0, 1.0};
        if (width <= kPi + TOL) { sp.outB = -1.5; sp.mode = "AND"; }
        else                    { sp.outB = -0.5; sp.mode = "OR"; }
        return sp;
    }
};

// =====================================================================================
// 4c. Automated kernel lift synthesizer
// =====================================================================================

// Row-major table of feature vectors (one row per labelled point).
struct Table {
    std::size_t rows = 0, cols = 0;
    std::vector<double> d;

    Table() = default;
    Table(std::size_t r, std::size_t c, double fill = 0.0) : rows(r), cols(c), d(r * c, fill) {}
    double& at(std::size_t i, std::size_t j) { return d[i * cols + j]; }
    double at(std::size_t i, std::size_t j) const { return d[i * cols + j]; }
    const double* row(std::size_t i) const { return d.data() + i * cols; }

    std::vector<double> column(std::size_t j) const {
        std::vector<double> c(rows);
        for (std::size_t i = 0; i < rows; ++i) c[i] = at(i, j);
        return c;
    }
    Table withColumn(const std::vector<double>& col) const {
        Table t(rows, cols + 1);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) t.at(i, j) = at(i, j);
            t.at(i, cols) = col[i];
        }
        return t;
    }
    Table selectCols(const std::vector<std::size_t>& idx) const {
        Table t(rows, idx.size());
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < idx.size(); ++j) t.at(i, j) = at(i, idx[j]);
        return t;
    }
};

using Labels = std::vector<char>;  // 1 = positive class

struct SynthesisResult {
    bool separableBefore = false;
    std::vector<LiftDecl> lifts;
    std::vector<double> weight;  // hyperplane over [original features, lifts...]
    double bias = 0.0;
    std::vector<std::string> log;
};

struct VecHash {
    std::size_t operator()(const std::vector<double>& v) const {
        std::size_t h = 1469598103934665603ull;
        for (double x : v) h ^= std::hash<double>{}(x) + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        return h;
    }
};

inline std::vector<double> roundedKey(const double* v, std::size_t n, int decimals) {
    std::vector<double> k(n);
    for (std::size_t i = 0; i < n; ++i) k[i] = nz(roundTo(v[i], decimals));
    return k;
}

// Apply a list of lifts row by row; returns the widened table (names are extended in place).
inline Table featurize(const Table& X, std::vector<std::string>& names, const std::vector<LiftDecl>& lifts) {
    Table F = X;
    for (const LiftDecl& ld : lifts) {
        const ExprProgram prog = ExprProgram::compile(*ld.expr, names);
        std::vector<double> col(F.rows);
        for (std::size_t i = 0; i < F.rows; ++i) col[i] = prog.run(F.row(i));
        F = F.withColumn(col);
        names.push_back(ld.name);
    }
    return F;
}

// itertools.combinations(range(n), k) in lexicographic order; fn returns true to stop early.
template <class Fn>
bool forEachCombination(std::size_t n, std::size_t k, Fn&& fn) {
    if (k > n) return false;
    std::vector<std::size_t> idx(k);
    for (std::size_t i = 0; i < k; ++i) idx[i] = i;
    while (true) {
        if (fn(idx)) return true;
        long i = static_cast<long>(k) - 1;
        while (i >= 0 && idx[static_cast<std::size_t>(i)] == n - k + static_cast<std::size_t>(i)) --i;
        if (i < 0) return false;
        ++idx[static_cast<std::size_t>(i)];
        for (std::size_t j = static_cast<std::size_t>(i) + 1; j < k; ++j) idx[j] = idx[j - 1] + 1;
    }
}

class KernelSynthesizer {
public:
    using Hyperplane = std::pair<std::vector<double>, double>;

    explicit KernelSynthesizer(int maxLifts = 3, int maxSparseSupport = 3, int maxUpdates = 5000)
        : maxLifts_(maxLifts), maxSparse_(maxSparseSupport), maxUpdates_(maxUpdates) {}

    // Perceptron algorithm with cycle detection. Returns augmented weights (w..., b) if the
    // data are linearly separable. If a weight vector ever repeats after an update, the
    // perceptron cycling theorem certifies NON-separability (separable runs strictly increase
    // w . w* with every update, so they can never revisit a state).
    std::optional<std::vector<double>> perceptronSeparable(const Table& F, const Labels& pos) const {
        const std::size_t n = F.rows, m = F.cols + 1;
        std::vector<double> A(n * m), y(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < F.cols; ++j) A[i * m + j] = F.at(i, j);
            A[i * m + F.cols] = 1.0;
            y[i] = pos[i] ? 1.0 : -1.0;
        }
        std::vector<double> w(m, 0.0);
        std::unordered_set<std::vector<double>, VecHash> seen;
        seen.insert(roundedKey(w.data(), m, 8));
        int updates = 0;
        while (updates <= maxUpdates_) {
            bool changed = false;
            for (std::size_t i = 0; i < n; ++i) {
                const double* Ai = &A[i * m];
                if (y[i] * dot(Ai, w.data(), m) <= 0.0) {
                    for (std::size_t j = 0; j < m; ++j) w[j] += y[i] * Ai[j];
                    ++updates;
                    if (!seen.insert(roundedKey(w.data(), m, 8)).second) return std::nullopt;  // cycle
                    changed = true;
                }
            }
            if (!changed) return w;
        }
        return std::nullopt;  // no convergence within the update budget
    }

    // Rescale so the margin gap is exactly 1 and centre the bias in it.
    static std::optional<Hyperplane> normalise(const Table& F, const Labels& pos,
                                               const std::vector<std::size_t>& cols,
                                               const std::vector<double>& wSub) {
        std::vector<double> w(F.cols, 0.0);
        for (std::size_t k = 0; k < cols.size(); ++k) w[cols[k]] = wSub[k];
        auto minMax = [&](const std::vector<double>& ww, double& minPos, double& maxNeg) {
            minPos = std::numeric_limits<double>::infinity();
            maxNeg = -std::numeric_limits<double>::infinity();
            for (std::size_t i = 0; i < F.rows; ++i) {
                const double s = dot(F.row(i), ww.data(), F.cols);
                if (pos[i]) minPos = std::min(minPos, s);
                else maxNeg = std::max(maxNeg, s);
            }
        };
        double minPos, maxNeg;
        minMax(w, minPos, maxNeg);
        const double gap = minPos - maxNeg;
        if (gap <= 1e-12) return std::nullopt;
        for (double& v : w) v /= gap;
        minMax(w, minPos, maxNeg);
        const double b = -(minPos + maxNeg) / 2.0;
        for (double& v : w) v = nz(roundTo(v, 10));
        return Hyperplane{w, nz(roundTo(b, 10))};
    }

    // Separating hyperplane with the fewest non-zero weights, or nullopt.
    std::optional<Hyperplane> cleanHyperplane(const Table& F, const Labels& pos) const {
        const bool allPos = std::all_of(pos.begin(), pos.end(), [](char c) { return c != 0; });
        const bool allNeg = std::none_of(pos.begin(), pos.end(), [](char c) { return c != 0; });
        if (allPos || allNeg) return Hyperplane{std::vector<double>(F.cols, 0.0), allPos ? 0.5 : -0.5};
        if (!perceptronSeparable(F, pos)) return std::nullopt;
        const std::size_t nFeat = F.cols;
        for (std::size_t size = 1; size <= std::min<std::size_t>(nFeat, static_cast<std::size_t>(maxSparse_)); ++size) {
            std::optional<Hyperplane> found;
            forEachCombination(nFeat, size, [&](const std::vector<std::size_t>& cols) {
                auto sol = perceptronSeparable(F.selectCols(cols), pos);
                if (!sol) return false;
                auto out = normalise(F, pos, cols, std::vector<double>(sol->begin(), sol->end() - 1));
                if (!out) return false;
                found = std::move(out);
                return true;
            });
            if (found) return found;
        }
        auto sol = perceptronSeparable(F, pos);
        std::vector<std::size_t> all(nFeat);
        for (std::size_t i = 0; i < nFeat; ++i) all[i] = i;
        return normalise(F, pos, all, std::vector<double>(sol->begin(), sol->end() - 1));
    }

    // Candidate basis functions on the current coordinates (library order matters for ties).
    static std::vector<ExprPtr> candidateLibrary(const std::vector<std::string>& names) {
        std::vector<ExprPtr> V;
        for (const auto& n : names) V.push_back(mkVar(n));
        const std::size_t n = V.size();
        std::vector<ExprPtr> lib;
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) pairs.emplace_back(i, j);
        for (auto [i, j] : pairs) lib.push_back(mkBin('*', V[i], V[j]));                      // u*v
        for (auto [i, j] : pairs) lib.push_back(mkCall("abs", mkBin('-', V[i], V[j])));       // |u-v|
        for (auto [i, j] : pairs) lib.push_back(mkCall("abs", mkBin('+', V[i], V[j])));       // |u+v|
        for (std::size_t i = 0; i < n; ++i) lib.push_back(mkBin('*', V[i], V[i]));            // squares
        for (std::size_t i = 0; i < n; ++i) lib.push_back(mkCall("abs", V[i]));               // |u|
        if (n <= 5)
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = i + 1; j < n; ++j)
                    for (std::size_t k = j + 1; k < n; ++k)
                        lib.push_back(mkBin('*', mkBin('*', V[i], V[j]), V[k]));              // cubic
        return lib;
    }

    // Evaluate the library on the data; drop constants and duplicates.
    static std::vector<std::pair<ExprPtr, std::vector<double>>> candidateColumns(
        const Table& F, const std::vector<std::string>& names, const std::vector<ExprPtr>& library) {
        std::set<std::vector<double>> seen;
        for (std::size_t j = 0; j < F.cols; ++j) {
            std::vector<double> col = F.column(j);
            seen.insert(roundedKey(col.data(), col.size(), 9));
            for (double& v : col) v = -v;
            seen.insert(roundedKey(col.data(), col.size(), 9));
        }
        std::vector<std::pair<ExprPtr, std::vector<double>>> kept;
        for (const ExprPtr& e : library) {
            const ExprProgram prog = ExprProgram::compile(*e, names);
            std::vector<double> col(F.rows);
            for (std::size_t i = 0; i < F.rows; ++i) col[i] = prog.run(F.row(i));
            const auto [mn, mx] = std::minmax_element(col.begin(), col.end());
            if (*mx - *mn < 1e-12) continue;
            auto key = roundedKey(col.data(), col.size(), 9);
            if (!seen.insert(key).second) continue;
            kept.emplace_back(e, std::move(col));
        }
        return kept;
    }

    SynthesisResult synthesize(const Table& F0, const Labels& pos, const std::vector<std::string>& names0,
                               const std::string& label = "") const {
        const std::size_t nPts = F0.rows, dim = F0.cols;
        const long nPos = std::count(pos.begin(), pos.end(), 1);
        SynthesisResult res;
        res.log.push_back(sfmt("synthesize %s: %zu points in R^%zu, %ld positive / %ld negative", label.c_str(),
                               nPts, dim, nPos, static_cast<long>(nPts) - nPos));

        // identical points with different labels can never be separated by any lift
        std::map<std::vector<double>, bool> seenLbl;
        for (std::size_t i = 0; i < nPts; ++i) {
            auto key = roundedKey(F0.row(i), dim, 9);
            auto it = seenLbl.find(key);
            if (it != seenLbl.end() && it->second != (pos[i] != 0))
                throw CompileError("the same point appears with two different labels; no feature lift can separate it");
            seenLbl[key] = pos[i] != 0;
        }

        if (auto sol = cleanHyperplane(F0, pos)) {
            res.log.push_back(sfmt("linearly separable already in R^%zu: no lift needed", dim));
            res.separableBefore = true;
            res.weight = sol->first;
            res.bias = sol->second;
            return res;
        }
        res.log.push_back(sfmt("perceptron cycling certificate: NOT linearly separable in R^%zu", dim));

        const auto library = candidateLibrary(names0);
        const auto cols = candidateColumns(F0, names0, library);
        res.log.push_back(sfmt("candidate basis functions after de-duplication: %zu", cols.size()));

        std::set<std::string> used(names0.begin(), names0.end());
        for (int k = 1; k <= maxLifts_; ++k) {
            using Cost = std::pair<int, std::vector<std::size_t>>;
            std::optional<std::tuple<Cost, std::vector<std::size_t>, Hyperplane>> best;
            long tried = 0, winners = 0;
            forEachCombination(cols.size(), static_cast<std::size_t>(k), [&](const std::vector<std::size_t>& combo) {
                ++tried;
                Table F = F0;
                for (std::size_t ci : combo) F = F.withColumn(cols[ci].second);
                auto out = cleanHyperplane(F, pos);
                if (!out) return false;
                ++winners;
                int nnz = 0;
                for (double v : out->first) nnz += (v != 0.0);
                Cost cost{nnz, combo};
                if (!best || cost < std::get<0>(*best)) best = std::make_tuple(cost, combo, *out);
                return false;
            });
            res.log.push_back(sfmt("k=%d: tried %ld lift set(s), %ld make the data separable", k, tried, winners));
            if (best) {
                const auto& [cost, combo, hp] = *best;
                (void)cost;
                std::vector<LiftDecl> lifts;
                int idx = 1;
                for (std::size_t ci : combo) {
                    std::string nm = "k" + std::to_string(idx++);
                    while (used.count(nm)) nm += "_";
                    used.insert(nm);
                    lifts.push_back(LiftDecl{nm, cols[ci].first, 0});
                }
                finish(res, F0, names0, pos, lifts, hp, dim, static_cast<std::size_t>(k));
                return res;
            }
        }

        // Flat search failed: try composed lifts, k2 built on top of k1 (e.g. |k1 - w|).
        {
            std::optional<std::tuple<std::pair<int, long>, ExprPtr, ExprPtr, Hyperplane>> best;
            long tried = 0;
            for (const auto& [e1, c1] : cols) {
                const Table F1 = F0.withColumn(c1);
                std::vector<std::string> names1 = names0;
                names1.push_back("k1");
                for (const auto& [e2, c2] : candidateColumns(F1, names1, candidateLibrary(names1))) {
                    ++tried;
                    auto out = cleanHyperplane(F1.withColumn(c2), pos);
                    if (!out) continue;
                    int nnz = 0;
                    for (double v : out->first) nnz += (v != 0.0);
                    const std::pair<int, long> cost{nnz, tried};
                    if (!best || cost < std::get<0>(*best)) best = std::make_tuple(cost, e1, e2, *out);
                }
            }
            res.log.push_back(sfmt("composed search (k2 built on k1): tried %ld pairs", tried));
            if (best) {
                const auto& [cost, e1, e2, hp] = *best;
                (void)cost;
                std::vector<LiftDecl> lifts{LiftDecl{"k1", e1, 0}, LiftDecl{"k2", e2, 0}};
                finish(res, F0, names0, pos, lifts, hp, dim, 2);
                return res;
            }
        }
        throw CompileError(sfmt("no separating lift found with at most %d basis functions (flat) or two composed levels", maxLifts_));
    }

private:
    int maxLifts_, maxSparse_, maxUpdates_;

    static void finish(SynthesisResult& res, const Table& F0, const std::vector<std::string>& names0,
                       const Labels& pos, const std::vector<LiftDecl>& lifts, const Hyperplane& hp,
                       std::size_t dim, std::size_t k) {
        std::vector<std::string> names = names0;
        const Table Ff = featurize(F0, names, lifts);
        for (std::size_t i = 0; i < Ff.rows; ++i) {
            const bool got = dot(Ff.row(i), hp.first.data(), Ff.cols) + hp.second > 0.0;
            if (got != (pos[i] != 0))
                throw CompileError("internal error: synthesized hyperplane does not reproduce the labels");
        }
        std::string desc;
        for (std::size_t i = 0; i < lifts.size(); ++i) {
            if (i) desc += ", ";
            desc += lifts[i].name + " = " + fmtExpr(*lifts[i].expr);
        }
        res.log.push_back(sfmt("minimal feature map Phi: R^%zu -> R^%zu  with  %s", dim, dim + k, desc.c_str()));
        res.log.push_back(sfmt("separating hyperplane: w = %s, b = %s", fmtVec(hp.first).c_str(), fmtG(hp.second).c_str()));
        res.separableBefore = false;
        res.lifts = lifts;
        res.weight = hp.first;
        res.bias = hp.second;
    }
};

// =====================================================================================
// 5. Compiler
// =====================================================================================

struct CompiledSubgroup {
    std::string name;
    std::vector<std::string> generatorNames;
    std::vector<std::pair<std::vector<std::string>, DMat>> elements;  // (shortest word, matrix)
    std::set<std::vector<double>> keys;

    bool contains(const DMat& M) const { return keys.count(matrixKey(M)) != 0; }
};

struct CompiledClassifier {
    std::string name;
    std::vector<double> weight;
    double bias = 0.0;
    std::optional<std::vector<LiftDecl>> lifts;      // snapshot of lifts (synthesized classifiers)
    std::optional<PerceptronBankLayer> bank;         // hidden units (sector classifiers)
    std::optional<std::pair<double, double>> sector;
};

// One runnable query: the IR network, its constant-folded form, and the vector fed into it.
struct Job {
    std::string kind;  // "eval" | "classify"
    std::string label;
    Network network;
    FoldedNetwork folded;
    std::vector<double> inputVec;
    std::optional<std::pair<int, int>> matrixShape;  // set in element mode for pretty printing
    std::optional<bool> expected;
    std::string shortLabel;
};

struct VerifyJob { std::string label; std::vector<Job> jobs; };
struct CheckJob { std::string label; Job lhs, rhs; };
using Task = std::variant<Job, VerifyJob, CheckJob>;

struct CompiledGroup {
    std::string name;
    std::map<std::string, DMat> generators;
    std::map<std::string, std::tuple<std::string, std::string, ExprPtr>> paramGenerators;  // name -> (param, kind, body)
    std::vector<LiftDecl> liftDecls;
    std::map<std::string, CompiledSubgroup> subgroups;
    std::map<std::string, CompiledClassifier> classifiers;
    std::map<std::string, std::pair<Table, Labels>> spaces;
    std::vector<std::string> declarations;
    std::vector<Task> tasks;

    std::optional<int> dim() const {
        if (!generators.empty()) return generators.begin()->second.rows;
        if (!paramGenerators.empty()) return 2;
        return std::nullopt;
    }
};

inline std::string pyListRepr(const std::vector<std::string>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + pyRepr(v[i]);
    return s + "]";
}

class SymNetCompiler {
public:
    static constexpr std::size_t MAX_GROUP_ORDER = 2048;

    explicit SymNetCompiler(KernelSynthesizer synth = KernelSynthesizer()) : synth_(std::move(synth)) {}

    std::vector<CompiledGroup> compileProgram(const std::vector<GroupDecl>& program) const {
        std::vector<CompiledGroup> out;
        for (const GroupDecl& g : program) out.push_back(compileGroup(g));
        return out;
    }

    CompiledGroup compileGroup(const GroupDecl& decl) const {
        CompiledGroup ctx;
        ctx.name = decl.name;
        for (const Statement& st : decl.statements) {
            try {
                compileStatement(ctx, st);
            } catch (const SymNetError& exc) {
                throw CompileError(sfmt("group %s, line %d: %s", decl.name.c_str(), statementLine(st), exc.what()));
            }
        }
        return ctx;
    }

private:
    KernelSynthesizer synth_;

    struct Resolved { std::string label; DMat M; };

    void compileStatement(CompiledGroup& ctx, const Statement& st) const {
        std::visit(
            overloaded{
                [&](const GenDecl& s) { compileGen(ctx, s); },
                [&](const LiftDecl& s) {
                    ctx.liftDecls.push_back(s);
                    ctx.declarations.push_back("lift " + s.name + " = " + fmtExpr(*s.expr) +
                                               "  ->  non-linear feature layer appended (R^N -> R^(N+1))");
                },
                [&](const SubgroupDecl& s) { compileSubgroup(ctx, s); },
                [&](const SeparateDecl& s) { compileSeparate(ctx, s); },
                [&](const SectorDecl& s) { compileSector(ctx, s); },
                [&](const SpaceDecl& s) { compileSpace(ctx, s); },
                [&](const SynthesizeStmt& s) { compileSynthesize(ctx, s); },
                [&](const EvalStmt& s) { ctx.tasks.push_back(compileEval(ctx, s)); },
                [&](const ClassifyStmt& s) { ctx.tasks.push_back(compileClassify(ctx, s)); },
                [&](const VerifyStmt& s) { ctx.tasks.push_back(compileVerify(ctx, s)); },
                [&](const CheckStmt& s) { ctx.tasks.push_back(compileCheck(ctx, s)); },
                [&](const SweepStmt& s) { ctx.tasks.push_back(compileSweep(ctx, s)); }},
            st);
    }

    // ---- generators ---------------------------------------------------------------
    // 2x2 matrix for an angle. rotation/reflection: degrees; phasor: radians.
    static DMat angleMatrix(const std::string& kind, double value) {
        if (kind == "phasor") return ContinuousSO2Group::matrix(value);
        const double a = radians(value);
        if (kind == "rotation") return ContinuousSO2Group::matrix(a);
        DMat m(2, 2);
        m(0, 0) = std::cos(2 * a); m(0, 1) = std::sin(2 * a);
        m(1, 0) = std::sin(2 * a); m(1, 1) = -std::cos(2 * a);
        return cleanMatrix(m);
    }

    static void compileGen(CompiledGroup& ctx, const GenDecl& st) {
        if (ctx.generators.count(st.name) || ctx.paramGenerators.count(st.name))
            throw CompileError("generator '" + st.name + "' already defined");
        if (st.param) {
            std::set<std::string> extra = freeVariables(*st.angle);
            extra.erase(*st.param);
            if (!extra.empty()) {
                std::vector<std::string> ex(extra.begin(), extra.end());
                throw CompileError("angle expression uses unknown variable(s) " + pyListRepr(ex));
            }
            if (ctx.dim() && *ctx.dim() != 2)
                throw CompileError(sfmt("parametrised generator is 2x2 but the group works in dimension %d", *ctx.dim()));
            ctx.paramGenerators[st.name] = {*st.param, st.kind, st.angle};
            ctx.declarations.push_back(sfmt(
                "gen %s(%s) = %s(%s)  ->  continuous family of 2x2 Linear weights (instantiated per use, e.g. %s(pi/4))",
                st.name.c_str(), st.param->c_str(), st.kind.c_str(), fmtExpr(*st.angle).c_str(), st.name.c_str()));
            return;
        }
        DMat M;
        if (st.kind == "matrix") {
            std::set<std::size_t> lens;
            for (const auto& r : st.rows) lens.insert(r.size());
            if (lens.size() != 1) throw CompileError("matrix rows have different lengths");
            const int rows = static_cast<int>(st.rows.size()), cols = static_cast<int>(st.rows[0].size());
            if (rows != cols) throw CompileError(sfmt("generator matrix must be square, got (%d, %d)", rows, cols));
            M = DMat(rows, cols);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j) M(i, j) = evalExpr(*st.rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)], Env{});
            M = cleanMatrix(M);
        } else {
            M = angleMatrix(st.kind, evalExpr(*st.angle, Env{}));
        }
        if (ctx.dim() && M.rows != *ctx.dim())
            throw CompileError(sfmt("generator '%s' is %dx%d but the group works in dimension %d", st.name.c_str(), M.rows, M.rows, *ctx.dim()));
        ctx.declarations.push_back("gen " + st.name + " = " + st.kind + "  ->  Linear weight W = " + fmtMat(M.data(), M.rows, M.cols));
        ctx.generators[st.name] = std::move(M);
    }

    // ---- subgroups, classifiers ---------------------------------------------------
    void compileSubgroup(CompiledGroup& ctx, const SubgroupDecl& st) const {
        if (ctx.subgroups.count(st.name)) throw CompileError("subgroup '" + st.name + "' already defined");
        for (const auto& g : st.generators)
            if (!ctx.generators.count(g)) throw CompileError("unknown (non-parametrised) generator '" + g + "' in span()");
        CompiledSubgroup sg;
        sg.name = st.name;
        sg.generatorNames = st.generators;
        sg.elements = closure(ctx, st.generators);
        for (const auto& [w, m] : sg.elements) sg.keys.insert(matrixKey(m));
        std::string gens;
        for (std::size_t i = 0; i < st.generators.size(); ++i) gens += (i ? ", " : "") + st.generators[i];
        ctx.declarations.push_back(sfmt("subgroup %s = span(%s)  ->  %zu elements (closure by BFS)", st.name.c_str(), gens.c_str(), sg.elements.size()));
        ctx.subgroups[st.name] = std::move(sg);
    }

    static void compileSeparate(CompiledGroup& ctx, const SeparateDecl& st) {
        if (ctx.classifiers.count(st.name)) throw CompileError("classifier '" + st.name + "' already defined");
        CompiledClassifier c;
        c.name = st.name;
        for (const auto& e : st.weight) c.weight.push_back(evalExpr(*e, Env{}));
        c.bias = evalExpr(*st.bias, Env{});
        ctx.declarations.push_back(sfmt("separate %s  ->  Hyperplane weight=%s, bias=%s", st.name.c_str(), fmtVec(c.weight).c_str(), fmtG(c.bias).c_str()));
        ctx.classifiers[st.name] = std::move(c);
    }

    static void compileSector(CompiledGroup& ctx, const SectorDecl& st) {
        if (ctx.classifiers.count(st.name)) throw CompileError("classifier '" + st.name + "' already defined");
        const double lo = evalExpr(*st.lo, Env{}), hi = evalExpr(*st.hi, Env{});
        const SectorSpec spec = ContinuousSO2Group::sectorPerceptrons(lo, hi);
        CompiledClassifier c;
        c.name = st.name;
        c.weight = spec.outW;
        c.bias = spec.outB;
        c.bank = PerceptronBankLayer{"sector:" + st.name, spec.W, spec.b};
        c.sector = std::make_pair(lo, hi);
        ctx.declarations.push_back(sfmt(
            "sector %s = [%.4g*pi, %.4g*pi] rad  ->  2 hyperplane units n1=%s, n2=%s + %s output (w=%s, b=%s)",
            st.name.c_str(), lo / kPi, hi / kPi, fmtVec(spec.W.row(0), 2).c_str(), fmtVec(spec.W.row(1), 2).c_str(),
            spec.mode.c_str(), fmtVec(spec.outW).c_str(), fmtG(spec.outB).c_str()));
        ctx.classifiers[st.name] = std::move(c);
    }

    // ---- finite state spaces + kernel synthesis -----------------------------------
    static void compileSpace(CompiledGroup& ctx, const SpaceDecl& st) {
        if (ctx.spaces.count(st.name)) throw CompileError("space '" + st.name + "' already defined");
        const std::size_t dim = st.points[0].size();
        Table X(st.points.size(), dim);
        for (std::size_t i = 0; i < st.points.size(); ++i) {
            if (st.points[i].size() != dim) throw CompileError("space points have different dimensions");
            for (std::size_t j = 0; j < dim; ++j) X.at(i, j) = evalExpr(*st.points[i][j], Env{});
        }
        Labels y;
        for (int c : st.labels) y.push_back(c != 0 ? 1 : 0);
        ctx.declarations.push_back(sfmt("space %s  ->  %zu labelled points in R^%zu", st.name.c_str(), X.rows, X.cols));
        ctx.spaces[st.name] = {std::move(X), std::move(y)};
    }

    DMat wordMatrix(const CompiledGroup& ctx, const Word& word) const {
        const auto factors = resolveWord(ctx, word);
        DMat M = DMat::identity(dimFor(ctx, factors));
        for (const Resolved& f : factors) M = matmul(M, f.M);
        return cleanMatrix(M);
    }

    void compileSynthesize(CompiledGroup& ctx, const SynthesizeStmt& st) const {
        if (ctx.classifiers.count(st.name)) throw CompileError("classifier '" + st.name + "' already defined");
        const std::vector<std::pair<std::vector<std::string>, DMat>>* elements = nullptr;
        Table X;
        Labels pos;
        std::string what;
        if (st.space) {
            auto it = ctx.spaces.find(*st.space);
            if (it == ctx.spaces.end()) throw CompileError("unknown space '" + *st.space + "'");
            X = it->second.first;
            pos = it->second.second;
            what = "space " + *st.space;
        } else {
            for (const std::string* nm : {&*st.subgroup, &*st.ambient})
                if (!ctx.subgroups.count(*nm)) throw CompileError("unknown subgroup '" + *nm + "'");
            const CompiledSubgroup& sub = ctx.subgroups.at(*st.subgroup);
            const CompiledSubgroup& amb = ctx.subgroups.at(*st.ambient);
            elements = &amb.elements;
            const std::size_t dim = static_cast<std::size_t>(amb.elements[0].second.a.size());
            X = Table(amb.elements.size(), dim);
            for (std::size_t i = 0; i < amb.elements.size(); ++i)
                for (std::size_t j = 0; j < dim; ++j) X.at(i, j) = amb.elements[i].second.a[j];
            std::set<std::vector<double>> target;
            if (st.coset) {
                const DMat g = wordMatrix(ctx, *st.coset);
                for (const auto& [w, h] : sub.elements) target.insert(matrixKey(cleanMatrix(matmul(g, h))));
                what = "coset " + fmtWordItems(*st.coset) + "*" + sub.name + " inside " + amb.name;
            } else {
                target = sub.keys;
                what = "subgroup " + sub.name + " inside " + amb.name;
            }
            for (const auto& [w, m] : amb.elements) pos.push_back(target.count(matrixKey(m)) ? 1 : 0);
        }

        std::vector<std::string> names0 = defaultNames(static_cast<int>(X.cols));
        const Table F0 = featurize(X, names0, ctx.liftDecls);
        const SynthesisResult res = synth_.synthesize(F0, pos, names0, st.name + " (" + what + ")");
        CompiledClassifier clf;
        clf.name = st.name;
        clf.weight = res.weight;
        clf.bias = res.bias;
        std::vector<LiftDecl> all = ctx.liftDecls;
        all.insert(all.end(), res.lifts.begin(), res.lifts.end());
        clf.lifts = std::move(all);
        for (const auto& line : res.log) ctx.declarations.push_back("[synth] " + line);
        ctx.declarations.push_back(sfmt("[synth] injected %zu LiftLayer(s) + Hyperplane layer into the compiled graph for '%s'", res.lifts.size(), st.name.c_str()));
        ctx.classifiers[st.name] = clf;

        std::vector<Job> jobs;
        if (!elements) {
            for (std::size_t i = 0; i < X.rows; ++i) {
                std::vector<double> row(X.row(i), X.row(i) + X.cols);
                const std::string shortLbl = fmtVec(row);
                jobs.push_back(classifyJob(ctx, clf, std::nullopt, row, pos[i] != 0, shortLbl));
            }
        } else {
            for (std::size_t i = 0; i < elements->size(); ++i) {
                Word items;
                for (const auto& g : (*elements)[i].first) items.push_back(Factor{g, 1, nullptr});
                jobs.push_back(classifyJob(ctx, clf, items, std::nullopt, pos[i] != 0, std::nullopt));
            }
        }
        const std::size_t n = jobs.size();
        ctx.tasks.push_back(VerifyJob{sfmt("synthesized classifier %s: separation of all %zu points", st.name.c_str(), n), std::move(jobs)});
    }

    // ---- group closure (BFS by right-multiplication) ------------------------------
    std::vector<std::pair<std::vector<std::string>, DMat>> closure(const CompiledGroup& ctx,
                                                                   const std::vector<std::string>& gens) const {
        const int N = ctx.generators.at(gens[0]).rows;
        const DMat identity = DMat::identity(N);
        std::set<std::vector<double>> seen{matrixKey(identity)};
        std::vector<std::pair<std::vector<std::string>, DMat>> elements{{{}, identity}};
        std::vector<std::pair<std::vector<std::string>, DMat>> frontier = elements;
        while (!frontier.empty()) {
            decltype(frontier) next;
            for (const auto& [word, M] : frontier)
                for (const auto& g : gens) {
                    DMat P = cleanMatrix(matmul(M, ctx.generators.at(g)));
                    if (seen.insert(matrixKey(P)).second) {
                        std::vector<std::string> w2 = word;
                        w2.push_back(g);
                        elements.emplace_back(w2, P);
                        next.emplace_back(std::move(w2), std::move(P));
                        if (elements.size() > MAX_GROUP_ORDER)
                            throw CompileError(sfmt("subgroup is infinite or larger than %zu elements", MAX_GROUP_ORDER));
                    }
                }
            frontier = std::move(next);
        }
        return elements;
    }

    // ---- words -> layers ------------------------------------------------------------
    // Expand exponents / arguments and look up matrices (left-to-right order).
    static std::vector<Resolved> resolveWord(const CompiledGroup& ctx, const Word& word) {
        std::vector<Resolved> factors;
        for (const Factor& f : word) {
            std::string label = f.name;
            DMat M;
            auto pg = ctx.paramGenerators.find(f.name);
            if (pg != ctx.paramGenerators.end()) {
                if (!f.arg) throw CompileError("generator '" + f.name + "' needs an argument, e.g. " + f.name + "(pi/4)");
                const double theta = evalExpr(*f.arg, Env{});
                const auto& [param, kind, body] = pg->second;
                M = angleMatrix(kind, evalExpr(*body, Env{{param, theta}}));
                label = f.name + sfmt("(%.4g)", theta);
            } else {
                if (f.arg) throw CompileError("generator '" + f.name + "' takes no argument");
                auto g = ctx.generators.find(f.name);
                if (g != ctx.generators.end()) {
                    M = g->second;
                } else if (f.name == "e") {
                    if (!ctx.dim()) throw CompileError("identity 'e' needs a generator to fix the dimension");
                    M = DMat::identity(*ctx.dim());
                } else {
                    throw CompileError("unknown generator '" + f.name + "'");
                }
            }
            if (f.exp < 0) {
                try {
                    M = cleanMatrix(inverse(M));
                } catch (const SingularMatrix&) {
                    throw CompileError("generator '" + f.name + "' is not invertible");
                }
                label += "^-1";
            }
            for (int i = 0; i < std::abs(f.exp); ++i) factors.push_back({label, M});
        }
        return factors;
    }

    // Rightmost factor acts first, so layers are emitted in reversed order.
    static std::vector<Layer> wordLayers(const std::vector<Resolved>& factors) {
        std::vector<Layer> layers;
        int step = 1;
        for (auto it = factors.rbegin(); it != factors.rend(); ++it, ++step)
            layers.push_back(LinearLayer{sfmt("L%d:%s", step, it->label.c_str()), it->M});
        return layers;
    }

    static int dimFor(const CompiledGroup& ctx, const std::vector<Resolved>& factors) {
        if (!factors.empty()) return factors[0].M.rows;
        if (!ctx.dim()) throw CompileError("cannot determine dimension: declare a generator first");
        return *ctx.dim();
    }

    static std::vector<double> evalVector(const std::vector<ExprPtr>& exprs) {
        std::vector<double> v;
        for (const auto& e : exprs) v.push_back(evalExpr(*e, Env{}));
        return v;
    }

    static Job makeJob(std::string kind, std::string label, Network net, std::vector<double> input,
                       std::optional<std::pair<int, int>> shape, std::optional<bool> expected, std::string shortLbl) {
        Job j;
        j.kind = std::move(kind);
        j.label = std::move(label);
        j.folded = foldNetwork(net);  // algebraic constant folding happens here, at compile time
        j.network = std::move(net);
        j.inputVec = std::move(input);
        j.matrixShape = shape;
        j.expected = expected;
        j.shortLabel = std::move(shortLbl);
        return j;
    }

    // ---- eval / check ---------------------------------------------------------------
    Job compileEval(const CompiledGroup& ctx, const EvalStmt& st) const {
        const auto factors = resolveWord(ctx, st.word);
        const int N = dimFor(ctx, factors);
        const std::string label = "eval " + fmtWordItems(st.word);
        if (st.point) {
            auto point = evalVector(*st.point);
            if (static_cast<int>(point.size()) != N)
                throw CompileError(sfmt("point has dimension %zu, generators have %d", point.size(), N));
            Network net{label, N, N, false, wordLayers(factors)};
            return makeJob("eval", label + " on " + fmtVec(point), std::move(net), point, std::nullopt, std::nullopt, "");
        }
        Network net{label, N * N, N, true, wordLayers(factors)};
        DMat I = DMat::identity(N);
        return makeJob("eval", label, std::move(net), I.a, std::make_pair(N, N), std::nullopt, "");
    }

    CheckJob compileCheck(const CompiledGroup& ctx, const CheckStmt& st) const {
        EvalStmt l, r;
        l.word = st.lhs; l.line = st.line;
        r.word = st.rhs; r.line = st.line;
        return CheckJob{"check " + fmtWordItems(st.lhs) + " == " + fmtWordItems(st.rhs), compileEval(ctx, l), compileEval(ctx, r)};
    }

    // ---- classify / verify / sweep ----------------------------------------------------
    static const CompiledClassifier& getClassifier(const CompiledGroup& ctx, const std::string& name) {
        auto it = ctx.classifiers.find(name);
        if (it == ctx.classifiers.end()) throw CompileError("unknown classifier '" + name + "'");
        return it->second;
    }

    // Append lift layers, optional hidden perceptron bank, then the output hyperplane.
    static Network finishClassifierNetwork(const CompiledGroup& ctx, const CompiledClassifier& clf,
                                           std::vector<Layer> layers, int N, bool elementMode, int inputDim,
                                           const std::string& label) {
        const std::vector<LiftDecl>& lifts = clf.lifts ? *clf.lifts : ctx.liftDecls;
        int dim = inputDim;
        std::vector<std::string> names = defaultNames(dim);
        for (const LiftDecl& ld : lifts) {
            std::set<std::string> missing = freeVariables(*ld.expr);
            for (const auto& n : names) missing.erase(n);
            if (!missing.empty()) {
                std::vector<std::string> mv(missing.begin(), missing.end());
                throw CompileError("lift '" + ld.name + "' uses unknown variable(s) " + pyListRepr(mv) + "; available: " + pyListRepr(names));
            }
            layers.push_back(LiftLayer{"lift:" + ld.name, names, ld.expr, ld.name});
            names.push_back(ld.name);
            ++dim;
        }
        if (clf.bank) {
            if (clf.bank->W.cols != dim)
                throw CompileError(sfmt("sector '%s' works on R^%d but the state vector has dimension %d", clf.name.c_str(), clf.bank->W.cols, dim));
            layers.push_back(*clf.bank);
            dim = clf.bank->W.rows;
        }
        if (static_cast<int>(clf.weight.size()) != dim)
            throw CompileError(sfmt("classifier '%s' has %zu weights but the feature vector has dimension %d", clf.name.c_str(), clf.weight.size(), dim));
        layers.push_back(HyperplaneLayer{"hyperplane:" + clf.name, clf.weight, clf.bias});
        return Network{label, inputDim, N, elementMode, std::move(layers)};
    }

    Job classifyJob(const CompiledGroup& ctx, const CompiledClassifier& clf, const std::optional<Word>& word,
                    const std::optional<std::vector<double>>& point, std::optional<bool> expected,
                    const std::optional<std::string>& shortLbl) const {
        const bool hasWord = word && !word->empty();
        const auto factors = hasWord ? resolveWord(ctx, *word) : std::vector<Resolved>{};
        const std::string wtxt = hasWord ? fmtWordItems(*word) : "e";
        if (point) {  // point mode
            const int N = static_cast<int>(point->size());
            for (const Resolved& f : factors)
                if (f.M.rows != N)
                    throw CompileError(sfmt("point has dimension %d but %s is %dx%d", N, f.label.c_str(), f.M.rows, f.M.rows));
            const std::string label = "classify " + clf.name + (hasWord ? " of " + wtxt : std::string()) + " on " + fmtVec(*point);
            Network net = finishClassifierNetwork(ctx, clf, wordLayers(factors), N, false, N, label);
            return makeJob("classify", label, std::move(net), *point, std::nullopt, expected, shortLbl ? *shortLbl : fmtVec(*point));
        }
        const int N = dimFor(ctx, factors);  // element mode
        const std::string label = "classify " + clf.name + " of " + wtxt;
        Network net = finishClassifierNetwork(ctx, clf, wordLayers(factors), N, true, N * N, label);
        return makeJob("classify", label, std::move(net), DMat::identity(N).a, std::make_pair(N, N), expected, shortLbl ? *shortLbl : wtxt);
    }

    Job compileClassify(const CompiledGroup& ctx, const ClassifyStmt& st) const {
        const CompiledClassifier& clf = getClassifier(ctx, st.classifier);
        std::optional<bool> expected;
        if (st.expect) expected = (*st.expect != 0);
        std::optional<std::vector<double>> point;
        if (st.point) point = evalVector(*st.point);
        return classifyJob(ctx, clf, st.word, point, expected, std::nullopt);
    }

    VerifyJob compileVerify(const CompiledGroup& ctx, const VerifyStmt& st) const {
        const CompiledClassifier& clf = getClassifier(ctx, st.classifier);
        for (const std::string* nm : {&st.subgroup, &st.ambient})
            if (!ctx.subgroups.count(*nm)) throw CompileError("unknown subgroup '" + *nm + "'");
        const CompiledSubgroup& sub = ctx.subgroups.at(st.subgroup);
        const CompiledSubgroup& amb = ctx.subgroups.at(st.ambient);
        std::vector<Job> jobs;
        for (const auto& [word, M] : amb.elements) {
            Word items;
            for (const auto& g : word) items.push_back(Factor{g, 1, nullptr});
            jobs.push_back(classifyJob(ctx, clf, items, std::nullopt, sub.contains(M), std::nullopt));
        }
        return VerifyJob{sfmt("verify %s == membership in %s over all %zu elements of %s", clf.name.c_str(), sub.name.c_str(), amb.elements.size(), amb.name.c_str()), std::move(jobs)};
    }

    VerifyJob compileSweep(const CompiledGroup& ctx, const SweepStmt& st) const {
        const CompiledClassifier& clf = getClassifier(ctx, st.classifier);
        if (!clf.sector) throw CompileError("classifier '" + clf.name + "' is not a sector");
        if (!ctx.paramGenerators.count(st.gen)) throw CompileError("'" + st.gen + "' is not a parametrised generator");
        const auto [lo, hi] = *clf.sector;
        const double a = evalExpr(*st.start, Env{}), b = evalExpr(*st.stop, Env{});
        std::vector<Job> jobs;
        for (int k = 0; k < st.steps; ++k) {  // half-open grid [a, b)
            const double theta = a + k * (b - a) / st.steps;
            const bool expected = ContinuousSO2Group::inSector(theta, lo, hi);
            Word w{Factor{st.gen, 1, mkNum(theta)}};
            jobs.push_back(classifyJob(ctx, clf, w, std::vector<double>{1.0, 0.0}, expected,
                                       sfmt("theta=%.3f*pi", theta / kPi)));
        }
        return VerifyJob{sfmt("sweep %s: %d angles of %s(theta) applied to (1, 0) vs analytic arc test", clf.name.c_str(), st.steps, st.gen.c_str()), std::move(jobs)};
    }
};

}  // namespace symnet
