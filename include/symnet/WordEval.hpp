// WordEval.hpp - binding-friendly word evaluation on fixed-size, 64-byte aligned stack buffers.
//
// Element-mode semantics (same as `classify <clf> of <word>` with no lifts):
//   state X0 = I,  W_compiled = F1 * F2 * ... * Fn  (folded right to left),
//   feature vector = vec(W_compiled) (row-major, k*k entries),
//   decision = Step(w . vec(W_compiled) + b).
#pragma once

#include "MatrixEngine.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace symnet {

constexpr int kMaxWordDim = 4;                           // fixed-size kernels exist for k = 1..4
constexpr int kMaxWordElems = kMaxWordDim * kMaxWordDim;

inline void checkWordArgs(int k, std::size_t factorsLen, std::size_t weightLen) {
    if (k < 2 || k > kMaxWordDim) throw std::invalid_argument("k must be 2, 3 or 4");
    const std::size_t kk = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);
    if (factorsLen % kk != 0) throw std::invalid_argument("factors length must be a multiple of k*k");
    if (weightLen != kk) throw std::invalid_argument("weight must have k*k entries (hyperplane over vec(W_compiled))");
}

// Fold a word (L row-major k x k matrices concatenated, in word order) into W_compiled.
// Every operand is copied into alignas(64) stack arrays first; out receives k*k doubles.
inline void foldWord(int k, const std::vector<double>& factors, double* out) {
    checkWordArgs(k, factors.size(), static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
    const std::size_t kk = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);
    alignas(64) double acc[kMaxWordElems], tmp[kMaxWordElems], f[kMaxWordElems];
    for (std::size_t i = 0; i < kMaxWordElems; ++i) acc[i] = tmp[i] = f[i] = 0.0;
    for (int i = 0; i < k; ++i) acc[i * k + i] = 1.0;
    for (std::size_t idx = factors.size() / kk; idx-- > 0;) {  // rightmost factor acts first
        for (std::size_t j = 0; j < kk; ++j) f[j] = factors[idx * kk + j];
        gemm(f, acc, tmp, k);  // acc <- F_idx * acc
        for (std::size_t j = 0; j < kk; ++j) acc[j] = tmp[j];
    }
    for (std::size_t j = 0; j < kk; ++j) out[j] = acc[j];
}

// Decide Step(w . vec(W_compiled) + b) for a word.
inline bool evalWordCompiled(int k, const std::vector<double>& factors, const std::vector<double>& weight, double bias) {
    checkWordArgs(k, factors.size(), weight.size());
    const std::size_t kk = static_cast<std::size_t>(k) * static_cast<std::size_t>(k);
    alignas(64) double w[kMaxWordElems];
    alignas(64) double wc[kMaxWordElems];
    for (std::size_t j = 0; j < kMaxWordElems; ++j) w[j] = wc[j] = 0.0;
    for (std::size_t j = 0; j < kk; ++j) w[j] = weight[j];
    foldWord(k, factors, wc);
    return dot(w, wc, kk) + bias > 0.0;
}

// Fold once, decide many times (O(1) in the word length per decision).
class CompiledWord {
public:
    CompiledWord(int k, const std::vector<double>& factors) : k_(k) {
        foldWord(k, factors, wc_);
        length_ = factors.size() / (static_cast<std::size_t>(k) * static_cast<std::size_t>(k));
    }
    int dim() const { return k_; }
    std::size_t length() const { return length_; }
    std::vector<double> matrix() const { return std::vector<double>(wc_, wc_ + k_ * k_); }
    bool decide(const std::vector<double>& weight, double bias) const {
        checkWordArgs(k_, 0, weight.size());
        alignas(64) double w[kMaxWordElems];
        for (std::size_t j = 0; j < kMaxWordElems; ++j) w[j] = 0.0;
        const std::size_t kk = static_cast<std::size_t>(k_) * static_cast<std::size_t>(k_);
        for (std::size_t j = 0; j < kk; ++j) w[j] = weight[j];
        return dot(w, wc_, kk) + bias > 0.0;
    }

private:
    int k_;
    std::size_t length_ = 0;
    alignas(64) double wc_[kMaxWordElems] = {};
};

}  // namespace symnet
