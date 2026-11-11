// MatrixEngine.hpp - matrix primitives for the SymNet engine.
//
//  * Mat<N>       : stack-allocated, 32-byte aligned NxN matrix (Matrix2x2 / 3x3 / 4x4)
//  * gemm/gemv    : direct k x k kernels. Fixed-size kernels for k = 1..4 are fully
//                   unrolled by the compiler; larger k uses a cache-friendly i-k-j loop.
//  * dot          : four independent accumulators over contiguous memory, so the compiler
//                   can emit SIMD without -ffast-math (reduction order is fixed).
//  * DMat         : dynamic row-major matrix used at compile time and for general k.
//
// The runtime never expands a Kronecker product: element-mode words act on a state matrix
// X in R^(k x k) directly (X <- W X), which is O(k^3) instead of O(k^4) for kron(W, I) vec(X).
#pragma once

#include "Common.hpp"

#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define SYMNET_RESTRICT __restrict__
#define SYMNET_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define SYMNET_RESTRICT __restrict
#define SYMNET_FORCE_INLINE __forceinline
#else
#define SYMNET_RESTRICT
#define SYMNET_FORCE_INLINE inline
#endif

namespace symnet {

// Number of k x k matrix-multiplication passes executed so far (instrumentation used to
// prove that a folded word costs exactly one pass).
inline std::size_t& gemmCounter() {
    static std::size_t counter = 0;
    return counter;
}

// ---- dot product --------------------------------------------------------------------
SYMNET_FORCE_INLINE double dot(const double* SYMNET_RESTRICT a, const double* SYMNET_RESTRICT b,
                               std::size_t n) {
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        s0 += a[i] * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
    }
    for (; i < n; ++i) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

// ---- fixed-size kernels (C = A*B, y = A*x), no aliasing between inputs and outputs ------
template <int N>
SYMNET_FORCE_INLINE void gemmFixed(const double* SYMNET_RESTRICT A, const double* SYMNET_RESTRICT B,
                                   double* SYMNET_RESTRICT C) {
    for (int i = 0; i < N; ++i) {
        double row[N];
        for (int j = 0; j < N; ++j) row[j] = 0.0;
        for (int k = 0; k < N; ++k) {
            const double aik = A[i * N + k];
            for (int j = 0; j < N; ++j) row[j] += aik * B[k * N + j];
        }
        for (int j = 0; j < N; ++j) C[i * N + j] = row[j];
    }
}

template <int N>
SYMNET_FORCE_INLINE void gemvFixed(const double* SYMNET_RESTRICT A, const double* SYMNET_RESTRICT x,
                                   double* SYMNET_RESTRICT y) {
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int k = 0; k < N; ++k) s += A[i * N + k] * x[k];
        y[i] = s;
    }
}

// ---- dispatching kernels for run-time k ---------------------------------------------
// C = A * B for k x k row-major matrices (C must not alias A or B).
inline void gemm(const double* A, const double* B, double* C, int k) {
    ++gemmCounter();
    switch (k) {
        case 1: C[0] = A[0] * B[0]; return;
        case 2: gemmFixed<2>(A, B, C); return;
        case 3: gemmFixed<3>(A, B, C); return;
        case 4: gemmFixed<4>(A, B, C); return;
        default: break;
    }
    std::fill(C, C + static_cast<std::size_t>(k) * k, 0.0);
    for (int i = 0; i < k; ++i)
        for (int kk = 0; kk < k; ++kk) {
            const double aik = A[i * k + kk];
            for (int j = 0; j < k; ++j) C[i * k + j] += aik * B[kk * k + j];
        }
}

// y = A * x for a k x k matrix (y must not alias x).
inline void gemvSquare(const double* A, const double* x, double* y, int k) {
    switch (k) {
        case 1: y[0] = A[0] * x[0]; return;
        case 2: gemvFixed<2>(A, x, y); return;
        case 3: gemvFixed<3>(A, x, y); return;
        case 4: gemvFixed<4>(A, x, y); return;
        default: break;
    }
    for (int i = 0; i < k; ++i) y[i] = dot(A + static_cast<std::size_t>(i) * k, x, static_cast<std::size_t>(k));
}

// y = A * x for a rows x cols matrix.
inline void gemv(const double* A, const double* x, double* y, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        y[i] = dot(A + static_cast<std::size_t>(i) * cols, x, static_cast<std::size_t>(cols));
}

// ---- stack-allocated fixed-size matrix ----------------------------------------------
template <int N>
struct alignas(32) Mat {
    double a[N * N];

    static Mat zero() {
        Mat m;
        for (double& v : m.a) v = 0.0;
        return m;
    }
    static Mat identity() {
        Mat m = zero();
        for (int i = 0; i < N; ++i) m.a[i * N + i] = 1.0;
        return m;
    }
    static Mat fromData(const double* p) {
        Mat m;
        for (int i = 0; i < N * N; ++i) m.a[i] = p[i];
        return m;
    }
    double& operator()(int i, int j) { return a[i * N + j]; }
    double operator()(int i, int j) const { return a[i * N + j]; }

    friend Mat operator*(const Mat& A, const Mat& B) {
        ++gemmCounter();
        Mat C;
        gemmFixed<N>(A.a, B.a, C.a);
        return C;
    }
    std::array<double, N> apply(const std::array<double, N>& x) const {
        std::array<double, N> y{};
        gemvFixed<N>(a, x.data(), y.data());
        return y;
    }
    double maxAbsDiff(const Mat& o) const {
        double m = 0.0;
        for (int i = 0; i < N * N; ++i) m = std::max(m, std::fabs(a[i] - o.a[i]));
        return m;
    }
};

using Matrix2x2 = Mat<2>;
using Matrix3x3 = Mat<3>;
using Matrix4x4 = Mat<4>;

// ---- dynamic row-major matrix ---------------------------------------------------------
struct DMat {
    int rows = 0, cols = 0;
    std::vector<double> a;

    DMat() = default;
    DMat(int r, int c, double fill = 0.0) : rows(r), cols(c), a(static_cast<std::size_t>(r) * c, fill) {}

    static DMat identity(int n) {
        DMat m(n, n, 0.0);
        for (int i = 0; i < n; ++i) m(i, i) = 1.0;
        return m;
    }
    double& operator()(int i, int j) { return a[static_cast<std::size_t>(i) * cols + j]; }
    double operator()(int i, int j) const { return a[static_cast<std::size_t>(i) * cols + j]; }
    double* data() { return a.data(); }
    const double* data() const { return a.data(); }
    bool isSquare() const { return rows == cols; }
    const double* row(int i) const { return a.data() + static_cast<std::size_t>(i) * cols; }
};

inline DMat matmul(const DMat& A, const DMat& B) {
    if (A.cols != B.rows) throw SymNetRuntimeError("matmul: incompatible shapes");
    if (A.isSquare() && B.isSquare() && A.rows == B.rows) {
        DMat C(A.rows, A.rows);
        gemm(A.data(), B.data(), C.data(), A.rows);
        return C;
    }
    DMat C(A.rows, B.cols, 0.0);
    for (int i = 0; i < A.rows; ++i)
        for (int k = 0; k < A.cols; ++k)
            for (int j = 0; j < B.cols; ++j) C(i, j) += A(i, k) * B(k, j);
    return C;
}

inline DMat kron(const DMat& A, const DMat& B) {
    DMat K(A.rows * B.rows, A.cols * B.cols, 0.0);
    for (int i = 0; i < A.rows; ++i)
        for (int j = 0; j < A.cols; ++j)
            for (int p = 0; p < B.rows; ++p)
                for (int q = 0; q < B.cols; ++q) K(i * B.rows + p, j * B.cols + q) = A(i, j) * B(p, q);
    return K;
}

struct SingularMatrix : std::runtime_error { SingularMatrix() : std::runtime_error("singular matrix") {} };

// Gauss-Jordan inverse with partial pivoting; throws SingularMatrix on an exactly zero pivot.
inline DMat inverse(const DMat& M) {
    if (!M.isSquare()) throw SingularMatrix();
    const int n = M.rows;
    DMat A = M, I = DMat::identity(n);
    for (int c = 0; c < n; ++c) {
        int p = c;
        for (int r = c + 1; r < n; ++r)
            if (std::fabs(A(r, c)) > std::fabs(A(p, c))) p = r;
        if (A(p, c) == 0.0) throw SingularMatrix();
        if (p != c)
            for (int j = 0; j < n; ++j) {
                std::swap(A(p, j), A(c, j));
                std::swap(I(p, j), I(c, j));
            }
        const double inv = 1.0 / A(c, c);
        for (int j = 0; j < n; ++j) {
            A(c, j) *= inv;
            I(c, j) *= inv;
        }
        for (int r = 0; r < n; ++r) {
            if (r == c) continue;
            const double f = A(r, c);
            if (f == 0.0) continue;
            for (int j = 0; j < n; ++j) {
                A(r, j) -= f * A(c, j);
                I(r, j) -= f * I(c, j);
            }
        }
    }
    return I;
}

// Zero out floating-point dust such as cos(pi/2) = 6e-17.
inline DMat cleanMatrix(DMat m, double tol = 1e-12) {
    for (double& v : m.a) v = (std::fabs(v) < tol) ? 0.0 : v + 0.0;
    return m;
}

// Hashable fingerprint of a matrix, robust to float dust and -0.0.
inline std::vector<double> matrixKey(const DMat& M) {
    std::vector<double> k(M.a.size());
    for (std::size_t i = 0; i < k.size(); ++i) k[i] = nz(roundTo(M.a[i], 6));
    return k;
}

inline double maxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

}  // namespace symnet
