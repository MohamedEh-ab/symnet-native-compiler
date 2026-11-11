// Common.hpp - error hierarchy, formatting and rounding helpers shared by every module.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace symnet {

constexpr double kPi = 3.14159265358979323846;

// ---- errors (mirror the Python hierarchy) -----------------------------------------
struct SymNetError : std::runtime_error { using std::runtime_error::runtime_error; };
struct LexError : SymNetError { using SymNetError::SymNetError; };
struct ParseError : SymNetError { using SymNetError::SymNetError; };
struct CompileError : SymNetError { using SymNetError::SymNetError; };
struct SymNetRuntimeError : SymNetError { using SymNetError::SymNetError; };

// Helper for std::visit with lambdas.
template <class... Ts> struct overloaded : Ts... { using Ts::operator()...; };
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// ---- printf-style std::string builder ---------------------------------------------
#if defined(__GNUC__) || defined(__clang__)
__attribute__((format(printf, 1, 2)))
#endif
inline std::string sfmt(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    va_list ap2;
    va_copy(ap2, ap);
    const int n = std::vsnprintf(nullptr, 0, fmt, ap);
    va_end(ap);
    std::string s(n > 0 ? static_cast<std::size_t>(n) : 0u, '\0');
    if (n > 0) std::vsnprintf(&s[0], s.size() + 1, fmt, ap2);
    va_end(ap2);
    return s;
}

// Python's f"{x:g}"
inline std::string fmtG(double x) { return sfmt("%g", x); }

// Python-style repr of a short string: 'text'
inline std::string pyRepr(const std::string& s) { return "'" + s + "'"; }

// numpy.round(x, decimals) == rint(x * 10^d) / 10^d
inline double roundTo(double x, int decimals) {
    const double p = std::pow(10.0, decimals);
    return std::nearbyint(x * p) / p;
}

// removes -0.0 (Python's "+ 0.0" idiom, robust against -ffast-math)
inline double nz(double x) { return x == 0.0 ? 0.0 : x; }

// Python's math.radians
inline double radians(double deg) { return deg * (kPi / 180.0); }

// Python float modulo (result takes the sign of the divisor)
inline double pyMod(double a, double b) {
    double m = std::fmod(a, b);
    if (m != 0.0) {
        if ((b < 0.0) != (m < 0.0)) m += b;
    } else {
        m = std::copysign(0.0, b);
    }
    return m;
}

// Compact vector formatting: [1, 0, 0.5]   (rounded to `prec` decimals, no -0)
inline std::string fmtVec(const double* v, std::size_t n, int prec = 3) {
    std::string s = "[";
    for (std::size_t i = 0; i < n; ++i) {
        if (i) s += ", ";
        s += fmtG(nz(roundTo(v[i], prec)));
    }
    return s + "]";
}
inline std::string fmtVec(const std::vector<double>& v, int prec = 3) {
    return fmtVec(v.data(), v.size(), prec);
}

// Compact matrix formatting (row-major data): [[0, -1], [1, 0]]
inline std::string fmtMat(const double* m, int rows, int cols) {
    std::string s = "[";
    for (int i = 0; i < rows; ++i) {
        if (i) s += ", ";
        s += fmtVec(m + static_cast<std::size_t>(i) * cols, static_cast<std::size_t>(cols));
    }
    return s + "]";
}

// Variable names for vector components: x, y, w, t, v4, v5, ...
inline std::vector<std::string> defaultNames(int n) {
    static const char* base[] = {"x", "y", "w", "t"};
    std::vector<std::string> out;
    for (int i = 0; i < n; ++i) out.push_back(i < 4 ? std::string(base[i]) : "v" + std::to_string(i));
    return out;
}

}  // namespace symnet
