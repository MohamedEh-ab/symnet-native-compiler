# SymNet: A Symbolic neural DSL Compiler for Group-Theoretic Classifiers with an Exact Finite-Field (Lefschetz / Zeta / Weil) Engine

[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17%2F20-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![Build System](https://img.shields.io/badge/CMake-3.18%2B-brightgreen.svg)](https://cmake.org/)
[![Python Extension](https://img.shields.io/badge/Pybind11-v2.13%2B-orange.svg)](https://github.com/pybind/pybind11)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/1453389569.svg)](https://doi.org/10.5281/zenodo.24123292)

**SymNet** is a domain-specific compiler and header-only C++17 runtime. It has two halves that share one matrix engine:

1. **The SymNet DSL compiler.** It translates symbolic group-theoretic expressions (finite matrix groups such as $D_4$ and the Klein 4-group), continuous $SO(2)$ phasors, and non-linear set partitions into constant-time linear tensor graphs. A word $g_1 g_2 \cdots g_L$ is folded into **one** $k \times k$ matrix at compile time, so evaluating it costs one matrix pass regardless of $L$ (**145 µs → 9 ns** at $L = 8192$ in the benchmark below).
2. **The exact finite-field passes (`symnet::ff`).** For an elliptic curve $E: y^2 = x^3 + ax + b$ over $\mathbb{F}_q$, the $q$-power Frobenius is lifted to a $2 \times 2$ integer matrix. Point counts over every extension field, the zeta function, and certificates for the Hasse–Weil bound and the functional equation are then computed in exact 64/128-bit integer arithmetic, with no floating point in any invariant.
3. **A Python extension (`symnet_native`).** Fixed-size matrices and word folding, the full DSL interpreter/compiler (`symnet_native.dsl`) and the finite-field passes (`symnet_native.finite_field`) are all callable from Python.

---

## Table of Contents

1. [Key Features](#key-features)
2. [How the Compiler Works](#how-the-compiler-works)
3. [Mathematical Background (finite-field passes)](#mathematical-background-finite-field-passes)
4. [Repository Layout](#repository-layout)
5. [Build and Test](#build-and-test)
6. [Usage](#usage)
7. [DSL Reference](#dsl-reference)
8. [API Reference](#api-reference)
   - [Python API (`symnet_native`)](#python-api-symnet_native)
   - [C++ API: DSL engine](#c-api-dsl-engine)
   - [C++ API: finite-field passes](#c-api-finite-field-passes)
9. [Benchmark](#benchmark)
10. [Validation](#validation)
11. [Limitations and Honest Notes](#limitations-and-honest-notes)
12. [Glossary](#glossary)
13. [Citation](#citation)

---

## Key Features

### DSL compiler and runtime

1. **Algebraic constant folding ($O(1)$ in word length).**
   The compiler pre-multiplies the generator matrices of a word at compile time:
   $$W_{\text{compiled}} = F_1 \times F_2 \times \cdots \times F_L \in \mathbb{R}^{k \times k}$$
   Evaluating word membership then takes **exactly one matrix pass**, independent of $L$.

2. **Automated non-linear feature-lift synthesis.**
   A Novikoff perceptron cycle detector proves that a partition is not linearly separable and searches a library of basis functions for the smallest lift that makes it separable. It finds $z = \lvert x - y \rvert$ for XOR and $z = m_{00} \cdot m_{11}$ for the Klein 4-group and $D_4$-centre partitions.

3. **Direct tensor execution, no Kronecker expansion.**
   Element-mode words act on a state matrix $X \in \mathbb{R}^{k \times k}$ as $X \leftarrow W X$ instead of building $\mathrm{kron}(W, I)$. This lowers the cost from $O(k^4)$ to $O(k^3)$. The engine uses unrolled fixed-size kernels for $k \in \{1, 2, 3, 4\}$ and four-accumulator dot products so the compiler can vectorise without `-ffast-math`. `Mat<N>` is 32-byte aligned; the `WordEval` buffers are `alignas(64)`.

4. **Continuous Lie group $SO(2)$.**
   Complex unit phasors $e^{i\theta}$ map to $2 \times 2$ rotation matrices, and angular sectors compile to a two-unit half-plane perceptron bank with AND (width $\le \pi$) or OR (width $> \pi$) output logic.

5. **Folded ≡ reference.**
   Every job is also run through a literal Kronecker-expanded reference path (a port of the original Python forward pass). The interpreter cross-checks the two on every job.

### Exact finite-field passes (new)

6. **Frobenius lift.** `EllipticCurveNode` → `FrobeniusGenerator::lift` produces the companion matrix

$$
F_q = \begin{pmatrix} 
0 & -q \\ 
1 & a_q 
\end{pmatrix}
$$

of $x^2 - a_q x + q$ as an exact `int64` $2 \times 2$ buffer (`Mat2`).

7. **Lefschetz trace pass.** $\mathrm{Tr}F$, $\det F$, $P(T) = \det(I - TF)$, $F^n$ by branch-free binary exponentiation, and $\\#E(\mathbb{F}_{q^n}) = \det(I - F^n)$. An AVX2 path is used for $\mathrm{Tr}F$ when available.
8. **Zeta collapse.** The series $\exp\!\big(\sum N_n T^n / n\big)$ is never summed. The pass emits the closed form $Z(T) = \dfrac{1 - tT + qT^2}{(1 - T)(1 - qT)}$ directly.
9. **Weil prover.** Exact integer checkers for the Hasse–Weil bound ($t^2 \le 4q$, compared as 128-bit integers) and for the genus-1 functional equation $Z(1/(qT)) = Z(T)$, returning structured `PROVED` / `REFUTED` certificates.
10. **Overflow discipline.** Every kernel is branch-free and carries a sticky overflow flag (`Checked<T>`). A value that cannot be certified inside `int64` is reported as such and never silently wrapped.
11. **Engine bridge.** `toEngine` / `fromEngine` convert between `ff::Mat2` and `symnet::Matrix2x2` and refuse (return `nullopt`) whenever the conversion would not be exact. `ff::mul` is counted in `symnet::gemmCounter()` like any engine matrix pass.


### Python API (new)

12. **`symnet_native.dsl`**: `Interpreter.run()` / `.compile()` over the DSL; navigate `Program → Group → Task → Job`, inspect synthesized lifts, the folded matrix `W_compiled`, and evaluate any job on the folded graph or the Kronecker-expanded reference.
13. **`symnet_native.finite_field`**: `EllipticCurve`, `lift`, `Mat2`, the Lefschetz / zeta / Weil passes, with `Checked<T>` overflow surfaced as `OverflowError` or `None`, never a wrapped value.
14. **Typed exceptions**: `SymNetError` with `LexError`, `ParseError`, `CompileError`, `SymNetRuntimeError`.

---

## How the Compiler Works

### A. DSL pipeline

```
SymNet DSL source (.sym or built-in example)
│
├── Lexer.hpp ········· tokenize()                    regex lexer → Token stream
├── Parser.hpp ········ Parser::parseProgram()        recursive descent → AST
│       └── AST.hpp ··· GroupDecl → Statement (std::variant of 12 statement kinds)
│                       Expr tree (Num | Var | Neg | BinOp | Call)
│
▼
SymNetCompiler  (Compiler.hpp)  ·····  compileGroup()  per group
│
├── Declarations
│     ├── gen          → DMat (cleaned) | parametrised family R(θ) → angleMatrix()
│     ├── subgroup     → closure(): BFS by right-multiplication (≤ 2048 elements)
│     ├── separate     → Hyperplane (weight, bias)
│     ├── sector       → ContinuousSO2Group::sectorPerceptrons() → perceptron bank + AND/OR output
│     ├── space / lift → labelled points; user-declared lift expressions
│     └── synthesize ──► KernelSynthesizer
│           ├── perceptronSeparable()      Novikoff cycling certificate (non-separable ⇒ cycle)
│           ├── candidateLibrary()         u·v, abs(u−v), abs(u+v), u², abs(u), cubic terms
│           ├── candidateColumns()         evaluate on the data, drop constants/duplicates
│           ├── flat search k = 1..3       fewest non-zero weights wins
│           ├── composed search            k2 built on top of k1 (e.g. abs(k1 − w))
│           └── cleanHyperplane()          sparse support search + margin normalisation
│
├── Queries:  eval | classify | verify | check | sweep
│     ├── resolveWord()          expand exponents / R(θ) arguments / inverses
│     ├── wordLayers()           one LinearLayer per factor (rightmost acts first)
│     ├── finishClassifierNetwork()   + LiftLayers + PerceptronBank + Hyperplane
│     │                          ⇒ Network (unfolded IR)
│     └── foldNetwork()  ◄── CONSTANT FOLDING PASS
│           ├── W_compiled = F1·F2···FL           (compile time, once)
│           └── lift expressions → constant-folded RPN tapes (ExprProgram)
│                                  ⇒ FoldedNetwork
▼
SymNetRuntime  (Runtime.hpp)
├── forward()            folded graph: ONE k×k pass → features, Step decision, soft score, trace
├── decideFast()         allocation-free hot path (the 9 ns path)
├── forwardReference()   unfolded Kronecker-expanded oracle (Python-faithful trace)
├── decideUnfolded()     O(L·k³) per-factor baseline
└── decideKron()         O(L·k⁴) dense-Kronecker baseline
▼
SymNetInterpreter  (Interpreter.hpp)   run(): print transcript, cross-check folded vs reference → Report
```

### B. Finite-field pipeline

```
EllipticCurveNode(a, b, q)                        y² = x³ + ax + b over F_q
│   ├── valid()            3 < q ≤ 2⁶², q odd, q % 3 ≠ 0, 4a³ + 27b² ≠ 0 (mod q)
│   └── trace a_q          with_trace(t)  |  resolve(TraceResolver)  (Schoof/SEA plugs in here)
▼
FrobeniusGenerator::lift()  ──►  FrobeniusNode { F = [[0, −q], [1, a_q]],  q }      (Mat2, int64)
│
├──► LefschetzTracePass
│        ├── trace(F)  determinant(F)  characteristic_polynomial(F) = 1 − tT + qT²
│        ├── power(F, n)            binary exponentiation, ≤ 64 steps, mask-select, sticky overflow
│        ├── trace_of_power(F, n)   Tr(Fⁿ) = αⁿ + βⁿ
│        └── count_points(F, n)     #E(F_{qⁿ}) = det(I − Fⁿ)
│
├──► ZetaFunctionPass::collapse(F, q)  ──►  ZetaNode { num = P(T), den = (1−T)(1−qT) }
│
├──► WeilProver::certify(node)  ──►  WeilCertificate
│        ├── HasseWeilReport           det F = q  ∧  t² ≤ 4q  (128-bit compare)
│        └── FunctionalEquationReport  (a) F·F⁻¹ → I₂   (b) FᵀJF = qJ   (c) c₂ = q·c₀ for P and D
│
└──► toEngine() / fromEngine()  ⇄  symnet::Matrix2x2   (exact-or-nullopt, |entry| ≤ 2⁵³)
```


### C. Python binding layer

```
Python  ──►  src/bindings.cpp ····· symnet_native: Matrix2x2/3x3/4x4, eval_word_compiled, CompiledWord
        │    src/bind_dsl.cpp ····· symnet_native.dsl ········ ┐  thin pybind11 registration only
        │    src/bind_finite_field.cpp  symnet_native.finite_field ┘
        ▼
include/symnet/Api.hpp   (pure C++ facade, no pybind11 types; unit-tested natively by tests/cpp/test_api.cpp)
        ├── Interpreter → Program → GroupHandle → TaskHandle → JobHandle → ForwardResult
        └── ffMul / ffDet / ffAdjugate / ffCharPoly / ffPower   Checked<T> overflow → std::overflow_error → OverflowError
        ▼
SymNetInterpreter · SymNetRuntime · ff::* passes   (the engine described above)
```
---

## Mathematical Background (finite-field passes)

For a prime $\ell \neq \mathrm{char}\mathbb{F}q$, $H^1{\text{ét}}(E_{\overline{\mathbb{F}}q}, \mathbb{Q}\ell)$ is 2-dimensional and the Frobenius acts on it by a $2 \times 2$ matrix $F$. Three facts drive every pass:

| Fact | Statement |
|---|---|
| Lefschetz fixed-point formula | $\\#E(\mathbb{F}_{q^n}) = \det(I - F^n) = q^n + 1 - \mathrm{Tr}(F^n)$ |
| Poincaré duality / Weil pairing | $\det F = q$, and $F^{\mathsf T} J F = qJ$ with $J = \begin{pmatrix} 0 & 1 \\ -1 & 0 \end{pmatrix}$ |
| Integrality | $\mathrm{Tr}F = a_q = q + 1 - \\#E(\mathbb{F}_q)$ is an integer independent of $\ell$ |

**Zeta function.** Since $\sum_{n \ge 1} c^n T^n / n = -\log(1 - cT)$, the series splits term by term and
$Z(E/\mathbb{F}_q, T) = \frac{\det(I - TF)}{(1 - T)(1 - qT)} = \frac{1 - tT + qT^2}{(1 - T)(1 - qT)}.$

**Hasse–Weil (Theorem 1).** With $\Delta = t^2 - 4q$: $\Delta \le 0 \iff |\alpha| = |\beta| = \sqrt{q} \iff |t| \le 2\sqrt{q}$. $t^2$ and $4q$ are compared as 128-bit integers, so there is no square root and no overflow.

**Functional equation (Theorem 2, corrected form).** For genus 1 the relation is
$Z(E, 1/(qT)) = Z(E, T).$
The commonly misquoted form $Z(1/(qT)) = qT^2 Z(T)$ is **false**: the factor $qT^2$ belongs to the numerator and denominator separately $qT^2 P(1/(qT)) = P(T)$ and $qT^2 D(1/(qT)) = D(T)$. The test suite checks this on exact rationals and confirms the literal form fails.

> **What "PROVED" means.** Hasse (1933) and Weil (1948) proved these statements long ago. The passes *check*, exactly and in integers, that a given $(F, q)$ satisfies them. With a correct trace the Hasse check must pass; a failure signals a wrong trace or a non-Frobenius matrix, never a counterexample to Weil.

---

## Repository Layout

```
SymNet/
├── CMakeLists.txt
├── README.md
├── include/symnet/                       header-only engine
│   ├── Common.hpp                        errors, formatting, numpy-style rounding
│   ├── MatrixEngine.hpp                  Mat<N>, gemm/gemv/dot kernels, DMat, kron, inverse
│   ├── AST.hpp                           Expr tree, Statement variant, ExprProgram (RPN, constant-folded)
│   ├── Lexer.hpp                         regex tokenizer
│   ├── Parser.hpp                        recursive-descent parser
│   ├── Compiler.hpp                      Network IR, foldNetwork, ContinuousSO2Group, KernelSynthesizer, SymNetCompiler
│   ├── Runtime.hpp                       SymNetRuntime (folded + reference + baselines)
│   ├── Interpreter.hpp                   SymNetInterpreter, Report
│   ├── WordEval.hpp                      fixed-size word folding used by the Python bindings
│   ├── Examples.hpp                      built-in DSL demonstration programs
│   ├── Api.hpp                           NEW: binding facade (handles + overflow-to-exception wrappers)
│   ├── Mat2.hpp                          NEW: exact int64 2x2 buffer, Checked<T>, U128, engine bridge
│   └── finite_field/                     NEW: exact finite-field layer
│       ├── FrobeniusGenerator.hpp            EllipticCurveNode -> Frobenius companion matrix
│       ├── LefschetzTracePass.hpp            Tr / det / P(T) / F^n / #E(F_{q^n})
│       ├── ZetaFunctionPass.hpp              closed-form zeta collapse
│       └── WeilProver.hpp                    Hasse-Weil and functional-equation certificates
├── src/
│   ├── main.cpp                          symnet_cli
│   ├── bindings.hpp                      shared declarations + submodule helper
│   ├── bindings.cpp                      module entry: matrices, word evaluation
│   ├── bind_dsl.cpp                      NEW: symnet_native.dsl + exception hierarchy
│   └── bind_finite_field.cpp             NEW: symnet_native.finite_field
├── tests/
│   ├── CMakeLists.txt
│   ├── cpp/
│   │   ├── test_engine.cpp                   DSL engine unit tests + benchmark
│   │   ├── test_api.cpp                      NEW: binding facade tests (what the Python API does)
│   │   ├── finite_field_test.cpp             NEW: brute-force-oracle validation of the ff passes
│   │   └── engine_integration_test.cpp       NEW: ff (exact int) vs MatrixEngine (double) cross-checks
│   └── python/
│       └── test_bindings.py                  matrices, word eval, dsl, finite_field
├── single_header/
│   └── symnet_native_single.cpp          generated by scripts/amalgamate.py (the CLI program)
└── scripts/
    └── amalgamate.py
```

> **Include root.** The finite-field headers include each other as `symnet/Mat2.hpp` and `symnet/finite_field/*.hpp`, so `include/` must be the include root (the CMake target `SymNet::native_cpp` already does this).

---

## Build and Test

Requires CMake ≥ 3.18 and a C++17 compiler. The Python module additionally needs Python ≥ 3.8 and pybind11 (found automatically after `pip install pybind11`, otherwise fetched).

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Options: `-DSYMNET_NATIVE_ARCH=OFF` (portable binaries), `-DSYMNET_STRICT_FP=ON` (`-ffp-contract=off`, bit-identical to numpy), `-DSYMNET_BUILD_PYTHON=OFF`, `-DSYMNET_FETCH_PYBIND11=OFF`.

`ctest` runs the unit tests, the finite-field suites, the facade tests, the single-header build and (when the module was built) the Python tests. **Finite-field tests by hand** (header-only; `-I` must point at the directory that contains `symnet/`):

```bash
g++ -std=c++17 -O3 -march=native -Wall -Wextra -I include tests/cpp/finite_field_test.cpp       -o ff_test && ./ff_test
g++ -std=c++17 -O3 -march=native -Wall -Wextra -I include tests/cpp/engine_integration_test.cpp -o ei_test && ./ei_test
```

Preprocessor switches: `SYMNET_FF_NO_GEMM_COUNT` (do not count `ff::mul` in `gemmCounter()`), `SYMNET_FF_PORTABLE_ARITH` (use the 128-bit portable multiply-overflow path instead of `__builtin_mul_overflow`). The AVX2 trace kernel is selected automatically when `__AVX2__` is defined.

---

## Usage

### CLI

```bash
./build/symnet_cli                 # built-in Phase 1 + Phase 2 demonstrations
./build/symnet_cli -v              # full transcript with per-layer traces
./build/symnet_cli -q              # summary line only
./build/symnet_cli program.sym     # your own DSL file
```

Exit status: `0` all checks passed, `1` a check failed, `2` usage/parse/compile error.

### C++: DSL engine

```cmake
target_link_libraries(my_app PRIVATE SymNet::native_cpp)
```

```cpp
#include <symnet/Interpreter.hpp>
symnet::SymNetInterpreter interp(10.0, &std::cout);
symnet::Report r = interp.run(dsl_text);       // r.passed, r.failed, r.xChecked, r.xFailed
```

### C++: finite-field certification

```cpp
#include "symnet/finite_field/WeilProver.hpp"
using namespace symnet::ff;

// y^2 = x^3 + 2x + 3 over F_97.  The trace must come from somewhere: here a_q = -2
// (from a point count; in production plug a Schoof/SEA resolver into TraceResolver).
EllipticCurveNode e(2, 3, 97);
auto node = FrobeniusGenerator::lift(e.with_trace(-2));      // optional<FrobeniusNode>, F = [[0,-97],[1,-2]]

auto n3   = LefschetzTracePass::count_points(node->F, 3);    // #E(F_{97^3}) = 912100
auto zeta = ZetaFunctionPass::collapse(node->F, node->q);    // (1 + 2T + 97T^2) / (1 - 98T + 97T^2)
auto cert = WeilProver::certify(*node);                      // hasse.verdict == Verdict::PROVED, ...
```

### Python

```bash
PYTHONPATH=build/python python3
```

```python
import symnet_native as sn
from symnet_native import dsl, finite_field as ff

# --- word folding (fixed-size) ---------------------------------------------------------
r, s = [0, -1, 1, 0], [-1, 0, 0, 1]
sn.eval_word_compiled(2, r + s + r, [0, 0, 0, 1], -0.5)     # True: r*s*r == s
cw = sn.CompiledWord(2, (r + s) * 4096)                     # fold once ...
cw.decide([0, 0, 0, 1], -0.5)                               # ... decide with 0 matrix passes

# --- the DSL ----------------------------------------------------------------------------
interp = dsl.Interpreter()
interp.run(source).passed                                   # run every query in a program
prog = interp.compile(xor_source)                           # inspect what the synthesizer found
prog.groups[0].classifiers[0].lifts                         # ['k1 = abs(x - y)']
job = prog.groups[0].tasks[0].jobs[0]
job.folded_factors, job.folded_matrix                       # constant folding, visible
job.evaluate(trace=True).decision                           # one k x k pass

# --- exact finite-field passes ----------------------------------------------------------
e = ff.EllipticCurve(2, 3, 97)
node = ff.lift(e.with_trace(-2))                            # F = [[0, -97], [1, -2]]
ff.count_points(node.F, 3)                                  # 912100  = #E(F_97^3)
str(ff.zeta_collapse(node.F, 97))                           # '(1 + 2T + 97T^2) / (1 - 98T + 97T^2)'
ff.certify(node).hasse.verdict                              # Verdict.PROVED
```

`eval_word_compiled` folds on every call (L matrix passes); `CompiledWord` folds once and then decides in $O(1)$. The trace $a_q$ must be supplied (`with_trace`) or produced by a resolver callable (`e.resolve(fn)`); no point counting is implemented. Install with `cmake --install build --prefix ~/.local`.

---

## DSL Reference

A program is a list of `group Name { ... }` blocks. `#` starts a comment.

| Statement | Syntax | Meaning |
|---|---|---|
| `gen` | `gen r = rotation(90)` · `gen s = matrix([[-1,0],[0,1]])` · `gen R(theta) = phasor(theta)` | Declare a generator. Kinds: `matrix`, `rotation` (degrees), `reflection` (degrees), `phasor` (radians). `R(theta)` declares a continuous parametrised family. |
| `lift` | `lift z = abs(x - y)` | Declare a non-linear feature layer appended to every classifier. |
| `subgroup` | `subgroup H = span(r, s)` | Finite subgroup by BFS closure (≤ 2048 elements). |
| `separate` | `separate C { weight = [0,0,1], bias = -0.5 }` | A hand-written hyperplane classifier. |
| `sector` | `sector Wedge = [pi/4, 3*pi/4]` | Closed angular sector on $SO(2)$, compiled to a perceptron bank with AND/OR output. |
| `space` | `space P { (0,0) class 0  (1,0) class 1 ... }` | Labelled point set. |
| `synthesize` | `synthesize C on P` · `synthesize C of H over D [coset g]` | Let the kernel synthesizer discover the lifts and hyperplane. |
| `eval` | `eval r * s * r [on (x, y)]` | Evaluate a word to a matrix (element mode) or to a point (point mode). |
| `classify` | `classify C on (1,0)` · `classify C of r * s [on (..)] [expect 0/1]` | Run a classifier on a point or on a group element. |
| `verify` | `verify C = H over D` | Check a classifier against subgroup membership for every element of `D`. |
| `check` | `check R(pi/4) * R(pi/4) == R(pi/2)` | Check that two words fold to the same matrix. |
| `sweep` | `sweep Wedge over R from 0 to 2*pi steps 360` | Compare a sector classifier with the analytic arc test over a half-open grid of angles. |

**Words:** `g1 * g2 * ...`, each factor `name`, `name(expr)`, optionally `^n` (negative `n` inverts). `e` is the identity.
**Expressions:** `+ - * /`, unary `-`, parentheses; functions `abs sqrt relu tanh sin cos`; constant `pi`.

```
group D4 {
    gen r = rotation(90)
    gen s = matrix([[-1, 0], [0, 1]])
    subgroup D = span(r, s)
    subgroup H = span(s)
    separate InH { weight = [0, 0, 0, 1], bias = -0.5 }
    classify InH of r * s * r expect 1
    verify InH = H over D
}
```

---

## API Reference

### Python API (`symnet_native`)

#### Top level

| Name | Signature | Returns | Description |
|---|---|---|---|
| `eval_word_compiled` | `(k, factors, weight, bias)` | `bool` | Decision $\mathrm{Step}(w \cdot \mathrm{vec}(W_{\text{compiled}}) + b)$ for a word. `factors` is $L$ row-major $k \times k$ matrices concatenated in word order (empty list = identity); `weight` has $k^2$ entries; `k` ∈ {2, 3, 4}. Folds on every call. |
| `CompiledWord` | `(k, factors)` | object | A word folded once into a single $k \times k$ matrix. |
| `Matrix2x2`, `Matrix3x3`, `Matrix4x4` | | classes | Fixed-size row-major `double` matrices. |
| `gemm_count()` / `reset_gemm_count()` | | `int` / `None` | Matrix-multiplication passes executed so far (shared by every submodule). |
| `dsl`, `finite_field` | | submodules | See below. Also importable as `symnet_native.dsl`, `symnet_native.finite_field`. |
| `__version__` | | `str` | Module version. |

**`CompiledWord`**: properties `dim`, `length` (factors folded), `matrix` ($W_{\text{compiled}}$, row-major); method `decide(weight, bias) -> bool` with **0 matrix passes**.

**`Matrix2x2` / `Matrix3x3` / `Matrix4x4`** (replace $N$ by 2, 3 or 4)

| Member | Kind | Description |
|---|---|---|
| `Matrix()` | constructor | Identity matrix. |
| `identity()`, `zero()` | static | Identity / all-zero matrix. |
| `from_list(flat)` | static | Build from $N^2$ row-major numbers; `ValueError` otherwise. |
| `dim` | class property | $N$ (read from the class, e.g. `Matrix3x3.dim`). |
| `to_list()` | method | Row-major list of $N^2$ floats. |
| `m[i, j]`, `m[i, j] = v` | item access | Get/set; `IndexError` if out of range. |
| `a * b`, `a @ b` | operators | Matrix product (counts one pass). |
| `apply(x)` | method | Matrix-vector product with a length-$N$ list. |
| `max_abs_diff(other)` | method | $\max_{ij} \lvert a_{ij} - b_{ij} \rvert$. |
| `a == b` | operator | Exact element-wise equality. |

#### Exceptions (top level)

| Exception | Raised for |
|---|---|
| `SymNetError` | Base class of everything below. |
| `LexError` | Unexpected character in DSL source. |
| `ParseError` | Syntax error. |
| `CompileError` | Unknown names, dimension mismatches, no separating lift, ... |
| `SymNetRuntimeError` | Run-time errors, e.g. `Job.decide()` on a job with no hyperplane. |
| `ValueError`, `IndexError` | Bad sizes / indices on matrix types and `finite_field.Mat2`. |
| `OverflowError` | A `finite_field` result that does not fit in `int64`. |

#### `symnet_native.dsl`

| Name | Signature | Returns | Description |
|---|---|---|---|
| `Interpreter` | `(sharpness=10.0)` | object | Lex, parse, compile, fold and run DSL programs. `sharpness` is the slope of the soft sigmoid score. |
| `Interpreter.run` | `(source, transcript=False)` | `RunResult` | Compile and run every query. `transcript=True` captures the full Python-style transcript. |
| `Interpreter.compile` | `(source)` | `Program` | Compile without running. |
| `tokenize` | `(source)` | `list[(kind, text, line)]` | `kind` is `'number'`, `'ident'`, `'symbol'` or `'eof'`. |

| Class | Members |
|---|---|
| `RunResult` | `passed`, `failed`, `x_checked`, `x_failed` (folded-vs-reference equivalence checks / mismatches), `transcript`, `ok` |
| `Program` | `groups` → `list[Group]` |
| `Group` | `name`, `declarations` (compiler log incl. `[synth]` lines), `classifiers` → `list[Classifier]` (sorted by name), `tasks` → `list[Task]` |
| `Classifier` | `name`, `weight`, `bias`, `lifts` (e.g. `['k1 = abs(x - y)']`), `is_sector` |
| `Task` | `kind` (`'job'`, `'verify'` batch, `'check'` = lhs, rhs), `label`, `jobs` → `list[Job]` |
| `Job` | `kind` (`'eval'`/`'classify'`), `label`, `short_label`, `expected`, `input`, `element_mode`, `dim`, `matrix_shape`, `architecture` (unfolded), `folded_architecture`, `folded_factors`, `folded_matrix` ($W_{\text{compiled}}$); methods `evaluate(trace=False, reference=False) -> ForwardResult`, `decide() -> bool` |
| `ForwardResult` | `input_vector`, `final_vector`, `decision`, `score`, `soft_score` (the last three are `None` for `eval` jobs), `trace` → `list[(label, vector)]` |

`Job.evaluate()` runs the constant-folded graph (one $k \times k$ pass); `reference=True` runs the unfolded Kronecker-expanded oracle, which yields the per-layer trace.

#### `symnet_native.finite_field`

| Name | Signature | Returns | Description |
|---|---|---|---|
| `EllipticCurve` | `(a, b, q)` | object | $y^2 = x^3 + ax + b$ over $\mathbb{F}_q$; properties `a`, `b`, `q`, `trace` (`None` if unresolved); methods `valid()`, `with_trace(a_q)`, `resolve(resolver)`. |
| `schoof_trace_stub` | `(curve)` | `None` | Explicit stub resolver; no point counting is implemented. |
| `lift` | `(curve)` | `FrobeniusNode \| None` | $F = \begin{pmatrix}0 & -q\\ 1 & a_q\end{pmatrix}$; `None` if invalid or the trace is unresolved. |
| `lift_to_engine` | `(curve)` | `Matrix2x2 \| None` | Same lift as a double matrix; `None` if an entry exceeds $2^{53}$. |
| `trace` | `(F)` | `int` | $\operatorname{Tr}F$. |
| `determinant` | `(F)` | `int` | $\det F$; `OverflowError` if not exact. |
| `characteristic_polynomial` | `(F)` | `Poly2` | $P(T) = 1 - tT + \det(F)\,T^2$. |
| `power` | `(F, n)` | `Mat2` | $F^n$ (≤ 64 steps); `OverflowError` if a needed intermediate leaves `int64`. |
| `trace_of_power` | `(F, n)` | `int \| None` | $\operatorname{Tr}(F^n)$; `None` on overflow. |
| `count_points` | `(F, n)` | `int \| None` | $\#E(\mathbb{F}_{q^n}) = \det(I - F^n)$; `None` for $n = 0$ or if not certifiable. |
| `zeta_collapse` | `(F, q)` | `ZetaNode \| None` | Closed-form zeta function; `None` if $\det F \ne q$ or on overflow. |
| `prove_hasse_weil` | `(F, q)` | `HasseWeilReport` | Exact check of $\det F = q$ and $t^2 \le 4q$. |
| `prove_functional_equation` | `(F, q)` | `FunctionalEquationReport` | Exact check of $Z(1/(qT)) = Z(T)$. |
| `certify` | `(node)` | `WeilCertificate` | Both checks for a `FrobeniusNode`. |

| Class | Members |
|---|---|
| `Mat2` | `Mat2()` identity, `Mat2(m00, m01, m10, m11)`, `identity()`, `from_list(flat)`, `to_list()`, `m[i, j]`, `a * b` / `a @ b` / `a - b` (`OverflowError` on overflow), `mul_checked(other) -> (Mat2, overflowed)`, `det()`, `adjugate()`, `transpose()`, `to_engine() -> Matrix2x2 \| None`, `from_engine(m)` (static), `==` |
| `Poly2` | `Poly2(c0, c1, c2)`, `c0`, `c1`, `c2`, `coefficients`, `==`, `str()` |
| `FrobeniusNode` | `FrobeniusNode(F, q)`, `F`, `q` |
| `ZetaNode` | `num`, `den` (both `Poly2`), `str()` → `'(1 + 2T + 97T^2) / (1 - 98T + 97T^2)'` |
| `Verdict` | enum `REFUTED`, `PROVED`; `bool(v)` is `True` for `PROVED` |
| `HasseWeilReport` | `verdict`, `trace`, `det_is_q`, `delta_negative`, `delta_zero`, `overflow` |
| `FunctionalEquationReport` | `verdict`, `inverse_collapse`, `weil_pairing`, `numerator_reflects`, `denominator_reflects`, `overflow` |
| `WeilCertificate` | `hasse`, `functional_equation` |

`EllipticCurve.resolve(resolver)` takes any callable `resolver(curve) -> int | None` (a Schoof/SEA implementation, a lookup table, ...).

### C++ API: DSL engine

| Header | Symbol | Description |
|---|---|---|
| `Lexer.hpp` | `tokenize(source)` | Regex tokenizer → `std::vector<Token>`. |
| `Parser.hpp` | `Parser(tokens).parseProgram()` | Recursive-descent parse → `std::vector<GroupDecl>`. |
| `Compiler.hpp` | `SymNetCompiler::compileProgram(program)` | AST → `CompiledGroup` (declarations + `Task`s: `Job`, `VerifyJob`, `CheckJob`). |
| `Compiler.hpp` | `foldNetwork(network)` | Constant-folding pass → `FoldedNetwork` (`Wc`, `foldedFactors`, lift tapes, bank, hyperplane). |
| `Compiler.hpp` | `KernelSynthesizer::synthesize(F, labels, names)` | Perceptron cycle detection and lift search → `SynthesisResult`. |
| `Compiler.hpp` | `ContinuousSO2Group` | `phasor`, `matrix`, `rotation`, `matrixFromPhasor`, `isRotation`, `inSector`, `sectorPerceptrons`. |
| `Runtime.hpp` | `SymNetRuntime::forward(net, x, trace)` | Folded execution → `ForwardResult` (features, decision, score, soft score, trace). |
| `Runtime.hpp` | `SymNetRuntime::decideFast(net, x, scratch)` | Allocation-free decision. |
| `Runtime.hpp` | `forwardReference`, `decideUnfolded`, `decideKron` | Reference oracle and benchmark baselines. |
| `Interpreter.hpp` | `SymNetInterpreter::run(source)` / `compile(source)` | End-to-end run → `Report` / compile only. |
| `WordEval.hpp` | `evalWordCompiled`, `foldWord`, `CompiledWord` | Fixed-size ($k \le 4$) word folding on `alignas(64)` buffers; used by the bindings. |
| `MatrixEngine.hpp` | `Mat<N>`, `gemm`, `gemv`, `dot`, `DMat`, `kron`, `inverse`, `gemmCounter()` | Matrix primitives and pass instrumentation. |

### C++ API: finite-field passes

All symbols are in `namespace symnet::ff`.

**`Mat2.hpp`: exact integer buffer**

| Symbol | Description |
|---|---|
| `Mat2` | 64-byte-aligned `int64[4]` row-major $2 \times 2$ matrix; `Mat2::identity()`. |
| `Checked<T>` | `{ value, overflow }`: a result plus a sticky overflow flag. |
| `Poly2` | Quadratic `c[0] + c[1]T + c[2]T²` with exact integer coefficients; `==`. |
| `ScaledMat2` | `{ num, den }`: a rational matrix, so inverses stay integral. |
| `mul(A, B)`, `sub(A, B)` | `Checked<Mat2>` product / difference (`mul` counts one pass in `gemmCounter()`). |
| `det(A)`, `adjugate(A)`, `transpose(A)` | Determinant and adjugate (both `Checked`), transpose. |
| `equal(A, B)`, `select(c, A, B)` | Lane-wise equality; branch-free select. |
| `add_ovf`, `sub_ovf`, `mul_ovf` | Scalar arithmetic that reports overflow. |
| `U128`, `mul64`, `gt`, `eq`, `magnitude` | 128-bit helpers for the exact Hasse comparison. |
| `toEngine(A)` / `fromEngine(E)` | `optional<symnet::Matrix2x2>` / `optional<Mat2>`; `nullopt` unless exact ($\lvert\cdot\rvert \le 2^{53}$, integer-valued). |

**`finite_field/FrobeniusGenerator.hpp`**

| Symbol | Description |
|---|---|
| `EllipticCurveNode(a, b, q)` | AST node for $y^2 = x^3 + ax + b$ over $\mathbb{F}_q$ ($a, b$ reduced mod $q$). Accessors `a()`, `b()`, `q()`, `trace()`. |
| `.valid()` | $3 < q \le 2^{62}$, $q$ odd, $q \bmod 3 \ne 0$, $4a^3 + 27b^2 \not\equiv 0$. Primality of $q$ is the caller's guarantee. |
| `.with_trace(t)` / `.resolve(resolver)` | Attach a known trace / ask a `TraceResolver`. |
| `TraceResolver` | `std::function<optional<int64>(const EllipticCurveNode&)>`: plug Schoof/SEA here. |
| `schoof_trace_stub` | Explicit stub that always returns `nullopt`. |
| `FrobeniusNode` | `{ Mat2 F; int64 q; }` |
| `FiniteFieldNode` | `std::variant<EllipticCurveNode, FrobeniusNode>` |
| `FrobeniusGenerator::lift(curve)` | → `optional<FrobeniusNode>`; `nullopt` if invalid or trace unresolved. Also accepts a `FiniteFieldNode`. |
| `FrobeniusGenerator::lift_to_engine(curve)` | Same lift as `symnet::Matrix2x2`; `nullopt` if an entry exceeds $2^{53}$. |

**`finite_field/LefschetzTracePass.hpp`**

| Function | Returns | Description |
|---|---|---|
| `trace(F)` | `int64` | $\mathrm{Tr}F$ (AVX2 diagonal sum when available). |
| `determinant(F)` | `Checked<int64>` | $\\det F$. |
| `characteristic_polynomial(F)` | `Checked<Poly2>` | $P(T) = 1 - tT + \\det(F)\,T^2$. |
| `power(F, n)` | `Checked<Mat2>` | $F^n$, binary exponentiation (≤ 64 steps), branch-free. |
| `trace_of_power(F, n)` | `optional<int64>` | $\mathrm{Tr}(F^n) = \alpha^n + \beta^n$; `nullopt` on overflow. |
| `count_points(F, n)` | `optional<int64>` | $\\#E(\mathbb{F}_{q^n}) = \det(I - F^n)$; `nullopt` for $n = 0$ or if not certifiable in `int64`. |

**`finite_field/ZetaFunctionPass.hpp`**

| Symbol | Description |
|---|---|
| `ZetaNode` | `{ Poly2 num, den; }`: $Z(T) = \text{num}/\text{den}$. |
| `ZetaFunctionPass::collapse(F, q)` | → `optional<ZetaNode>`: num $= 1 - tT + qT^2$, den $= (1-T)(1-qT)$. Rejects $\det F \ne q$ or overflow. |
| `ZetaFunctionPass::collapse_ast(F, q)` | Heap-allocated form (`unique_ptr`, null on failure). |
| `to_string(Poly2)`, `to_string(ZetaNode)` | Diagnostic pretty-printers. |

**`finite_field/WeilProver.hpp`**

| Symbol | Description |
|---|---|
| `Verdict` | `REFUTED` / `PROVED`; `to_string(v)`. |
| `WeilProver::prove_hasse_weil(F, q)` | → `HasseWeilReport { verdict, trace, det_is_q, delta_negative, delta_zero, overflow }`. |
| `WeilProver::prove_functional_equation(F, q)` | → `FunctionalEquationReport { verdict, inverse_collapse, weil_pairing, numerator_reflects, denominator_reflects, overflow }`. |
| `WeilProver::certify(node)` | → `WeilCertificate { hasse, functional_equation }`. |

---

## Benchmark

D4 word `(r*s)^(L/2)` classified against the subgroup `{e, s}`, one decision per pass (x86_64, `-O3 -march=native`; absolute numbers vary by machine and run):

| word length L | folded (this engine) | per-factor O(L k³) | dense Kronecker O(L k⁴) |
|---:|---:|---:|---:|
| 3    | 9 ns | 22 ns   | 61 ns    |
| 128  | 9 ns | 790 ns  | 2.2 µs   |
| 8192 | 9 ns | 53 µs   | **145 µs** |

The folded pass stays at about 9 ns from $L = 3$ to $L = 8192$. Reproduce with `build/tests/unit_tests`.

---

## Validation

| Suite | File | Checks | What it establishes |
|---|---|---:|---|
| DSL engine | `tests/cpp/test_engine.cpp` (`--quick`) | 1,553 | XOR / Klein-4 / $D_4$ lift synthesis, one-pass folding, $SO(2)$ composition to $< 10^{-15}$, dense sector sweeps, folded ≡ reference |
| Finite field | `tests/cpp/finite_field_test.cpp` | 48,209 | All 3,190 nonsingular curves over $\mathbb{F}p$, $p \in {5,\dots,31}$: trace, $\det$, $P(T)$, $\\#E(\mathbb{F}p)$, both Weil theorems, zeta recurrence $kZ_k = \sum N_j Z{k-j}$; $\\#E(\mathbb{F}{p^2})$ vs a direct count over $\mathbb{F}{p^2}$; worked example over $\mathbb{F}{97}$; negative tests |
| Exact vs engine | `tests/cpp/engine_integration_test.cpp` | 22,470 | `ff::mul` / `ff::power` agree with `symnet::Matrix2x2` and `symnet::inverse` where doubles are exact; conversions refuse inexact values; large $q \approx 2^{62}$ is certified by the exact path while the double engine refuses it |
| Binding facade | `tests/cpp/test_api.cpp` | 53 | Everything the Python `dsl` / `finite_field` API does, tested natively: tokenizer, `run`, error classes, synthesized-lift inspection, folded matrix, one-pass evaluation, folded ≡ reference, `Checked<T>` → `OverflowError` |
| Python | `tests/python/test_bindings.py` | 77 | Matrix types, `eval_word_compiled`, `CompiledWord` (8192 factors fold once; 10,000 `decide()` calls use 0 passes), the `dsl` and `finite_field` APIs, exception hierarchy |

All six CTest suites (unit tests, finite field, engine integration, binding facade, single-header build, Python bindings) pass in a CMake build against a real pybind11 on Linux (WSL), including the 77 Python-side checks.

The brute-force point counters in `finite_field_test.cpp` are **test-only oracles**; the headers contain no point-counting loop. The finite-field counts above were obtained with `g++ 13 -O3 -march=native`, and the 48,209 checks also passed with the AVX2 and `__builtin_mul_overflow` paths disabled (`-mno-avx2 -DSYMNET_FF_PORTABLE_ARITH`).

---

## Limitations and Honest Notes

- **Computing $a_q$ is not $O(1)$.** Schoof/SEA is polynomial in $\log q$. The library carries the trace as an externally supplied value (`with_trace` or a `TraceResolver`); `schoof_trace_stub` is an explicit placeholder, and no point-counting loop exists in the headers.
- **$F^n$ is not $O(1)$ in $n$.** $\mathrm{Tr}(F^n)$ has $\Theta(n \log q)$ bits, so exact output cannot be constant-time. The cost is at most 64 binary-exponentiation steps, and results that leave `int64` are reported as `nullopt`/`overflow`, never wrapped (for $q = 97$, `count_points` stops being certifiable by $n = 40$).
- **The $2 \times 2$ matrix is a representative.** Frobenius is defined only up to $GL_2(\mathbb{Q}_\ell)$-conjugacy. Its entries carry no meaning; its conjugation invariants (trace, determinant, characteristic polynomial, traces of powers, the zeta function) are the meaningful content.
- **Preconditions.** $q$ must be an odd prime with $3 < q \le 2^{62}$; primality is assumed, not tested.
- **Python `Program` objects are snapshots.** Each `Job` owns a private copy of its compiled graph; compiling a very large program therefore costs memory proportional to the number of jobs. The DSL AST itself is not exposed to Python.
- **`eval_word_compiled` supports $k \in \{2, 3, 4\}$** (fixed-size buffers) and folds on every call.
- **Floating-point reproducibility.** With `-march=native`, fused multiply-add can change the last digit versus numpy; use `-DSYMNET_STRICT_FP=ON` for bit-identical results. The exact `ff` path is unaffected, since it uses no floating point.

---

## Glossary

| Term | Meaning |
|---|---|
| **AST** | Abstract syntax tree produced by the parser: `GroupDecl` → `Statement` variants and `Expr` trees. |
| **AND / OR sector logic** | Output rule of a sector's two half-plane units: AND for width $\le \pi$ (intersection), OR for width $> \pi$ (union). |
| **`Checked<T>`** | A value paired with a sticky "may have overflowed `int64`" flag. |
| **Characteristic polynomial $P(T)$** | $\det(I - TF) = 1 - tT + qT^2$; its reciprocal roots are the Frobenius eigenvalues. |
| **Companion matrix** | $\begin{pmatrix}0 & -q\\ 1 & a_q\end{pmatrix}$, a matrix whose characteristic polynomial is $x^2 - a_q x + q$. |
| **Constant folding** | Pre-multiplying all generator matrices of a word at compile time into $W_{\text{compiled}}$. |
| **Coset** | For a subgroup $H$ and element $g$, the set $gH$. |
| **Discriminant $\Delta$** | $t^2 - 4q$; Hasse–Weil holds iff $\Delta \le 0$. |
| **$D_4$** | Dihedral group of the square (8 elements), generated here by a $90°$ rotation and a reflection. |
| **Element mode / point mode** | Element mode evaluates a word to a group element, with state $X \in \mathbb{R}^{k \times k}$ starting at $I$. Point mode applies the word to a vector. |
| **Étale cohomology** | The cohomology theory whose $H^1$ of an elliptic curve is 2-dimensional and carries the Frobenius action. |
| **Folded network** | The compiled graph: one linear pass $W_{\text{compiled}}$, then lifts, an optional perceptron bank, and a hyperplane. |
| **Frobenius endomorphism $\pi_q$** | The $q$-power map on points; its action on $H^1$ is the matrix $F$. |
| **Functional equation** | For genus 1, $Z(1/(qT)) = Z(T)$. |
| **Generator** | A matrix (finite group), or a parametrised family such as $R(\theta)$, from which words are built. |
| **Hasse–Weil bound** | $\lvert a_q \rvert \le 2\sqrt{q}$, equivalently the Riemann hypothesis for $E/\mathbb{F}_q$. |
| **Hyperplane / Step** | The output unit $\mathrm{Step}(w \cdot v + b) = 1$ if $w \cdot v + b > 0$, else $0$. |
| **Klein 4-group** | $V = \{I, a, b, ab\}$, here realised as diagonal sign matrices. |
| **Kernel synthesizer** | The compiler stage that proves non-separability and injects the minimum lift layers needed. |
| **Kronecker expansion** | The dense $\mathrm{kron}(W, I)$ matrix the engine avoids at run time ($O(k^4)$ vs $O(k^3)$). |
| **Lefschetz trace formula** | $\\#E(\mathbb{F}_{q^n}) = \det(I - F^n) = q^n + 1 - \mathrm{Tr}(F^n)$. |
| **Lift (feature lift)** | A non-linear coordinate (e.g. `abs(x - y)`) appended to the feature vector so a partition becomes linearly separable. |
| **Matrix pass** | One $k \times k$ product or mat-vec, counted by `symnet::gemmCounter()`. |
| **Novikoff cycle detection** | The perceptron cycling theorem: a repeated weight vector after an update certifies non-separability. |
| **Perceptron bank** | A hidden layer of step units, used for the two half-planes of a sector. |
| **Phasor** | The complex number $e^{i\theta}$; maps to the rotation matrix $\begin{pmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{pmatrix}$. |
| **Schoof / SEA** | Polynomial-time point-counting algorithms that compute $a_q$. |
| **Sector** | A closed arc $[\text{lo}, \text{hi}]$ of the circle $SO(2)$. |
| **$SO(2)$** | The circle group of planar rotations, isomorphic to $U(1)$. |
| **Supersingular** | The case $\Delta = 0$ with $q$ a square ($t^2 = 4q$). |
| **Trace $a_q$** | $q + 1 - \\#E(\mathbb{F}_q)$; the integer trace of Frobenius. |
| **`U128`** | Portable 128-bit unsigned pair used for the overflow-free comparison $t^2$ vs $4q$. |
| **Verdict** | `PROVED` / `REFUTED` result of a Weil check. |
| **Weil pairing** | The Frobenius relation $F^{\mathsf T} J F = qJ$ (Poincaré duality on $H^1$). |
| **Word** | A product $g_1 g_2 \cdots g_L$ of generators. |
| **Zeta function $Z(T)$** | $\exp\!\big(\sum N_n T^n/n\big) = P(T)/\big((1-T)(1-qT)\big)$. |

---

## Citation

If you use SymNet in academic work, please cite the archived release:

> SymNet. Zenodo. https://doi.org/10.5281/zenodo.24123292
