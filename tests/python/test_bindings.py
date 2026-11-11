"""Tests for the symnet_native Python extension (pybind11).

Run through CTest (which sets PYTHONPATH to the build artifact directory) or directly:

    python3 tests/python/test_bindings.py [path/to/dir/containing/symnet_native.so]

If the module is not importable, the script also searches common build directories next to the
repository root (build/, build/python, build/*/python).
"""
import glob
import math
import os
import sys


def _import_module():
    try:
        import symnet_native  # noqa: F401
        return symnet_native
    except ImportError:
        pass
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = []
    if len(sys.argv) > 1:
        candidates.append(sys.argv[1])
    candidates += [os.environ.get("SYMNET_MODULE_DIR", "")]
    for pattern in ("build", "build/python", "build/*", "build/*/python", "build/lib*"):
        candidates += glob.glob(os.path.join(root, pattern))
    for d in candidates:
        if d and glob.glob(os.path.join(d, "symnet_native*")):
            sys.path.insert(0, d)
            try:
                import symnet_native  # noqa: F401
                return symnet_native
            except ImportError:
                sys.path.pop(0)
    raise SystemExit("cannot import symnet_native: build the module and set PYTHONPATH "
                     "or pass the directory containing it as the first argument")


sn = _import_module()

R = [0, -1, 1, 0]   # rotation(90)
S = [-1, 0, 0, 1]   # reflection diag(-1, 1)
W = [0, 0, 0, 1]    # InH hyperplane over vec(W_compiled), bias -0.5


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS", msg)


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


# --- fixed-size matrices -----------------------------------------------------------------
r, s = sn.Matrix2x2.from_list(R), sn.Matrix2x2.from_list(S)
check((r * s * r) == s, "Matrix2x2: r*s*r == s")
check((r @ r).to_list() == [-1, 0, 0, -1], "Matrix2x2: r@r == -I")
check(sn.Matrix2x2().to_list() == [1, 0, 0, 1], "Matrix2x2(): identity by default")
check(r[1, 0] == 1.0, "Matrix2x2: tuple indexing")
check(r.apply([1.0, 0.0]) == [0.0, 1.0], "Matrix2x2.apply rotates (1,0) to (0,1)")
c, d = math.cos(math.pi / 4), math.sin(math.pi / 4)
q = sn.Matrix2x2.from_list([c, -d, d, c])
h = sn.Matrix2x2.from_list([math.cos(math.pi / 2), -1.0, 1.0, math.cos(math.pi / 2)])
check((q * q).max_abs_diff(h) < 1e-15, "Matrix2x2: R(pi/4)^2 == R(pi/2) within 1e-15")
check(sn.Matrix3x3.dim == 3 and sn.Matrix4x4.dim == 4, "Matrix3x3 / Matrix4x4 report their dimension")
check(sn.Matrix4x4.zero().to_list() == [0.0] * 16, "Matrix4x4.zero()")
m3 = sn.Matrix3x3.from_list([1, 2, 3, 4, 5, 6, 7, 8, 9])
check((m3 * sn.Matrix3x3.identity()) == m3 and (m3 * m3)[0, 0] == 30.0, "Matrix3x3: identity and product")
m4 = sn.Matrix4x4.identity()
m4[2, 3] = 5.0
check((m4 * m4)[2, 3] == 10.0, "Matrix4x4: product and item assignment")
check(raises(ValueError, lambda: sn.Matrix2x2.from_list([1, 2, 3])), "from_list rejects wrong length")
check(raises(IndexError, lambda: r[2, 0]), "indexing rejects out-of-range")

# --- eval_word_compiled ------------------------------------------------------------------
word = R + S + R
check(sn.eval_word_compiled(2, word, W, -0.5) is True, "eval_word_compiled: r*s*r is in H")
check(sn.eval_word_compiled(2, R + S, W, -0.5) is False, "eval_word_compiled: r*s is not in H")
check(sn.eval_word_compiled(2, [], W, -0.5) is True, "eval_word_compiled: empty word is the identity (in H)")
check(raises(ValueError, lambda: sn.eval_word_compiled(5, [], [0] * 25, 0)), "rejects unsupported k")
check(raises(ValueError, lambda: sn.eval_word_compiled(2, [1, 2, 3], W, 0)), "rejects ragged factor list")
check(raises(ValueError, lambda: sn.eval_word_compiled(2, word, [1.0], 0)), "rejects wrong weight length")

# --- CompiledWord: the O(1) folded pass --------------------------------------------------
check(sn.CompiledWord(2, word).matrix == S and sn.CompiledWord(2, word).length == 3,
      "CompiledWord: folds r*s*r into the single matrix s")

L = 8192
long_word = (R + S) * (L // 2)
sn.reset_gemm_count()
cw = sn.CompiledWord(2, long_word)
check(cw.length == L and sn.gemm_count() == L, "CompiledWord: folding %d factors costs %d passes, once" % (L, L))

sn.reset_gemm_count()
decisions = {cw.decide(W, -0.5) for _ in range(10000)}
check(sn.gemm_count() == 0 and len(decisions) == 1,
      "CompiledWord.decide(): 10000 decisions use 0 matrix passes (O(1) in word length)")
check(cw.decide(W, -0.5) == sn.eval_word_compiled(2, long_word, W, -0.5),
      "CompiledWord agrees with eval_word_compiled on a long word")

sn.reset_gemm_count()
sn.eval_word_compiled(2, word, W, -0.5)
check(sn.gemm_count() == 3, "eval_word_compiled folds per call: 3 factors -> 3 passes (use CompiledWord to avoid it)")
check(raises(ValueError, lambda: sn.CompiledWord(2, [1.0, 2.0, 3.0])), "CompiledWord rejects ragged factor list")
check(raises(ValueError, lambda: cw.decide([1.0], 0.0)), "CompiledWord.decide rejects wrong weight length")

print("all symnet_native tests passed")
