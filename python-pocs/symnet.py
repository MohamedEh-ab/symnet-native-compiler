#!/usr/bin/env python3
"""
SymNet (Phase 2): a DSL that compiles group theory into a differentiable MLP graph.

    DSL text --tokenize--> tokens --Parser--> AST --SymNetCompiler--> Network
    (list of layers) --SymNetRuntime--> (final vector, boolean decision)

Layer types of the compiled network
-----------------------------------
    LinearLayer         v -> W v                    group generator (or kron(W, I))
    LiftLayer           v -> [v, f(v)]              non-linear feature lift
    PerceptronBankLayer v -> Step(W v + b)          hidden perceptron units
    HyperplaneLayer     v -> Step(w . v + b)        output perceptron

Phase 2 additions
-----------------
1. SymNetKernelSynthesizer
   Decides whether a labelled finite state space (a point set, a subgroup H of
   a finite matrix group G, or a left coset gH) is linearly separable, and if
   not, searches a library of non-linear basis functions (interaction terms,
   absolute differences, ...) for the SMALLEST set of lifts that makes it
   separable.  The synthesized LiftLayers are injected into the compiled graph.

   Separability test: the perceptron algorithm on integer/rounded data either
   converges (separable) or revisits an earlier weight vector (the perceptron
   cycling theorem), which is a certificate of NON-separability.

2. ContinuousSO2Group
   SO(2) = U(1): the phasor e^(i*theta) <-> [[cos t, -sin t], [sin t, cos t]].
   Supports parametrised generators `gen R(theta) = phasor(theta)` and angular
   sector decisions `sector S = [lo, hi]`, compiled to a two-layer perceptron
   (two half-plane hyperplanes combined by AND for width <= pi, OR otherwise).

Group words:  w = g1 * g2 * g3  is the matrix product G1 G2 G3, so the RIGHTMOST
factor is applied first.  Element mode uses kron(W, I) on vec(I) so that the
final state vec(M) can be classified by a hyperplane; point mode applies W to a
geometric point.  Angles: rotation()/reflection() use degrees, phasor() uses
radians.

Only the Python standard library and NumPy are required.
"""

from __future__ import annotations

import itertools
import math
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np


# =============================================================================
# 0. Errors and small numeric / formatting helpers
# =============================================================================

class SymNetError(Exception):
    """Base class for all SymNet errors."""


class LexError(SymNetError):
    """Raised on an unrecognised character."""


class ParseError(SymNetError):
    """Raised on a syntax error."""


class CompileError(SymNetError):
    """Raised on a semantic error while compiling the AST."""


class SymNetRuntimeError(SymNetError):
    """Raised on a shape or evaluation error in the runtime."""


def fmt_vec(v, prec: int = 3) -> str:
    """Compact vector formatting: [1, 0, 0.5]."""
    arr = np.round(np.asarray(v, dtype=float), prec) + 0.0  # +0.0 removes -0.0
    return np.array2string(arr, separator=", ",
                           formatter={"float_kind": lambda t: f"{t:g}"})


def fmt_mat(m) -> str:
    """Compact matrix formatting: [[0, -1], [1, 0]]."""
    return "[" + ", ".join(fmt_vec(row) for row in np.asarray(m, dtype=float)) + "]"


def clean_matrix(m: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Zero out floating point dust such as cos(pi/2) = 6e-17."""
    out = np.array(m, dtype=float)
    out[np.abs(out) < tol] = 0.0
    return out + 0.0


def matrix_key(M: np.ndarray) -> tuple:
    """Hashable fingerprint of a matrix, robust to float dust and -0.0."""
    return tuple((np.round(np.asarray(M, dtype=float), 6) + 0.0).ravel())


def default_names(n: int) -> List[str]:
    """Variable names for vector components: x, y, w, t, v4, v5, ..."""
    base = ["x", "y", "w", "t"]
    return [base[i] if i < 4 else f"v{i}" for i in range(n)]


# =============================================================================
# 1. Lexer
# =============================================================================

@dataclass(frozen=True)
class Token:
    kind: str   # NUMBER | IDENT | SYMBOL | EOF
    text: str
    line: int


_TOKEN_SPEC = [
    ("COMMENT", r"\#[^\n]*"),
    ("SKIP", r"[ \t\r\n]+"),
    ("NUMBER", r"\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?"),
    ("IDENT", r"[A-Za-z_][A-Za-z_0-9]*"),
    ("SYMBOL", r"[{}()\[\],=+\-*/^]"),
    ("MISMATCH", r"."),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))


def tokenize(source: str) -> List[Token]:
    """Convert DSL source text into a token list terminated by an EOF token."""
    tokens: List[Token] = []
    line = 1
    for m in _TOKEN_RE.finditer(source):
        kind, text = m.lastgroup, m.group()
        if kind in ("SKIP", "COMMENT"):
            line += text.count("\n")
        elif kind == "MISMATCH":
            raise LexError(f"line {line}: unexpected character {text!r}")
        else:
            tokens.append(Token(kind, text, line))
    tokens.append(Token("EOF", "", line))
    return tokens


# =============================================================================
# 2. AST
# =============================================================================

# ---- expressions (lifts, matrix entries, weights, biases, angles) ------------

@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class Neg:
    operand: "Expr"


@dataclass(frozen=True)
class BinOp:
    op: str
    left: "Expr"
    right: "Expr"


@dataclass(frozen=True)
class Call:
    func: str
    arg: "Expr"


Expr = Union[Num, Var, Neg, BinOp, Call]

_FUNCS = {
    "abs": abs,
    "sqrt": math.sqrt,
    "relu": lambda t: max(0.0, t),
    "tanh": math.tanh,
    "sin": math.sin,   # radians
    "cos": math.cos,   # radians
}
_CONSTS = {"pi": math.pi}


def eval_expr(node: Expr, env: Dict[str, float]) -> float:
    """Evaluate an expression tree numerically (no Python eval involved)."""
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Var):
        if node.name in env:
            return float(env[node.name])
        if node.name in _CONSTS:
            return _CONSTS[node.name]
        raise CompileError(f"unknown variable '{node.name}'")
    if isinstance(node, Neg):
        return -eval_expr(node.operand, env)
    if isinstance(node, Call):
        return float(_FUNCS[node.func](eval_expr(node.arg, env)))
    if isinstance(node, BinOp):
        a, b = eval_expr(node.left, env), eval_expr(node.right, env)
        if node.op == "+":
            return a + b
        if node.op == "-":
            return a - b
        if node.op == "*":
            return a * b
        if b == 0:
            raise SymNetRuntimeError("division by zero in expression")
        return a / b
    raise CompileError(f"bad expression node {node!r}")


def free_variables(node: Expr) -> Set[str]:
    """Names an expression depends on (constants like pi excluded)."""
    if isinstance(node, Num):
        return set()
    if isinstance(node, Var):
        return set() if node.name in _CONSTS else {node.name}
    if isinstance(node, Neg):
        return free_variables(node.operand)
    if isinstance(node, Call):
        return free_variables(node.arg)
    return free_variables(node.left) | free_variables(node.right)


def fmt_expr(node: Expr, top: bool = True) -> str:
    """Pretty-print an expression, e.g. abs(x - y) or x * t."""
    if isinstance(node, Num):
        return f"{node.value:g}"
    if isinstance(node, Var):
        return node.name
    if isinstance(node, Neg):
        return "-" + fmt_expr(node.operand, False)
    if isinstance(node, Call):
        return f"{node.func}({fmt_expr(node.arg)})"
    text = f"{fmt_expr(node.left, False)} {node.op} {fmt_expr(node.right, False)}"
    return text if top else f"({text})"


# ---- statements ---------------------------------------------------------------

@dataclass(frozen=True)
class Factor:
    """One factor of a group word: name, optional argument R(theta), exponent."""
    name: str
    exp: int = 1
    arg: Optional[Expr] = None


Word = List[Factor]


def fmt_word_items(items: Word) -> str:
    """Pretty print a parsed word such as R(pi / 4) * s^2."""
    parts = []
    for f in items:
        s = f.name
        if f.arg is not None:
            s += f"({fmt_expr(f.arg)})"
        if f.exp != 1:
            s += f"^{f.exp}"
        parts.append(s)
    return " * ".join(parts)


@dataclass
class GenDecl:
    name: str
    kind: str                                  # matrix | rotation | reflection | phasor
    rows: Optional[List[List[Expr]]]
    angle: Optional[Expr]
    line: int
    param: Optional[str] = None                # name of theta in `gen R(theta) = ...`


@dataclass
class LiftDecl:
    name: str
    expr: Expr
    line: int


@dataclass
class SubgroupDecl:
    name: str
    generators: List[str]
    line: int


@dataclass
class SeparateDecl:
    name: str
    weight: List[Expr]
    bias: Expr
    line: int


@dataclass
class SectorDecl:
    name: str
    lo: Expr
    hi: Expr
    line: int


@dataclass
class SpaceDecl:
    name: str
    points: List[List[Expr]]
    labels: List[int]
    line: int


@dataclass
class SynthesizeStmt:
    name: str
    space: Optional[str]
    subgroup: Optional[str]
    ambient: Optional[str]
    coset: Optional[Word]
    line: int


@dataclass
class EvalStmt:
    word: Word
    point: Optional[List[Expr]]
    line: int


@dataclass
class ClassifyStmt:
    classifier: str
    point: Optional[List[Expr]]
    word: Optional[Word]
    expect: Optional[int]
    line: int


@dataclass
class VerifyStmt:
    classifier: str
    subgroup: str
    ambient: str
    line: int


@dataclass
class CheckStmt:
    lhs: Word
    rhs: Word
    line: int


@dataclass
class SweepStmt:
    classifier: str
    gen: str
    start: Expr
    stop: Expr
    steps: int
    line: int


Statement = Union[GenDecl, LiftDecl, SubgroupDecl, SeparateDecl, SectorDecl, SpaceDecl,
                  SynthesizeStmt, EvalStmt, ClassifyStmt, VerifyStmt, CheckStmt, SweepStmt]


@dataclass
class GroupDecl:
    name: str
    statements: List[Statement]
    line: int


# =============================================================================
# 3. Parser (recursive descent)
# =============================================================================

class Parser:
    """
    Grammar
    -------
    program   := group*
    group     := 'group' IDENT '{' statement* '}'
    statement :=
        'gen' IDENT ('(' IDENT ')')? '=' ( 'matrix' '(' '[' row (',' row)* ']' ')'
                                         | ('rotation'|'reflection'|'phasor') '(' expr ')' )
      | 'lift' IDENT '=' expr
      | 'subgroup' IDENT '=' 'span' '(' IDENT (',' IDENT)* ')'
      | 'separate' IDENT '{' 'weight' '=' '[' expr,* ']' ','? 'bias' '=' expr '}'
      | 'sector' IDENT '=' '[' expr ',' expr ']'                 # radians, ccw from lo to hi
      | 'space' IDENT '{' ( '(' expr,* ')' 'class' NUMBER ','? )* '}'
      | 'synthesize' IDENT ( 'on' IDENT | 'of' IDENT 'over' IDENT ('coset' word)? )
      | 'eval' word ('on' '(' expr,* ')')?
      | 'classify' IDENT ( 'on' '(' expr,* ')' | 'of' word ('on' '(' expr,* ')')? ) ('expect' NUMBER)?
      | 'verify' IDENT '=' IDENT 'over' IDENT
      | 'check' word '=' '=' word
      | 'sweep' IDENT 'over' IDENT 'from' expr 'to' expr 'steps' NUMBER
    word      := factor ('*' factor)*      factor := IDENT ('(' expr ')')? ('^' '-'? NUMBER)?
    expr      := term (('+'|'-') term)*    term := unary (('*'|'/') unary)*
    unary     := '-' unary | atom          atom := NUMBER | IDENT | IDENT '(' expr ')' | '(' expr ')'
    """

    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.i = 0

    # ---- token helpers -------------------------------------------------------
    def peek(self) -> Token:
        return self.toks[self.i]

    def advance(self) -> Token:
        tok = self.toks[self.i]
        if tok.kind != "EOF":
            self.i += 1
        return tok

    def at_symbol(self, s: str) -> bool:
        t = self.peek()
        return t.kind == "SYMBOL" and t.text == s

    def accept_symbol(self, s: str) -> bool:
        if self.at_symbol(s):
            self.advance()
            return True
        return False

    def expect_symbol(self, s: str) -> Token:
        t = self.peek()
        if not self.at_symbol(s):
            raise ParseError(f"line {t.line}: expected '{s}' but found {t.text or 'end of input'!r}")
        return self.advance()

    def at_ident(self, word: Optional[str] = None) -> bool:
        t = self.peek()
        return t.kind == "IDENT" and (word is None or t.text == word)

    def expect_ident(self, word: Optional[str] = None) -> Token:
        t = self.peek()
        if not self.at_ident(word):
            wanted = f"'{word}'" if word else "an identifier"
            raise ParseError(f"line {t.line}: expected {wanted} but found {t.text or 'end of input'!r}")
        return self.advance()

    def expect_number(self) -> float:
        neg = self.accept_symbol("-")
        t = self.peek()
        if t.kind != "NUMBER":
            raise ParseError(f"line {t.line}: expected a number but found {t.text or 'end of input'!r}")
        self.advance()
        return -float(t.text) if neg else float(t.text)

    # ---- program / group -----------------------------------------------------
    def parse_program(self) -> List[GroupDecl]:
        groups: List[GroupDecl] = []
        while self.peek().kind != "EOF":
            groups.append(self.parse_group())
        if not groups:
            raise ParseError("empty program: expected at least one 'group' block")
        return groups

    def parse_group(self) -> GroupDecl:
        kw = self.expect_ident("group")
        name = self.expect_ident().text
        self.expect_symbol("{")
        stmts: List[Statement] = []
        while not self.at_symbol("}"):
            if self.peek().kind == "EOF":
                raise ParseError(f"line {self.peek().line}: unterminated group '{name}'")
            stmts.append(self.parse_statement())
        self.expect_symbol("}")
        return GroupDecl(name, stmts, kw.line)

    # ---- statements ----------------------------------------------------------
    def parse_statement(self) -> Statement:
        t = self.peek()
        if t.kind != "IDENT":
            raise ParseError(f"line {t.line}: expected a statement keyword but found {t.text!r}")
        handlers = {
            "gen": self.parse_gen, "lift": self.parse_lift,
            "subgroup": self.parse_subgroup, "separate": self.parse_separate,
            "sector": self.parse_sector, "space": self.parse_space,
            "synthesize": self.parse_synthesize, "eval": self.parse_eval,
            "classify": self.parse_classify, "verify": self.parse_verify,
            "check": self.parse_check, "sweep": self.parse_sweep,
        }
        if t.text not in handlers:
            raise ParseError(f"line {t.line}: unknown statement '{t.text}'")
        return handlers[t.text]()

    def parse_gen(self) -> GenDecl:
        kw = self.expect_ident("gen")
        name = self.expect_ident().text
        param = None
        if self.accept_symbol("("):
            param = self.expect_ident().text
            self.expect_symbol(")")
        self.expect_symbol("=")
        kind = self.expect_ident().text
        if param is not None and kind == "matrix":
            raise ParseError(f"line {kw.line}: a parametrised generator needs an angle kind "
                             f"(phasor/rotation/reflection), not matrix")
        self.expect_symbol("(")
        rows: Optional[List[List[Expr]]] = None
        angle: Optional[Expr] = None
        if kind == "matrix":
            self.expect_symbol("[")
            rows = [self.parse_expr_row()]
            while self.accept_symbol(","):
                rows.append(self.parse_expr_row())
            self.expect_symbol("]")
        elif kind in ("rotation", "reflection", "phasor"):
            angle = self.parse_expr()
        else:
            raise ParseError(f"line {kw.line}: unknown generator kind '{kind}'")
        self.expect_symbol(")")
        return GenDecl(name, kind, rows, angle, kw.line, param)

    def parse_expr_row(self) -> List[Expr]:
        self.expect_symbol("[")
        items = [self.parse_expr()]
        while self.accept_symbol(","):
            items.append(self.parse_expr())
        self.expect_symbol("]")
        return items

    def parse_lift(self) -> LiftDecl:
        kw = self.expect_ident("lift")
        name = self.expect_ident().text
        self.expect_symbol("=")
        return LiftDecl(name, self.parse_expr(), kw.line)

    def parse_subgroup(self) -> SubgroupDecl:
        kw = self.expect_ident("subgroup")
        name = self.expect_ident().text
        self.expect_symbol("=")
        self.expect_ident("span")
        self.expect_symbol("(")
        gens = [self.expect_ident().text]
        while self.accept_symbol(","):
            gens.append(self.expect_ident().text)
        self.expect_symbol(")")
        return SubgroupDecl(name, gens, kw.line)

    def parse_separate(self) -> SeparateDecl:
        kw = self.expect_ident("separate")
        name = self.expect_ident().text
        self.expect_symbol("{")
        weight: Optional[List[Expr]] = None
        bias: Optional[Expr] = None
        while not self.at_symbol("}"):
            key = self.expect_ident().text
            self.expect_symbol("=")
            if key == "weight":
                self.expect_symbol("[")
                weight = [self.parse_expr()]
                while self.accept_symbol(","):
                    weight.append(self.parse_expr())
                self.expect_symbol("]")
            elif key == "bias":
                bias = self.parse_expr()
            else:
                raise ParseError(f"line {kw.line}: unknown field '{key}' in separate block")
            self.accept_symbol(",")
        self.expect_symbol("}")
        if weight is None or bias is None:
            raise ParseError(f"line {kw.line}: separate '{name}' needs both weight and bias")
        return SeparateDecl(name, weight, bias, kw.line)

    def parse_sector(self) -> SectorDecl:
        kw = self.expect_ident("sector")
        name = self.expect_ident().text
        self.expect_symbol("=")
        self.expect_symbol("[")
        lo = self.parse_expr()
        self.expect_symbol(",")
        hi = self.parse_expr()
        self.expect_symbol("]")
        return SectorDecl(name, lo, hi, kw.line)

    def parse_space(self) -> SpaceDecl:
        kw = self.expect_ident("space")
        name = self.expect_ident().text
        self.expect_symbol("{")
        points: List[List[Expr]] = []
        labels: List[int] = []
        while not self.at_symbol("}"):
            points.append(self.parse_point())
            self.expect_ident("class")
            labels.append(int(self.expect_number()))
            self.accept_symbol(",")
        self.expect_symbol("}")
        if not points:
            raise ParseError(f"line {kw.line}: space '{name}' has no points")
        return SpaceDecl(name, points, labels, kw.line)

    def parse_synthesize(self) -> SynthesizeStmt:
        kw = self.expect_ident("synthesize")
        name = self.expect_ident().text
        if self.at_ident("on"):
            self.advance()
            return SynthesizeStmt(name, self.expect_ident().text, None, None, None, kw.line)
        self.expect_ident("of")
        sub = self.expect_ident().text
        self.expect_ident("over")
        amb = self.expect_ident().text
        coset = None
        if self.at_ident("coset"):
            self.advance()
            coset = self.parse_word()
        return SynthesizeStmt(name, None, sub, amb, coset, kw.line)

    def parse_point(self) -> List[Expr]:
        self.expect_symbol("(")
        pts = [self.parse_expr()]
        while self.accept_symbol(","):
            pts.append(self.parse_expr())
        self.expect_symbol(")")
        return pts

    def parse_word(self) -> Word:
        items: Word = []
        while True:
            name = self.expect_ident().text
            arg = None
            if self.at_symbol("("):
                self.advance()
                arg = self.parse_expr()
                self.expect_symbol(")")
            exp = 1
            if self.accept_symbol("^"):
                value = self.expect_number()
                if not float(value).is_integer():
                    raise ParseError(f"line {self.peek().line}: exponent must be an integer")
                exp = int(value)
            items.append(Factor(name, exp, arg))
            if not self.accept_symbol("*"):
                return items

    def parse_eval(self) -> EvalStmt:
        kw = self.expect_ident("eval")
        word = self.parse_word()
        point = None
        if self.at_ident("on"):
            self.advance()
            point = self.parse_point()
        return EvalStmt(word, point, kw.line)

    def parse_classify(self) -> ClassifyStmt:
        kw = self.expect_ident("classify")
        clf = self.expect_ident().text
        point: Optional[List[Expr]] = None
        word: Optional[Word] = None
        if self.at_ident("on"):
            self.advance()
            point = self.parse_point()
        elif self.at_ident("of"):
            self.advance()
            word = self.parse_word()
            if self.at_ident("on"):
                self.advance()
                point = self.parse_point()
        else:
            raise ParseError(f"line {kw.line}: expected 'on (...)' or 'of <word>' after classifier name")
        expect = None
        if self.at_ident("expect"):
            self.advance()
            expect = int(self.expect_number())
        return ClassifyStmt(clf, point, word, expect, kw.line)

    def parse_verify(self) -> VerifyStmt:
        kw = self.expect_ident("verify")
        clf = self.expect_ident().text
        self.expect_symbol("=")
        sub = self.expect_ident().text
        self.expect_ident("over")
        amb = self.expect_ident().text
        return VerifyStmt(clf, sub, amb, kw.line)

    def parse_check(self) -> CheckStmt:
        kw = self.expect_ident("check")
        lhs = self.parse_word()
        self.expect_symbol("=")
        self.expect_symbol("=")
        rhs = self.parse_word()
        return CheckStmt(lhs, rhs, kw.line)

    def parse_sweep(self) -> SweepStmt:
        kw = self.expect_ident("sweep")
        clf = self.expect_ident().text
        self.expect_ident("over")
        gen = self.expect_ident().text
        self.expect_ident("from")
        start = self.parse_expr()
        self.expect_ident("to")
        stop = self.parse_expr()
        self.expect_ident("steps")
        steps = int(self.expect_number())
        if steps < 1:
            raise ParseError(f"line {kw.line}: steps must be >= 1")
        return SweepStmt(clf, gen, start, stop, steps, kw.line)

    # ---- expressions ---------------------------------------------------------
    def parse_expr(self) -> Expr:
        node = self.parse_term()
        while self.at_symbol("+") or self.at_symbol("-"):
            op = self.advance().text
            node = BinOp(op, node, self.parse_term())
        return node

    def parse_term(self) -> Expr:
        node = self.parse_unary()
        while self.at_symbol("*") or self.at_symbol("/"):
            op = self.advance().text
            node = BinOp(op, node, self.parse_unary())
        return node

    def parse_unary(self) -> Expr:
        if self.accept_symbol("-"):
            return Neg(self.parse_unary())
        return self.parse_atom()

    def parse_atom(self) -> Expr:
        t = self.peek()
        if t.kind == "NUMBER":
            self.advance()
            return Num(float(t.text))
        if t.kind == "IDENT":
            self.advance()
            if self.at_symbol("("):
                if t.text not in _FUNCS:
                    raise ParseError(f"line {t.line}: unknown function '{t.text}'")
                self.advance()
                arg = self.parse_expr()
                self.expect_symbol(")")
                return Call(t.text, arg)
            return Var(t.text)
        if self.accept_symbol("("):
            node = self.parse_expr()
            self.expect_symbol(")")
            return node
        raise ParseError(f"line {t.line}: unexpected {t.text or 'end of input'!r} in expression")


# =============================================================================
# 4. Network IR (what the compiler emits and the runtime executes)
# =============================================================================

@dataclass
class LinearLayer:
    """v -> W v  (no bias).  Implements a group generator."""
    name: str
    W: np.ndarray


@dataclass
class LiftLayer:
    """v -> concat(v, [f(v)]).  Non-linear feature lift into R^(N+1)."""
    name: str
    var_names: Tuple[str, ...]     # name bound to each current component
    expr: Expr
    target: str                    # name of the new feature


@dataclass
class PerceptronBankLayer:
    """v -> Step(W v + b) in {0,1}^m.  Hidden layer of m hyperplane units."""
    name: str
    W: np.ndarray                  # shape (m, N)
    b: np.ndarray                  # shape (m,)


@dataclass
class HyperplaneLayer:
    """v -> Step(w . v + b)  (a single output perceptron)."""
    name: str
    weight: np.ndarray
    bias: float


Layer = Union[LinearLayer, LiftLayer, PerceptronBankLayer, HyperplaneLayer]


@dataclass
class Network:
    name: str
    input_dim: int
    layers: List[Layer]

    def architecture(self) -> str:
        """One-line description such as: input(2) -> Lift[k1] -> R^3 -> Step."""
        dim = self.input_dim
        parts = [f"input({dim})"]
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                parts.append(f"Linear[{layer.W.shape[0]}x{layer.W.shape[1]}] {layer.name}")
            elif isinstance(layer, LiftLayer):
                dim += 1
                parts.append(f"Lift[{layer.target}] -> R^{dim}")
            elif isinstance(layer, PerceptronBankLayer):
                parts.append(f"Perceptron bank[{layer.W.shape[0]}x{layer.W.shape[1]}] {layer.name}")
            else:
                parts.append(f"Step(w.v+b) {layer.name}")
        return " -> ".join(parts)


# =============================================================================
# 4b. Continuous Lie group SO(2) (complex phasors)
# =============================================================================

@dataclass
class SectorSpec:
    """Two hidden half-plane perceptrons plus the AND/OR output perceptron."""
    W: np.ndarray
    b: np.ndarray
    out_w: np.ndarray
    out_b: float
    mode: str            # 'AND' (width <= pi) or 'OR' (width > pi)


class ContinuousSO2Group:
    """SO(2) ~ U(1).   e^(i*theta)  <->  [[cos t, -sin t], [sin t, cos t]].

    The map is a group homomorphism: e^(i*a) * e^(i*b) = e^(i*(a+b)), and
    matrix multiplication of the images adds angles.  Everything here is
    exact up to floating point; `clean_matrix` removes the dust.
    """

    TOL = 1e-9

    @staticmethod
    def phasor(theta: float) -> complex:
        return complex(math.cos(theta), math.sin(theta))

    @staticmethod
    def matrix(theta: float) -> np.ndarray:
        c, s = math.cos(theta), math.sin(theta)
        return clean_matrix(np.array([[c, -s], [s, c]]))

    @staticmethod
    def matrix_from_phasor(z: complex) -> np.ndarray:
        return clean_matrix(np.array([[z.real, -z.imag], [z.imag, z.real]]))

    @classmethod
    def is_rotation(cls, M: np.ndarray) -> bool:
        M = np.asarray(M, dtype=float)
        return (M.shape == (2, 2)
                and abs(M[0, 0] - M[1, 1]) < cls.TOL
                and abs(M[0, 1] + M[1, 0]) < cls.TOL
                and abs(M[0, 0] ** 2 + M[1, 0] ** 2 - 1.0) < cls.TOL)

    @staticmethod
    def phasor_of(M: np.ndarray) -> complex:
        return complex(M[0, 0], M[1, 0])

    @staticmethod
    def angle_of(M: np.ndarray) -> float:
        return math.atan2(M[1, 0], M[0, 0])

    @classmethod
    def in_sector(cls, theta: float, lo: float, hi: float) -> bool:
        """Analytic ground truth: is theta in the closed ccw arc [lo, hi]?"""
        two_pi = 2.0 * math.pi
        rel = (theta - lo) % two_pi
        return rel <= (hi - lo) + cls.TOL or rel >= two_pi - cls.TOL

    @classmethod
    def sector_perceptrons(cls, lo: float, hi: float, eps: float = 1e-9) -> SectorSpec:
        """Compile the closed sector [lo, hi] (radians, counter-clockwise) to
        two hyperplane units + an output perceptron.

        Unit 1:  cross(u_lo, v) >= 0   i.e.  n1 . v >= 0,  n1 = (-sin lo, cos lo)
        Unit 2:  cross(v, u_hi) >= 0   i.e.  n2 . v >= 0,  n2 = ( sin hi, -cos hi)
        Width <= pi: sector = half-plane1 AND half-plane2.
        Width  > pi: sector = half-plane1 OR  half-plane2.
        `eps` makes the boundary rays belong to the sector (closed sector).
        """
        width = hi - lo
        if not (0.0 < width < 2.0 * math.pi - cls.TOL):
            raise CompileError(f"sector width {width:.4f} rad must lie in (0, 2*pi)")
        n1 = np.array([-math.sin(lo), math.cos(lo)])
        n2 = np.array([math.sin(hi), -math.cos(hi)])
        W = np.vstack([n1, n2])
        b = np.full(2, eps)
        if width <= math.pi + cls.TOL:
            return SectorSpec(W, b, np.array([1.0, 1.0]), -1.5, "AND")
        return SectorSpec(W, b, np.array([1.0, 1.0]), -0.5, "OR")


# =============================================================================
# 4c. Automated kernel lift synthesizer
# =============================================================================

@dataclass
class SynthesisResult:
    separable_before: bool
    lifts: List[LiftDecl]
    weight: np.ndarray            # hyperplane over [original features, lifts...]
    bias: float
    log: List[str]


def featurize(X: np.ndarray, names: List[str], lifts: List[LiftDecl]
              ) -> Tuple[np.ndarray, List[str]]:
    """Apply a list of lifts row by row; returns the widened matrix and names."""
    F = np.asarray(X, dtype=float)
    names = list(names)
    for ld in lifts:
        col = np.array([eval_expr(ld.expr, dict(zip(names, row))) for row in F])
        F = np.column_stack([F, col])
        names.append(ld.name)
    return F, names


class SymNetKernelSynthesizer:
    """Finds the fewest non-linear lifts that make a labelled finite set
    linearly separable by ONE hyperplane.

    Search order: k = 0 (already separable?), then k = 1, 2, ... lifts chosen
    from a library of basis functions built on the current coordinates:
        products u*v, absolute differences |u-v|, absolute sums |u+v|,
        squares u*u, magnitudes |u|, and (for few coordinates) triple products.
    Among all minimal-k solutions the one whose hyperplane has the fewest
    non-zero weights wins (ties: library order), so XOR yields z = |x - y|
    with weights [0, 0, 1].
    """

    def __init__(self, max_lifts: int = 3, max_sparse_support: int = 3,
                 max_updates: int = 5000):
        self.max_lifts = max_lifts
        self.max_sparse_support = max_sparse_support
        self.max_updates = max_updates

    # ---- separability test ---------------------------------------------------
    def perceptron_separable(self, F: np.ndarray, pos: np.ndarray) -> Optional[np.ndarray]:
        """Return augmented weights (w..., b) if F is linearly separable.

        The perceptron converges iff the data are separable.  If a weight
        vector ever repeats after an update the algorithm is cycling, which
        proves the data are NOT separable (a separable run strictly increases
        w . w* with every update, so it can never revisit a state).
        """
        n = F.shape[0]
        A = np.hstack([F, np.ones((n, 1))])
        y = np.where(pos, 1.0, -1.0)
        w = np.zeros(A.shape[1])
        seen = {tuple(np.round(w, 8))}
        updates = 0
        while updates <= self.max_updates:
            changed = False
            for i in range(n):
                if y[i] * float(A[i] @ w) <= 0.0:
                    w = w + y[i] * A[i]
                    updates += 1
                    key = tuple(np.round(w, 8))
                    if key in seen:
                        return None            # cycle: certificate of non-separability
                    seen.add(key)
                    changed = True
            if not changed:
                return w
        return None                            # no convergence within the update budget

    @staticmethod
    def _normalise(F: np.ndarray, pos: np.ndarray, cols: Tuple[int, ...],
                   w_sub: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """Rescale so the margin gap is exactly 1 and centre the bias in it."""
        w = np.zeros(F.shape[1])
        w[list(cols)] = w_sub
        s = F @ w
        gap = s[pos].min() - s[~pos].max()
        if gap <= 1e-12:
            return None
        w = w / gap
        s = F @ w
        b = -(s[pos].min() + s[~pos].max()) / 2.0
        return np.round(w, 10) + 0.0, float(round(b, 10)) + 0.0

    def clean_hyperplane(self, F: np.ndarray, pos: np.ndarray
                         ) -> Optional[Tuple[np.ndarray, float]]:
        """Separating hyperplane with the fewest non-zero weights, or None."""
        if pos.all() or (~pos).all():
            return np.zeros(F.shape[1]), (0.5 if pos.all() else -0.5)
        if self.perceptron_separable(F, pos) is None:
            return None
        n_feat = F.shape[1]
        for size in range(1, min(n_feat, self.max_sparse_support) + 1):
            for cols in itertools.combinations(range(n_feat), size):
                sol = self.perceptron_separable(F[:, cols], pos)
                if sol is not None:
                    out = self._normalise(F, pos, cols, sol[:-1])
                    if out is not None:
                        return out
        sol = self.perceptron_separable(F, pos)
        return self._normalise(F, pos, tuple(range(n_feat)), sol[:-1])

    # ---- candidate basis functions ------------------------------------------
    @staticmethod
    def candidate_library(names: List[str]) -> List[Expr]:
        V = [Var(n) for n in names]
        n = len(V)
        lib: List[Expr] = []
        pairs = list(itertools.combinations(range(n), 2))
        lib += [BinOp("*", V[i], V[j]) for i, j in pairs]                       # interaction
        lib += [Call("abs", BinOp("-", V[i], V[j])) for i, j in pairs]          # |u - v|
        lib += [Call("abs", BinOp("+", V[i], V[j])) for i, j in pairs]          # |u + v|
        lib += [BinOp("*", V[i], V[i]) for i in range(n)]                       # squares
        lib += [Call("abs", V[i]) for i in range(n)]                            # |u|
        if n <= 5:
            lib += [BinOp("*", BinOp("*", V[i], V[j]), V[k])
                    for i, j, k in itertools.combinations(range(n), 3)]         # cubic
        return lib

    @staticmethod
    def _candidate_columns(F: np.ndarray, names: List[str], library: List[Expr]
                           ) -> List[Tuple[Expr, np.ndarray]]:
        """Evaluate the library on the data; drop constants and duplicates."""
        seen = set()
        for j in range(F.shape[1]):
            col = np.round(F[:, j], 9) + 0.0
            seen.add(tuple(col))
            seen.add(tuple(-col + 0.0))
        kept: List[Tuple[Expr, np.ndarray]] = []
        for expr in library:
            col = np.array([eval_expr(expr, dict(zip(names, row))) for row in F])
            if col.max() - col.min() < 1e-12:
                continue
            key = tuple(np.round(col, 9) + 0.0)
            if key in seen:
                continue
            seen.add(key)
            kept.append((expr, col))
        return kept

    # ---- main entry ----------------------------------------------------------
    def synthesize(self, F0: np.ndarray, pos: np.ndarray, names0: List[str],
                   label: str = "") -> SynthesisResult:
        F0 = np.asarray(F0, dtype=float)
        pos = np.asarray(pos, dtype=bool)
        n_pts, dim = F0.shape
        log = [f"synthesize {label}: {n_pts} points in R^{dim}, "
               f"{int(pos.sum())} positive / {int((~pos).sum())} negative"]

        # identical points with different labels can never be separated by any lift
        seen_lbl: Dict[tuple, bool] = {}
        for row, p in zip(F0, pos):
            key = tuple(np.round(row, 9) + 0.0)
            if key in seen_lbl and seen_lbl[key] != bool(p):
                raise CompileError("the same point appears with two different labels; "
                                   "no feature lift can separate it")
            seen_lbl[key] = bool(p)

        sol = self.clean_hyperplane(F0, pos)
        if sol is not None:
            log.append(f"linearly separable already in R^{dim}: no lift needed")
            return SynthesisResult(True, [], sol[0], sol[1], log)
        log.append(f"perceptron cycling certificate: NOT linearly separable in R^{dim}")

        library = self.candidate_library(names0)
        cols = self._candidate_columns(F0, names0, library)
        log.append(f"candidate basis functions after de-duplication: {len(cols)}")

        used = set(names0)
        for k in range(1, self.max_lifts + 1):
            best = None
            tried = 0
            winners = 0
            for combo in itertools.combinations(range(len(cols)), k):
                tried += 1
                F = np.column_stack([F0] + [cols[i][1] for i in combo])
                out = self.clean_hyperplane(F, pos)
                if out is None:
                    continue
                winners += 1
                cost = (int(np.count_nonzero(out[0])), combo)
                if best is None or cost < best[0]:
                    best = (cost, combo, out)
            log.append(f"k={k}: tried {tried} lift set(s), {winners} make the data separable")
            if best is not None:
                _, combo, (w, b) = best
                lifts: List[LiftDecl] = []
                for idx, ci in enumerate(combo, start=1):
                    nm = f"k{idx}"
                    while nm in used:
                        nm += "_"
                    used.add(nm)
                    lifts.append(LiftDecl(nm, cols[ci][0], 0))
                F_final, _ = featurize(F0, names0, lifts)
                ok = np.array_equal((F_final @ w + b) > 0, pos)
                if not ok:
                    raise CompileError("internal error: synthesized hyperplane does not "
                                       "reproduce the labels")
                desc = ", ".join(f"{ld.name} = {fmt_expr(ld.expr)}" for ld in lifts)
                log.append(f"minimal feature map Phi: R^{dim} -> R^{dim + k}  with  {desc}")
                log.append(f"separating hyperplane: w = {fmt_vec(w)}, b = {b:g}")
                return SynthesisResult(False, lifts, w, b, log)
        # Flat search failed: try composed lifts, k2 built on top of k1 (e.g. |k1 - w|).
        best = None
        tried = 0
        for e1, c1 in cols:
            F1 = np.column_stack([F0, c1])
            names1 = names0 + ["k1"]
            for e2, c2 in self._candidate_columns(F1, names1, self.candidate_library(names1)):
                tried += 1
                out = self.clean_hyperplane(np.column_stack([F1, c2]), pos)
                if out is None:
                    continue
                cost = (int(np.count_nonzero(out[0])), tried)
                if best is None or cost < best[0]:
                    best = (cost, e1, e2, out)
        log.append(f"composed search (k2 built on k1): tried {tried} pairs")
        if best is not None:
            _, e1, e2, (w, b) = best
            lifts = [LiftDecl("k1", e1, 0), LiftDecl("k2", e2, 0)]
            F_final, _ = featurize(F0, names0, lifts)
            if not np.array_equal((F_final @ w + b) > 0, pos):
                raise CompileError("internal error: synthesized hyperplane does not "
                                   "reproduce the labels")
            desc = ", ".join(f"{ld.name} = {fmt_expr(ld.expr)}" for ld in lifts)
            log.append(f"minimal feature map Phi: R^{dim} -> R^{dim + 2}  with  {desc}")
            log.append(f"separating hyperplane: w = {fmt_vec(w)}, b = {b:g}")
            return SynthesisResult(False, lifts, w, b, log)
        raise CompileError(f"no separating lift found with at most {self.max_lifts} basis functions "
                           f"(flat) or two composed levels")


# =============================================================================
# 5. Compiler
# =============================================================================

@dataclass
class CompiledSubgroup:
    name: str
    generator_names: Tuple[str, ...]
    elements: List[Tuple[Tuple[str, ...], np.ndarray]]   # (shortest word, matrix)
    keys: Set[tuple]

    def contains(self, M: np.ndarray) -> bool:
        return matrix_key(M) in self.keys


@dataclass
class CompiledClassifier:
    name: str
    weight: np.ndarray                          # output perceptron weights
    bias: float
    lifts: Optional[List[LiftDecl]] = None      # snapshot of lifts (synthesized classifiers)
    bank: Optional[PerceptronBankLayer] = None  # hidden units (sector classifiers)
    sector: Optional[Tuple[float, float]] = None


@dataclass
class Job:
    """One runnable query: a network plus the vector fed into it."""
    kind: str                                  # 'eval' | 'classify'
    label: str
    network: Network
    input_vec: np.ndarray
    matrix_shape: Optional[Tuple[int, int]]    # set in element mode for pretty printing
    expected: Optional[bool] = None
    short: str = ""                            # compact row label for tables


@dataclass
class VerifyJob:
    """A batch of classify jobs, each compared with its expected label."""
    label: str
    jobs: List[Job]


@dataclass
class CheckJob:
    """Asserts that two words compile to the same group element."""
    label: str
    lhs: Job
    rhs: Job


@dataclass
class CompiledGroup:
    name: str
    generators: Dict[str, np.ndarray] = field(default_factory=dict)
    param_generators: Dict[str, Tuple[str, str, Expr]] = field(default_factory=dict)
    lift_decls: List[LiftDecl] = field(default_factory=list)
    subgroups: Dict[str, CompiledSubgroup] = field(default_factory=dict)
    classifiers: Dict[str, CompiledClassifier] = field(default_factory=dict)
    spaces: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    declarations: List[str] = field(default_factory=list)
    tasks: List[Union[Job, VerifyJob, CheckJob]] = field(default_factory=list)

    @property
    def dim(self) -> Optional[int]:
        if self.generators:
            return next(iter(self.generators.values())).shape[0]
        if self.param_generators:
            return 2
        return None


class SymNetCompiler:
    """Turns the AST into Networks (generators -> Linear, lifts -> Lift,
    separate/sector blocks -> perceptron layers) and bundles them as jobs."""

    MAX_GROUP_ORDER = 2048

    def __init__(self, synthesizer: Optional[SymNetKernelSynthesizer] = None):
        self.synth = synthesizer or SymNetKernelSynthesizer()

    # ---- entry point ---------------------------------------------------------
    def compile_program(self, program: List[GroupDecl]) -> List[CompiledGroup]:
        return [self.compile_group(g) for g in program]

    def compile_group(self, decl: GroupDecl) -> CompiledGroup:
        ctx = CompiledGroup(decl.name)
        for st in decl.statements:
            try:
                self._compile_statement(ctx, st)
            except SymNetError as exc:
                raise CompileError(f"group {decl.name}, line {st.line}: {exc}") from exc
        return ctx

    def _compile_statement(self, ctx: CompiledGroup, st: Statement) -> None:
        if isinstance(st, GenDecl):
            self._compile_gen(ctx, st)
        elif isinstance(st, LiftDecl):
            ctx.lift_decls.append(st)
            ctx.declarations.append(f"lift {st.name} = {fmt_expr(st.expr)}  ->  non-linear "
                                    f"feature layer appended (R^N -> R^(N+1))")
        elif isinstance(st, SubgroupDecl):
            self._compile_subgroup(ctx, st)
        elif isinstance(st, SeparateDecl):
            self._compile_separate(ctx, st)
        elif isinstance(st, SectorDecl):
            self._compile_sector(ctx, st)
        elif isinstance(st, SpaceDecl):
            self._compile_space(ctx, st)
        elif isinstance(st, SynthesizeStmt):
            self._compile_synthesize(ctx, st)
        elif isinstance(st, EvalStmt):
            ctx.tasks.append(self._compile_eval(ctx, st))
        elif isinstance(st, ClassifyStmt):
            ctx.tasks.append(self._compile_classify(ctx, st))
        elif isinstance(st, VerifyStmt):
            ctx.tasks.append(self._compile_verify(ctx, st))
        elif isinstance(st, CheckStmt):
            ctx.tasks.append(self._compile_check(ctx, st))
        elif isinstance(st, SweepStmt):
            ctx.tasks.append(self._compile_sweep(ctx, st))
        else:
            raise CompileError(f"unsupported statement {st!r}")

    # ---- generators ----------------------------------------------------------
    @staticmethod
    def _angle_matrix(kind: str, value: float) -> np.ndarray:
        """2x2 matrix for an angle.  rotation/reflection: degrees; phasor: radians."""
        if kind == "phasor":
            return ContinuousSO2Group.matrix(value)
        a = math.radians(value)
        if kind == "rotation":
            return ContinuousSO2Group.matrix(a)
        return clean_matrix(np.array([[math.cos(2 * a), math.sin(2 * a)],
                                      [math.sin(2 * a), -math.cos(2 * a)]]))

    def _compile_gen(self, ctx: CompiledGroup, st: GenDecl) -> None:
        if st.name in ctx.generators or st.name in ctx.param_generators:
            raise CompileError(f"generator '{st.name}' already defined")
        if st.param is not None:
            extra = free_variables(st.angle) - {st.param}
            if extra:
                raise CompileError(f"angle expression uses unknown variable(s) {sorted(extra)}")
            if ctx.dim is not None and ctx.dim != 2:
                raise CompileError(f"parametrised generator is 2x2 but the group works in "
                                   f"dimension {ctx.dim}")
            ctx.param_generators[st.name] = (st.param, st.kind, st.angle)
            ctx.declarations.append(
                f"gen {st.name}({st.param}) = {st.kind}({fmt_expr(st.angle)})  ->  continuous "
                f"family of 2x2 Linear weights (instantiated per use, e.g. {st.name}(pi/4))")
            return
        if st.kind == "matrix":
            if len({len(r) for r in st.rows}) != 1:
                raise CompileError("matrix rows have different lengths")
            M = np.array([[eval_expr(e, {}) for e in row] for row in st.rows], dtype=float)
            if M.shape[0] != M.shape[1]:
                raise CompileError(f"generator matrix must be square, got {M.shape}")
            M = clean_matrix(M)
        else:
            M = self._angle_matrix(st.kind, eval_expr(st.angle, {}))
        if ctx.dim is not None and M.shape[0] != ctx.dim:
            raise CompileError(f"generator '{st.name}' is {M.shape[0]}x{M.shape[0]} "
                               f"but the group works in dimension {ctx.dim}")
        ctx.generators[st.name] = M
        ctx.declarations.append(f"gen {st.name} = {st.kind}  ->  Linear weight W = {fmt_mat(M)}")

    # ---- subgroups, classifiers ---------------------------------------------
    def _compile_subgroup(self, ctx: CompiledGroup, st: SubgroupDecl) -> None:
        if st.name in ctx.subgroups:
            raise CompileError(f"subgroup '{st.name}' already defined")
        for g in st.generators:
            if g not in ctx.generators:
                raise CompileError(f"unknown (non-parametrised) generator '{g}' in span()")
        elements = self._closure(ctx, st.generators)
        keys = {matrix_key(m) for _, m in elements}
        ctx.subgroups[st.name] = CompiledSubgroup(st.name, tuple(st.generators), elements, keys)
        ctx.declarations.append(f"subgroup {st.name} = span({', '.join(st.generators)})  ->  "
                                f"{len(elements)} elements (closure by BFS)")

    def _compile_separate(self, ctx: CompiledGroup, st: SeparateDecl) -> None:
        if st.name in ctx.classifiers:
            raise CompileError(f"classifier '{st.name}' already defined")
        w = np.array([eval_expr(e, {}) for e in st.weight], dtype=float)
        b = float(eval_expr(st.bias, {}))
        ctx.classifiers[st.name] = CompiledClassifier(st.name, w, b)
        ctx.declarations.append(f"separate {st.name}  ->  Hyperplane weight={fmt_vec(w)}, bias={b:g}")

    def _compile_sector(self, ctx: CompiledGroup, st: SectorDecl) -> None:
        if st.name in ctx.classifiers:
            raise CompileError(f"classifier '{st.name}' already defined")
        lo, hi = eval_expr(st.lo, {}), eval_expr(st.hi, {})
        spec = ContinuousSO2Group.sector_perceptrons(lo, hi)
        bank = PerceptronBankLayer(f"sector:{st.name}", spec.W, spec.b)
        ctx.classifiers[st.name] = CompiledClassifier(st.name, spec.out_w, spec.out_b,
                                                      bank=bank, sector=(lo, hi))
        ctx.declarations.append(
            f"sector {st.name} = [{lo / math.pi:.4g}*pi, {hi / math.pi:.4g}*pi] rad  ->  "
            f"2 hyperplane units n1={fmt_vec(spec.W[0])}, n2={fmt_vec(spec.W[1])} "
            f"+ {spec.mode} output (w={fmt_vec(spec.out_w)}, b={spec.out_b:g})")

    # ---- finite state spaces + kernel synthesis ------------------------------
    def _compile_space(self, ctx: CompiledGroup, st: SpaceDecl) -> None:
        if st.name in ctx.spaces:
            raise CompileError(f"space '{st.name}' already defined")
        X = np.array([[eval_expr(e, {}) for e in p] for p in st.points], dtype=float)
        y = np.array([bool(c) for c in st.labels])
        ctx.spaces[st.name] = (X, y)
        ctx.declarations.append(f"space {st.name}  ->  {X.shape[0]} labelled points in R^{X.shape[1]}")

    def _word_matrix(self, ctx: CompiledGroup, word: Word) -> np.ndarray:
        factors = self._resolve_word(ctx, word)
        M = np.eye(self._dim_for(ctx, factors))
        for _, F in factors:
            M = M @ F
        return clean_matrix(M)

    def _compile_synthesize(self, ctx: CompiledGroup, st: SynthesizeStmt) -> None:
        if st.name in ctx.classifiers:
            raise CompileError(f"classifier '{st.name}' already defined")
        elements = None
        if st.space is not None:
            if st.space not in ctx.spaces:
                raise CompileError(f"unknown space '{st.space}'")
            X, pos = ctx.spaces[st.space]
            what = f"space {st.space}"
        else:
            for nm in (st.subgroup, st.ambient):
                if nm not in ctx.subgroups:
                    raise CompileError(f"unknown subgroup '{nm}'")
            sub, amb = ctx.subgroups[st.subgroup], ctx.subgroups[st.ambient]
            elements = amb.elements
            X = np.array([M.ravel() for _, M in elements])
            if st.coset is not None:
                g = self._word_matrix(ctx, st.coset)
                target = {matrix_key(clean_matrix(g @ h)) for _, h in sub.elements}
                what = f"coset {fmt_word_items(st.coset)}*{sub.name} inside {amb.name}"
            else:
                target = sub.keys
                what = f"subgroup {sub.name} inside {amb.name}"
            pos = np.array([matrix_key(M) in target for _, M in elements])

        F0, names0 = featurize(X, default_names(X.shape[1]), ctx.lift_decls)
        res = self.synth.synthesize(F0, pos, names0, label=f"{st.name} ({what})")
        clf = CompiledClassifier(st.name, res.weight, res.bias,
                                 lifts=list(ctx.lift_decls) + res.lifts)
        ctx.classifiers[st.name] = clf
        ctx.declarations.extend("[synth] " + line for line in res.log)
        ctx.declarations.append(f"[synth] injected {len(res.lifts)} LiftLayer(s) + "
                                f"Hyperplane layer into the compiled graph for '{st.name}'")

        jobs: List[Job] = []
        if elements is None:
            for row, p in zip(X, pos):
                jobs.append(self._classify_job(ctx, clf, None, row, bool(p), fmt_vec(row)))
        else:
            for (word, _), p in zip(elements, pos):
                items = [Factor(g) for g in word]
                jobs.append(self._classify_job(ctx, clf, items, None, bool(p)))
        ctx.tasks.append(VerifyJob(
            f"synthesized classifier {st.name}: separation of all {len(jobs)} points", jobs))

    # ---- group closure -------------------------------------------------------
    def _closure(self, ctx: CompiledGroup, gen_names: List[str]
                 ) -> List[Tuple[Tuple[str, ...], np.ndarray]]:
        """All elements generated by gen_names (BFS by right-multiplication)."""
        N = ctx.generators[gen_names[0]].shape[0]
        identity = np.eye(N)
        seen = {matrix_key(identity)}
        elements: List[Tuple[Tuple[str, ...], np.ndarray]] = [((), identity)]
        frontier = [((), identity)]
        while frontier:
            nxt = []
            for word, M in frontier:
                for g in gen_names:
                    P = clean_matrix(M @ ctx.generators[g])
                    k = matrix_key(P)
                    if k not in seen:
                        seen.add(k)
                        item = (word + (g,), P)
                        elements.append(item)
                        nxt.append(item)
                        if len(elements) > self.MAX_GROUP_ORDER:
                            raise CompileError("subgroup is infinite or larger than "
                                               f"{self.MAX_GROUP_ORDER} elements")
            frontier = nxt
        return elements

    # ---- words -> layers -----------------------------------------------------
    def _resolve_word(self, ctx: CompiledGroup, word: Word) -> List[Tuple[str, np.ndarray]]:
        """Expand exponents / arguments and look up matrices (left-to-right order)."""
        factors: List[Tuple[str, np.ndarray]] = []
        for f in word:
            label = f.name
            if f.name in ctx.param_generators:
                if f.arg is None:
                    raise CompileError(f"generator '{f.name}' needs an argument, "
                                       f"e.g. {f.name}(pi/4)")
                theta = eval_expr(f.arg, {})
                param, kind, body = ctx.param_generators[f.name]
                M = self._angle_matrix(kind, eval_expr(body, {param: theta}))
                label = f"{f.name}({theta:.4g})"
            else:
                if f.arg is not None:
                    raise CompileError(f"generator '{f.name}' takes no argument")
                if f.name in ctx.generators:
                    M = ctx.generators[f.name]
                elif f.name == "e":
                    if ctx.dim is None:
                        raise CompileError("identity 'e' needs a generator to fix the dimension")
                    M = np.eye(ctx.dim)
                else:
                    raise CompileError(f"unknown generator '{f.name}'")
            if f.exp < 0:
                try:
                    M = clean_matrix(np.linalg.inv(M))
                except np.linalg.LinAlgError as exc:
                    raise CompileError(f"generator '{f.name}' is not invertible") from exc
                label += "^-1"
            for _ in range(abs(f.exp)):
                factors.append((label, M))
        return factors

    @staticmethod
    def _word_layers(factors: List[Tuple[str, np.ndarray]], N: int,
                     element_mode: bool) -> List[Layer]:
        """Rightmost factor acts first, so layers are emitted in reversed order."""
        layers: List[Layer] = []
        for step, (name, M) in enumerate(reversed(factors), start=1):
            W = np.kron(M, np.eye(N)) if element_mode else M
            layers.append(LinearLayer(f"L{step}:{name}", W))
        return layers

    @staticmethod
    def _dim_for(ctx: CompiledGroup, factors) -> int:
        N = factors[0][1].shape[0] if factors else ctx.dim
        if N is None:
            raise CompileError("cannot determine dimension: declare a generator first")
        return N

    @staticmethod
    def _eval_vector(exprs: List[Expr]) -> np.ndarray:
        return np.array([eval_expr(e, {}) for e in exprs], dtype=float)

    # ---- eval / check --------------------------------------------------------
    def _compile_eval(self, ctx: CompiledGroup, st: EvalStmt) -> Job:
        factors = self._resolve_word(ctx, st.word)
        N = self._dim_for(ctx, factors)
        label = f"eval {fmt_word_items(st.word)}"
        if st.point is not None:
            point = self._eval_vector(st.point)
            if point.shape[0] != N:
                raise CompileError(f"point has dimension {point.shape[0]}, generators have {N}")
            net = Network(label, N, self._word_layers(factors, N, element_mode=False))
            return Job("eval", f"{label} on {fmt_vec(point)}", net, point, None)
        net = Network(label, N * N, self._word_layers(factors, N, element_mode=True))
        return Job("eval", label, net, np.eye(N).ravel(), (N, N))

    def _compile_check(self, ctx: CompiledGroup, st: CheckStmt) -> CheckJob:
        lhs = self._compile_eval(ctx, EvalStmt(st.lhs, None, st.line))
        rhs = self._compile_eval(ctx, EvalStmt(st.rhs, None, st.line))
        return CheckJob(f"check {fmt_word_items(st.lhs)} == {fmt_word_items(st.rhs)}", lhs, rhs)

    # ---- classify / verify / sweep ------------------------------------------
    @staticmethod
    def _get_classifier(ctx: CompiledGroup, name: str) -> CompiledClassifier:
        if name not in ctx.classifiers:
            raise CompileError(f"unknown classifier '{name}'")
        return ctx.classifiers[name]

    def _finish_classifier_network(self, ctx: CompiledGroup, clf: CompiledClassifier,
                                   layers: List[Layer], input_dim: int, label: str) -> Network:
        """Append lift layers, optional hidden perceptron bank, then the output hyperplane."""
        lifts = clf.lifts if clf.lifts is not None else ctx.lift_decls
        dim = input_dim
        names = default_names(dim)
        for ld in lifts:
            missing = free_variables(ld.expr) - set(names)
            if missing:
                raise CompileError(f"lift '{ld.name}' uses unknown variable(s) {sorted(missing)}; "
                                   f"available: {names}")
            layers.append(LiftLayer(f"lift:{ld.name}", tuple(names), ld.expr, ld.name))
            names = names + [ld.name]
            dim += 1
        if clf.bank is not None:
            if clf.bank.W.shape[1] != dim:
                raise CompileError(f"sector '{clf.name}' works on R^{clf.bank.W.shape[1]} "
                                   f"but the state vector has dimension {dim}")
            layers.append(clf.bank)
            dim = clf.bank.W.shape[0]
        if clf.weight.shape[0] != dim:
            raise CompileError(f"classifier '{clf.name}' has {clf.weight.shape[0]} weights "
                               f"but the feature vector has dimension {dim}")
        layers.append(HyperplaneLayer(f"hyperplane:{clf.name}", clf.weight, clf.bias))
        return Network(label, input_dim, layers)

    def _classify_job(self, ctx: CompiledGroup, clf: CompiledClassifier,
                      word: Optional[Word], point: Optional[np.ndarray],
                      expected: Optional[bool], short: Optional[str] = None) -> Job:
        factors = self._resolve_word(ctx, word) if word else []
        wtxt = fmt_word_items(word) if word else "e"
        if point is not None:                                   # point mode
            N = point.shape[0]
            for name, M in factors:
                if M.shape[0] != N:
                    raise CompileError(f"point has dimension {N} but {name} is {M.shape[0]}x{M.shape[0]}")
            label = f"classify {clf.name}" + (f" of {wtxt}" if word else "") + f" on {fmt_vec(point)}"
            layers = self._word_layers(factors, N, element_mode=False)
            net = self._finish_classifier_network(ctx, clf, layers, N, label)
            return Job("classify", label, net, point, None, expected,
                       short if short is not None else fmt_vec(point))
        N = self._dim_for(ctx, factors)                         # element mode
        label = f"classify {clf.name} of {wtxt}"
        layers = self._word_layers(factors, N, element_mode=True)
        net = self._finish_classifier_network(ctx, clf, layers, N * N, label)
        return Job("classify", label, net, np.eye(N).ravel(), (N, N), expected,
                   short if short is not None else wtxt)

    def _compile_classify(self, ctx: CompiledGroup, st: ClassifyStmt) -> Job:
        clf = self._get_classifier(ctx, st.classifier)
        expected = None if st.expect is None else bool(st.expect)
        point = None if st.point is None else self._eval_vector(st.point)
        return self._classify_job(ctx, clf, st.word, point, expected)

    def _compile_verify(self, ctx: CompiledGroup, st: VerifyStmt) -> VerifyJob:
        clf = self._get_classifier(ctx, st.classifier)
        for nm in (st.subgroup, st.ambient):
            if nm not in ctx.subgroups:
                raise CompileError(f"unknown subgroup '{nm}'")
        sub, amb = ctx.subgroups[st.subgroup], ctx.subgroups[st.ambient]
        jobs = [self._classify_job(ctx, clf, [Factor(g) for g in word], None, sub.contains(M))
                for word, M in amb.elements]
        return VerifyJob(f"verify {clf.name} == membership in {sub.name} "
                         f"over all {len(amb.elements)} elements of {amb.name}", jobs)

    def _compile_sweep(self, ctx: CompiledGroup, st: SweepStmt) -> VerifyJob:
        clf = self._get_classifier(ctx, st.classifier)
        if clf.sector is None:
            raise CompileError(f"classifier '{clf.name}' is not a sector")
        if st.gen not in ctx.param_generators:
            raise CompileError(f"'{st.gen}' is not a parametrised generator")
        lo, hi = clf.sector
        a, b = eval_expr(st.start, {}), eval_expr(st.stop, {})
        jobs = []
        for k in range(st.steps):                               # half-open grid [a, b)
            theta = a + k * (b - a) / st.steps
            expected = ContinuousSO2Group.in_sector(theta, lo, hi)
            jobs.append(self._classify_job(
                ctx, clf, [Factor(st.gen, 1, Num(theta))], np.array([1.0, 0.0]), expected,
                short=f"theta={theta / math.pi:.3f}*pi"))
        return VerifyJob(f"sweep {clf.name}: {st.steps} angles of {st.gen}(theta) applied to (1, 0) "
                         f"vs analytic arc test", jobs)


# =============================================================================
# 6. Runtime
# =============================================================================

@dataclass
class ForwardResult:
    input_vector: np.ndarray
    final_vector: np.ndarray                  # state fed into the first decision layer
    decision: Optional[bool]                  # None when the network has no hyperplane
    score: Optional[float]                    # w . v + b of the output perceptron
    soft_score: Optional[float]               # sigmoid(sharpness * score): differentiable surrogate
    trace: List[Tuple[str, np.ndarray]]       # (layer name, state after that layer)


class SymNetRuntime:
    """Forward pass: Linear -> Lift -> (Perceptron bank) -> Hyperplane step.

    The hard Step gives the boolean decision.  `soft_score` replaces the final
    Step with a steep sigmoid so the same graph is differentiable.
    """

    def __init__(self, sharpness: float = 10.0):
        self.sharpness = sharpness

    def forward(self, net: Network, x: np.ndarray) -> ForwardResult:
        v = np.asarray(x, dtype=float).copy()
        if v.shape != (net.input_dim,):
            raise SymNetRuntimeError(f"network '{net.name}' expects input of shape "
                                     f"({net.input_dim},), got {v.shape}")
        trace: List[Tuple[str, np.ndarray]] = [("input", v.copy())]
        decision: Optional[bool] = None
        score: Optional[float] = None
        soft: Optional[float] = None
        final: Optional[np.ndarray] = None

        for layer in net.layers:
            if isinstance(layer, LinearLayer):
                if layer.W.shape[1] != v.shape[0]:
                    raise SymNetRuntimeError(f"{layer.name}: W is {layer.W.shape}, input is {v.shape}")
                v = layer.W @ v
                trace.append((layer.name, v.copy()))
            elif isinstance(layer, LiftLayer):
                if len(layer.var_names) != v.shape[0]:
                    raise SymNetRuntimeError(f"{layer.name}: expected {len(layer.var_names)} "
                                             f"components, got {v.shape[0]}")
                env = {n: float(val) for n, val in zip(layer.var_names, v)}
                v = np.append(v, eval_expr(layer.expr, env))
                trace.append((layer.name, v.copy()))
            elif isinstance(layer, PerceptronBankLayer):
                if layer.W.shape[1] != v.shape[0]:
                    raise SymNetRuntimeError(f"{layer.name}: W is {layer.W.shape}, input is {v.shape}")
                if final is None:
                    final = v.copy()
                v = ((layer.W @ v + layer.b) > 0.0).astype(float)
                trace.append((f"{layer.name} [hidden step units]", v.copy()))
            else:  # HyperplaneLayer terminates the network
                if layer.weight.shape[0] != v.shape[0]:
                    raise SymNetRuntimeError(f"{layer.name}: {layer.weight.shape[0]} weights "
                                             f"vs feature dim {v.shape[0]}")
                if final is None:
                    final = v.copy()
                score = float(layer.weight @ v + layer.bias)
                decision = score > 0.0                      # Step(s) = 1 if s > 0 else 0
                soft = float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, self.sharpness * score)))))
                trace.append((f"{layer.name} [w.v+b]", np.array([score])))
                break
        if final is None:
            final = v
        return ForwardResult(np.asarray(x, dtype=float), final, decision, score, soft, trace)


# =============================================================================
# 7. Interpreter (parse -> compile -> run -> report)
# =============================================================================

@dataclass
class Report:
    passed: int = 0
    failed: int = 0

    def add(self, ok: bool) -> None:
        self.passed += int(ok)
        self.failed += int(not ok)


def phasor_str(M: np.ndarray) -> str:
    """Describe a 2x2 matrix as a complex unit e^(i*theta) when it is a rotation."""
    if not ContinuousSO2Group.is_rotation(M):
        return ""
    z = ContinuousSO2Group.phasor_of(M)
    re, im = round(z.real, 9) + 0.0, round(z.imag, 9) + 0.0
    th = ContinuousSO2Group.angle_of(M)
    return f"phasor e^(i*theta) = {re:g}{im:+g}i, theta = {th / math.pi:.4g}*pi rad"


class SymNetInterpreter:
    """Glue code: runs DSL text end to end and prints every intermediate state."""

    def __init__(self, sharpness: float = 10.0):
        self.compiler = SymNetCompiler()
        self.runtime = SymNetRuntime(sharpness)

    def run(self, source: str) -> Report:
        print("--- DSL source " + "-" * 60)
        for n, line in enumerate(source.strip("\n").splitlines(), 1):
            print(f"{n:3d} | {line}")
        tokens = tokenize(source)
        program = Parser(tokens).parse_program()
        n_stmts = sum(len(g.statements) for g in program)
        print(f"\n[lexer/parser] {len(tokens) - 1} tokens -> {len(program)} group(s), "
              f"{n_stmts} statement(s)")
        groups = self.compiler.compile_program(program)

        report = Report()
        for g in groups:
            print(f"\n[compiler] group {g.name}")
            for d in g.declarations:
                print(f"   {d}")
            for task in g.tasks:
                if isinstance(task, VerifyJob):
                    self._run_verify(task, report)
                elif isinstance(task, CheckJob):
                    self._run_check(task, report)
                else:
                    self._run_job(task, report)
        return report

    # ---- single job ----------------------------------------------------------
    def _run_job(self, job: Job, report: Report) -> None:
        res = self.runtime.forward(job.network, job.input_vec)
        print(f"\n[runtime] {job.label}")
        print(f"   network: {job.network.architecture()}")
        for label, vec in res.trace:
            line = f"   {label:<40} {fmt_vec(vec)}"
            if job.matrix_shape and (label == "input" or label.startswith("L")) \
                    and vec.size == int(np.prod(job.matrix_shape)):
                line += f"   = {fmt_mat(vec.reshape(job.matrix_shape))}"
            print(line)
        print(f"   final point in feature space: {fmt_vec(res.final_vector)}")
        if job.matrix_shape and res.decision is None:
            M = res.final_vector.reshape(job.matrix_shape)
            print(f"   resulting group element (matrix): {fmt_mat(M)}")
            ph = phasor_str(M)
            if ph:
                print(f"   {ph}")
        if res.decision is not None:
            print(f"   Step({res.score:+.3f}) = {int(res.decision)}  ->  decision = {res.decision}"
                  f"   (soft sigmoid score {res.soft_score:.4f})")
            if job.expected is not None:
                ok = res.decision == job.expected
                report.add(ok)
                print(f"   expected {job.expected} -> {'PASS' if ok else 'FAIL'}")

    # ---- check ---------------------------------------------------------------
    def _run_check(self, cj: CheckJob, report: Report) -> None:
        a = self.runtime.forward(cj.lhs.network, cj.lhs.input_vec)
        b = self.runtime.forward(cj.rhs.network, cj.rhs.input_vec)
        shape = cj.lhs.matrix_shape
        Ma, Mb = a.final_vector.reshape(shape), b.final_vector.reshape(shape)
        ok = bool(np.allclose(Ma, Mb, atol=1e-9))
        report.add(ok)
        print(f"\n[runtime] {cj.label}")
        for tag, job, res, M in (("lhs", cj.lhs, a, Ma), ("rhs", cj.rhs, b, Mb)):
            print(f"   {tag}: {job.network.architecture()}")
            for label, vec in res.trace[1:]:
                print(f"        {label:<24} {fmt_mat(vec.reshape(shape))}")
            ph = phasor_str(M)
            print(f"        => {fmt_mat(M)}" + (f"   [{ph}]" if ph else ""))
        print(f"   max |lhs - rhs| = {np.abs(Ma - Mb).max():.2e}  ->  {'PASS' if ok else 'FAIL'}")

    # ---- batches -------------------------------------------------------------
    def _run_verify(self, vj: VerifyJob, report: Report) -> None:
        print(f"\n[runtime] {vj.label}")
        print(f"   {'item':<20}{'feature point (before decision)':<36}{'score':>7}  "
              f"{'decision':<9}{'expected':<9}result")
        good = 0
        for job in vj.jobs:
            res = self.runtime.forward(job.network, job.input_vec)
            ok = res.decision == job.expected
            report.add(ok)
            good += int(ok)
            print(f"   {job.short:<20}{fmt_vec(res.final_vector):<36}{res.score:>+7.2f}  "
                  f"{str(res.decision):<9}{str(job.expected):<9}{'PASS' if ok else 'FAIL'}")
        print(f"   => {good}/{len(vj.jobs)} correct" + ("  (100% separation)" if good == len(vj.jobs) else ""))


# =============================================================================
# 8. Demonstrations
# =============================================================================

# ---- Phase 1 regression programs (run with --all) ---------------------------------

KLEIN_XOR_SOURCE = """
# Phase 1: hand-written lift z = |x - y|
group Klein4 {
    lift z = abs(x - y)
    separate XorClassifier { weight = [0, 0, 1], bias = -0.5 }
    classify XorClassifier on (0, 0) expect 0
    classify XorClassifier on (1, 0) expect 1
    classify XorClassifier on (0, 1) expect 1
    classify XorClassifier on (1, 1) expect 0
}
"""

D4_SOURCE = """
# Phase 1: dihedral group D4, word r*s*r, subgroup H = {e, s}
group D4 {
    gen r = rotation(90)
    gen s = matrix([[-1, 0], [0, 1]])
    subgroup D = span(r, s)
    subgroup H = span(s)
    separate InH { weight = [0, 0, 0, 1], bias = -0.5 }
    eval r * s * r
    classify InH of r * s * r expect 1
    verify InH = H over D
}
"""

# ---- Phase 2, Test Case 1: automatic kernel lift synthesis ------------------------

XOR_SPACE_SOURCE = """
# 1a. Raw XOR point set.  NO lift and NO weights are written by hand:
#     the synthesizer discovers the lift and the hyperplane.
group XORSpace {
    space Points {
        (0, 0) class 0
        (1, 0) class 1
        (0, 1) class 1
        (1, 1) class 0
    }
    synthesize XorAuto on Points
    classify XorAuto on (1, 0) expect 1       # ordinary DSL queries use the auto-built pipeline
    classify XorAuto on (1, 1) expect 0
}
"""

KLEIN_GROUP_SOURCE = """
# 1b. Klein 4-group V = {I, a, b, c=ab} as 2x2 matrices, with subgroup H = {I, c}.
#     On vec(M) = (m00, m01, m10, m11) the partition {I,c} | {a,b} is XOR in disguise.
group Klein4Matrices {
    gen a = matrix([[-1, 0], [0, 1]])
    gen b = matrix([[1, 0], [0, -1]])
    gen c = matrix([[-1, 0], [0, -1]])
    subgroup V = span(a, b)
    subgroup H = span(c)

    synthesize InH of H over V
    synthesize InCosetA of H over V coset a      # left coset aH = {a, b}

    classify InH of a * b expect 1
    classify InH of a expect 0
    classify InCosetA of b expect 1
    classify InCosetA of c expect 0
}
"""

D4_CENTER_SOURCE = """
# 1c. Centre Z(D4) = {e, r^2} sits between the other elements in R^4 (the origin lies
#     in the convex hulls of both classes), so no hyperplane works until a lift is added.
group D4Center {
    gen r = rotation(90)
    gen s = matrix([[-1, 0], [0, 1]])
    gen c = rotation(180)
    subgroup D = span(r, s)
    subgroup Z = span(c)
    synthesize InCenter of Z over D
    classify InCenter of r * r expect 1
    classify InCenter of s expect 0
    classify InCenter of r * s expect 0
}
"""

# ---- Phase 2, Test Case 2: continuous SO(2) phasors -------------------------------

SO2_SOURCE = """
# 2. Continuous rotation group SO(2) = U(1), generators R(theta) = e^(i theta).
group ContinuousSO2 {
    gen R(theta) = phasor(theta)                 # radians

    sector Wedge   = [pi/4, 3*pi/4]              # width pi/2 : two half-planes AND-ed
    sector Reflex  = [pi/2, 2*pi]                # width 3pi/2: two half-planes OR-ed

    eval R(pi/4) * R(pi/4)                       # composition of two phasors
    check R(pi/4) * R(pi/4) == R(pi/2)           # must equal [[0,-1],[1,0]] = i
    eval R(pi/4) * R(pi/4) on (1, 0)             # action on the vector (1, 0)

    classify Wedge  of R(pi/4) * R(pi/4) on (1, 0) expect 1    # lands at 90 deg: inside
    classify Wedge  of R(pi/8) on (1, 0) expect 0              # 22.5 deg: outside
    classify Reflex of R(pi/4) on (1, 0) expect 0              # 45 deg: outside

    sweep Wedge  over R from 0 to 2*pi steps 8   # 0, 45, ..., 315 deg (45 and 135 are edges)
    sweep Reflex over R from 0 to 2*pi steps 8
}
"""


def banner(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    interp = SymNetInterpreter(sharpness=10.0)
    total = Report()

    def run(src: str) -> None:
        r = interp.run(src)
        total.passed += r.passed
        total.failed += r.failed

    if "--all" in argv:
        banner("PHASE 1 REGRESSION (hand-written lift, D4 word and subgroup)")
        run(KLEIN_XOR_SOURCE)
        run(D4_SOURCE)

    banner("TEST CASE 1: automatic kernel lift synthesis for finite groups")
    print("No `lift` or hyperplane weights appear in these programs: SymNetKernelSynthesizer\n"
          "tests linear separability, searches basis functions and injects the lift layers.")
    run(XOR_SPACE_SOURCE)
    run(KLEIN_GROUP_SOURCE)
    run(D4_CENTER_SOURCE)

    banner("TEST CASE 2: continuous SO(2) complex phasors and angular sector hyperplanes")
    run(SO2_SOURCE)

    banner(f"SUMMARY: {total.passed} checks passed, {total.failed} failed")
    return 0 if total.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
