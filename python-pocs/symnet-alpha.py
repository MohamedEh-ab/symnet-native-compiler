#!/usr/bin/env python3
"""
SymNet: a tiny DSL that compiles group-theory definitions into an MLP graph.

Pipeline
--------
    DSL text --tokenize--> tokens --Parser--> AST --SymNetCompiler--> Network
    (list of layers) --SymNetRuntime--> (final vector, boolean decision)

Layer types of the compiled network
-----------------------------------
    LinearLayer      v -> W v                  (a group generator, or kron(W, I))
    LiftLayer        v -> [v, f(v)]            (non-linear feature lift, e.g. |x-y|)
    HyperplaneLayer  v -> Step(w . v + b)      (perceptron decision layer)

How group words become layers
-----------------------------
A word  w = g1 * g2 * g3  is the matrix product  M = G1 G2 G3.
Acting on a column vector, the RIGHTMOST factor is applied first, so the
network applies layers in the order G3, G2, G1.

Two evaluation modes exist:
  * point mode    ("eval w on (x, y)"):  the input is a geometric point and
                  every layer is the generator matrix W (N x N).
  * element mode  (default):  the input is vec(I) (row-major flattened identity)
                  and every layer is kron(W, I_N), which is exactly
                  "left-multiply the matrix M by W" written as a linear map on
                  vec(M).  The final vector is vec(w), a feature vector that a
                  hyperplane can classify (subgroup / coset membership).

Only the Python standard library and NumPy are used.
"""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np


# =============================================================================
# 0. Errors and small formatting helpers
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


def fmt_word_items(items: List[Tuple[str, int]]) -> str:
    """Pretty print a parsed word such as [('r',1),('s',1),('r',2)] -> r * s * r^2."""
    return " * ".join(n if e == 1 else f"{n}^{e}" for n, e in items)


def clean_matrix(m: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Zero out floating point dust such as cos(90 deg) = 6e-17."""
    out = np.array(m, dtype=float)
    out[np.abs(out) < tol] = 0.0
    return out + 0.0


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

# ---- expressions (used by lifts, matrix entries, weights, biases) -----------

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


# ---- statements ---------------------------------------------------------------

Word = List[Tuple[str, int]]   # [(generator name, integer exponent), ...]


@dataclass
class GenDecl:
    name: str
    kind: str                                  # matrix | rotation | reflection
    rows: Optional[List[List[Expr]]]
    angle: Optional[Expr]
    line: int


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


Statement = Union[GenDecl, LiftDecl, SubgroupDecl, SeparateDecl,
                  EvalStmt, ClassifyStmt, VerifyStmt]


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
    statement := 'gen' IDENT '=' ( 'matrix' '(' '[' row (',' row)* ']' ')'
                                 | 'rotation' '(' expr ')'          # degrees
                                 | 'reflection' '(' expr ')' )      # axis angle, degrees
               | 'lift' IDENT '=' expr
               | 'subgroup' IDENT '=' 'span' '(' IDENT (',' IDENT)* ')'
               | 'separate' IDENT '{' 'weight' '=' '[' expr,* ']' ','? 'bias' '=' expr '}'
               | 'eval' word ('on' '(' expr,* ')')?
               | 'classify' IDENT ( 'on' '(' expr,* ')' | 'of' word ) ('expect' NUMBER)?
               | 'verify' IDENT '=' IDENT 'over' IDENT
    word      := factor ('*' factor)*         factor := IDENT ('^' '-'? NUMBER)?
    expr      := term (('+'|'-') term)*       term := unary (('*'|'/') unary)*
    unary     := '-' unary | atom             atom := NUMBER | IDENT | IDENT '(' expr ')' | '(' expr ')'
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
            "eval": self.parse_eval, "classify": self.parse_classify,
            "verify": self.parse_verify,
        }
        if t.text not in handlers:
            raise ParseError(f"line {t.line}: unknown statement '{t.text}'")
        return handlers[t.text]()

    def parse_gen(self) -> GenDecl:
        kw = self.expect_ident("gen")
        name = self.expect_ident().text
        self.expect_symbol("=")
        kind = self.expect_ident().text
        self.expect_symbol("(")
        rows: Optional[List[List[Expr]]] = None
        angle: Optional[Expr] = None
        if kind == "matrix":
            self.expect_symbol("[")
            rows = [self.parse_expr_row()]
            while self.accept_symbol(","):
                rows.append(self.parse_expr_row())
            self.expect_symbol("]")
        elif kind in ("rotation", "reflection"):
            angle = self.parse_expr()
        else:
            raise ParseError(f"line {kw.line}: unknown generator kind '{kind}'")
        self.expect_symbol(")")
        return GenDecl(name, kind, rows, angle, kw.line)

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
            exp = 1
            if self.accept_symbol("^"):
                value = self.expect_number()
                if not float(value).is_integer():
                    raise ParseError(f"line {self.peek().line}: exponent must be an integer")
                exp = int(value)
            items.append((name, exp))
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

def default_names(n: int) -> List[str]:
    """Variable names for vector components: x, y, w, t, v4, v5, ..."""
    base = ["x", "y", "w", "t"]
    return [base[i] if i < 4 else f"v{i}" for i in range(n)]


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
class HyperplaneLayer:
    """v -> Step(w . v + b)  (a single perceptron)."""
    name: str
    weight: np.ndarray
    bias: float


Layer = Union[LinearLayer, LiftLayer, HyperplaneLayer]


@dataclass
class Network:
    name: str
    input_dim: int
    layers: List[Layer]

    def architecture(self) -> str:
        """One-line description such as: in(2) -> Lift[z] (3) -> Step(1)."""
        dim = self.input_dim
        parts = [f"input({dim})"]
        for layer in self.layers:
            if isinstance(layer, LinearLayer):
                parts.append(f"Linear[{layer.W.shape[0]}x{layer.W.shape[1]}] {layer.name}")
            elif isinstance(layer, LiftLayer):
                dim += 1
                parts.append(f"Lift[{layer.target}] -> R^{dim}")
            else:
                parts.append(f"Step(w.v+b) {layer.name}")
        return " -> ".join(parts)


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
    weight: np.ndarray
    bias: float


@dataclass
class Job:
    """One runnable query: a network plus the vector fed into it."""
    kind: str                                  # 'eval' | 'classify'
    label: str
    network: Network
    input_vec: np.ndarray
    matrix_shape: Optional[Tuple[int, int]]    # set in element mode for pretty printing
    expected: Optional[bool] = None


@dataclass
class VerifyJob:
    """A batch of classify jobs checked against exact subgroup membership."""
    label: str
    jobs: List[Job]


@dataclass
class CompiledGroup:
    name: str
    generators: Dict[str, np.ndarray] = field(default_factory=dict)
    lift_decls: List[LiftDecl] = field(default_factory=list)
    subgroups: Dict[str, CompiledSubgroup] = field(default_factory=dict)
    classifiers: Dict[str, CompiledClassifier] = field(default_factory=dict)
    declarations: List[str] = field(default_factory=list)
    tasks: List[Union[Job, VerifyJob]] = field(default_factory=list)

    @property
    def dim(self) -> Optional[int]:
        if not self.generators:
            return None
        return next(iter(self.generators.values())).shape[0]


def matrix_key(M: np.ndarray) -> tuple:
    """Hashable fingerprint of a matrix, robust to float dust and -0.0."""
    return tuple((np.round(np.asarray(M, dtype=float), 6) + 0.0).ravel())


class SymNetCompiler:
    """Turns the AST into Networks (generators -> Linear, lifts -> Lift,
    separate blocks -> Hyperplane) and bundles them as runnable jobs."""

    MAX_GROUP_ORDER = 2048

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
            self._compile_lift(ctx, st)
        elif isinstance(st, SubgroupDecl):
            self._compile_subgroup(ctx, st)
        elif isinstance(st, SeparateDecl):
            self._compile_separate(ctx, st)
        elif isinstance(st, EvalStmt):
            ctx.tasks.append(self._compile_eval(ctx, st))
        elif isinstance(st, ClassifyStmt):
            ctx.tasks.append(self._compile_classify(ctx, st))
        elif isinstance(st, VerifyStmt):
            ctx.tasks.append(self._compile_verify(ctx, st))
        else:
            raise CompileError(f"unsupported statement {st!r}")

    # ---- declarations --------------------------------------------------------
    def _compile_gen(self, ctx: CompiledGroup, st: GenDecl) -> None:
        if st.name in ctx.generators:
            raise CompileError(f"generator '{st.name}' already defined")
        if st.kind == "matrix":
            width = {len(r) for r in st.rows}
            if len(width) != 1:
                raise CompileError("matrix rows have different lengths")
            M = np.array([[eval_expr(e, {}) for e in row] for row in st.rows], dtype=float)
            if M.shape[0] != M.shape[1]:
                raise CompileError(f"generator matrix must be square, got {M.shape}")
        else:
            a = math.radians(eval_expr(st.angle, {}))
            c, s = math.cos(a), math.sin(a)
            if st.kind == "rotation":
                M = np.array([[c, -s], [s, c]])
            else:  # reflection across the line through the origin at angle a
                M = np.array([[math.cos(2 * a), math.sin(2 * a)],
                              [math.sin(2 * a), -math.cos(2 * a)]])
        M = clean_matrix(M)
        if ctx.dim is not None and M.shape[0] != ctx.dim:
            raise CompileError(f"generator '{st.name}' is {M.shape[0]}x{M.shape[0]} "
                               f"but the group works in dimension {ctx.dim}")
        ctx.generators[st.name] = M
        ctx.declarations.append(f"gen {st.name} = {st.kind}  ->  Linear weight W = {fmt_mat(M)}")

    def _compile_lift(self, ctx: CompiledGroup, st: LiftDecl) -> None:
        ctx.lift_decls.append(st)
        ctx.declarations.append(f"lift {st.name}  ->  non-linear feature layer appended "
                                f"(R^N -> R^(N+1))")

    def _compile_subgroup(self, ctx: CompiledGroup, st: SubgroupDecl) -> None:
        if st.name in ctx.subgroups:
            raise CompileError(f"subgroup '{st.name}' already defined")
        for g in st.generators:
            if g not in ctx.generators:
                raise CompileError(f"unknown generator '{g}' in span()")
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
        """Expand exponents and look up matrices, preserving left-to-right order."""
        factors: List[Tuple[str, np.ndarray]] = []
        for name, exp in word:
            if name in ctx.generators:
                M = ctx.generators[name]
            elif name == "e":
                if ctx.dim is None:
                    raise CompileError("identity 'e' needs at least one generator to fix the dimension")
                M = np.eye(ctx.dim)
            else:
                raise CompileError(f"unknown generator '{name}'")
            if exp < 0:
                try:
                    M = clean_matrix(np.linalg.inv(M))
                except np.linalg.LinAlgError as exc:
                    raise CompileError(f"generator '{name}' is not invertible") from exc
            for _ in range(abs(exp)):
                factors.append((name if exp > 0 else f"{name}^-1", M))
        return factors

    def _word_layers(self, factors: List[Tuple[str, np.ndarray]], N: int,
                     element_mode: bool) -> List[Layer]:
        """Rightmost factor acts first, so layers are emitted in reversed order."""
        layers: List[Layer] = []
        for step, (name, M) in enumerate(reversed(factors), start=1):
            W = np.kron(M, np.eye(N)) if element_mode else M
            layers.append(LinearLayer(f"L{step}:{name}", W))
        return layers

    def _dim_for(self, ctx: CompiledGroup, factors) -> int:
        N = factors[0][1].shape[0] if factors else ctx.dim
        if N is None:
            raise CompileError("cannot determine dimension: declare a generator first")
        return N

    def _eval_vector(self, exprs: List[Expr]) -> np.ndarray:
        return np.array([eval_expr(e, {}) for e in exprs], dtype=float)

    # ---- eval ----------------------------------------------------------------
    def _compile_eval(self, ctx: CompiledGroup, st: EvalStmt) -> Job:
        factors = self._resolve_word(ctx, st.word)
        N = self._dim_for(ctx, factors)
        label = f"eval {fmt_word_items(st.word)}"
        if st.point is not None:
            point = self._eval_vector(st.point)
            if point.shape[0] != N:
                raise CompileError(f"point has dimension {point.shape[0]}, generators have {N}")
            layers = self._word_layers(factors, N, element_mode=False)
            net = Network(label, N, layers)
            return Job("eval", f"{label} on {fmt_vec(point)}", net, point, None)
        layers = self._word_layers(factors, N, element_mode=True)
        net = Network(label, N * N, layers)
        return Job("eval", label, net, np.eye(N).ravel(), (N, N))

    # ---- classify / separate -------------------------------------------------
    def _get_classifier(self, ctx: CompiledGroup, name: str) -> CompiledClassifier:
        if name not in ctx.classifiers:
            raise CompileError(f"unknown classifier '{name}'")
        return ctx.classifiers[name]

    def _finish_classifier_network(self, ctx: CompiledGroup, clf: CompiledClassifier,
                                   layers: List[Layer], input_dim: int,
                                   label: str) -> Network:
        """Append all lift layers declared so far, then the hyperplane layer."""
        dim = input_dim
        names = default_names(dim)
        for ld in ctx.lift_decls:
            missing = free_variables(ld.expr) - set(names)
            if missing:
                raise CompileError(f"lift '{ld.name}' uses unknown variable(s) {sorted(missing)}; "
                                   f"available: {names}")
            layers.append(LiftLayer(f"lift:{ld.name}", tuple(names), ld.expr, ld.name))
            names = names + [ld.name]
            dim += 1
        if clf.weight.shape[0] != dim:
            raise CompileError(f"classifier '{clf.name}' has {clf.weight.shape[0]} weights "
                               f"but the feature vector has dimension {dim}")
        layers.append(HyperplaneLayer(f"hyperplane:{clf.name}", clf.weight, clf.bias))
        return Network(label, input_dim, layers)

    def _compile_classify(self, ctx: CompiledGroup, st: ClassifyStmt) -> Job:
        clf = self._get_classifier(ctx, st.classifier)
        expected = None if st.expect is None else bool(st.expect)
        if st.point is not None:
            point = self._eval_vector(st.point)
            label = f"classify {clf.name} on {fmt_vec(point)}"
            net = self._finish_classifier_network(ctx, clf, [], point.shape[0], label)
            return Job("classify", label, net, point, None, expected)
        return self._classify_word_job(ctx, clf, st.word, expected)

    def _classify_word_job(self, ctx: CompiledGroup, clf: CompiledClassifier,
                           word: Word, expected: Optional[bool]) -> Job:
        factors = self._resolve_word(ctx, word)
        N = self._dim_for(ctx, factors)
        label = f"classify {clf.name} of {fmt_word_items(word) if word else 'e'}"
        layers = self._word_layers(factors, N, element_mode=True)
        net = self._finish_classifier_network(ctx, clf, layers, N * N, label)
        return Job("classify", label, net, np.eye(N).ravel(), (N, N), expected)

    def _compile_verify(self, ctx: CompiledGroup, st: VerifyStmt) -> VerifyJob:
        clf = self._get_classifier(ctx, st.classifier)
        for nm in (st.subgroup, st.ambient):
            if nm not in ctx.subgroups:
                raise CompileError(f"unknown subgroup '{nm}'")
        sub, amb = ctx.subgroups[st.subgroup], ctx.subgroups[st.ambient]
        jobs = []
        for word, M in amb.elements:
            items: Word = [(g, 1) for g in word]
            jobs.append(self._classify_word_job(ctx, clf, items, sub.contains(M)))
        label = (f"verify {clf.name} == membership in {sub.name} "
                 f"over all {len(amb.elements)} elements of {amb.name}")
        return VerifyJob(label, jobs)


# =============================================================================
# 6. Runtime
# =============================================================================

@dataclass
class ForwardResult:
    input_vector: np.ndarray
    final_vector: np.ndarray                  # point in R^(N+k) fed into the last layer
    decision: Optional[bool]                  # None when the network has no hyperplane
    score: Optional[float]                    # w . v + b
    soft_score: Optional[float]               # sigmoid(sharpness * score): differentiable surrogate
    trace: List[Tuple[str, np.ndarray]]       # (layer name, state after that layer)


class SymNetRuntime:
    """Forward pass: Linear layers -> Lift layers -> Hyperplane step.

    The hard Step gives the boolean decision.  `soft_score` replaces Step with
    a steep sigmoid so the same graph is differentiable for gradient training.
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
        final = v

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
            else:  # HyperplaneLayer
                if layer.weight.shape[0] != v.shape[0]:
                    raise SymNetRuntimeError(f"{layer.name}: {layer.weight.shape[0]} weights "
                                             f"vs feature dim {v.shape[0]}")
                final = v.copy()
                score = float(layer.weight @ v + layer.bias)
                decision = score > 0.0                      # Step(s) = 1 if s > 0 else 0
                soft = float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, self.sharpness * score)))))
                trace.append((f"{layer.name} [w.v+b]", np.array([score])))
                break   # hyperplane layers terminate the network
        if decision is None:
            final = v
        return ForwardResult(np.asarray(x, dtype=float), final, decision, score, soft, trace)


# =============================================================================
# 7. Interpreter (parse -> compile -> run -> report)
# =============================================================================

@dataclass
class Report:
    passed: int = 0
    failed: int = 0


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
                else:
                    self._run_job(task, report)
        return report

    # ---- single job ----------------------------------------------------------
    def _run_job(self, job: Job, report: Report) -> None:
        res = self.runtime.forward(job.network, job.input_vec)
        print(f"\n[runtime] {job.label}")
        print(f"   network: {job.network.architecture()}")
        for label, vec in res.trace:
            line = f"   {label:<34} {fmt_vec(vec)}"
            is_score = "[w.v+b]" in label
            if job.matrix_shape and not is_score and vec.size == np.prod(job.matrix_shape) \
                    and not label.startswith("lift:"):
                line += f"   = {fmt_mat(vec.reshape(job.matrix_shape))}"
            print(line)
        print(f"   final point in feature space: {fmt_vec(res.final_vector)}")
        if job.matrix_shape and res.decision is None:
            print(f"   resulting group element (matrix): {fmt_mat(res.final_vector.reshape(job.matrix_shape))}")
        if res.decision is not None:
            print(f"   Step({res.score:+.3f}) = {int(res.decision)}  ->  decision = {res.decision}"
                  f"   (soft sigmoid score {res.soft_score:.4f})")
            if job.expected is not None:
                ok = res.decision == job.expected
                report.passed += ok
                report.failed += (not ok)
                print(f"   expected {job.expected} -> {'PASS' if ok else 'FAIL'}")

    # ---- verify batch --------------------------------------------------------
    def _run_verify(self, vj: VerifyJob, report: Report) -> None:
        print(f"\n[runtime] {vj.label}")
        print(f"   {'word':<14}{'matrix':<26}{'score':>7}  {'decision':<9}{'exact':<7}result")
        for job in vj.jobs:
            res = self.runtime.forward(job.network, job.input_vec)
            # last linear state = vec(word matrix); fall back to vec(I) for the empty word
            lin_trace = [v for lbl, v in res.trace if lbl != "input" and "[w.v+b]" not in lbl]
            mat_vec = lin_trace[-1] if lin_trace else res.input_vector
            word = job.label.split(" of ", 1)[1]
            ok = res.decision == job.expected
            report.passed += ok
            report.failed += (not ok)
            mat = fmt_mat(mat_vec.reshape(job.matrix_shape))
            print(f"   {word:<14}{mat:<26}{res.score:>+7.2f}  {str(res.decision):<9}"
                  f"{str(job.expected):<7}{'PASS' if ok else 'FAIL'}")


# =============================================================================
# 8. Demonstrations
# =============================================================================

KLEIN_XOR_SOURCE = """
# Test Case 1: XOR / Klein 4-group.  Not linearly separable in R^2, so the
# compiler lifts (x, y) -> (x, y, |x - y|) and one perceptron finishes the job.
group Klein4 {
    lift z = abs(x - y)
    separate XorClassifier { weight = [0, 0, 1], bias = -0.5 }

    classify XorClassifier on (0, 0) expect 0     # Blue
    classify XorClassifier on (1, 0) expect 1     # Red
    classify XorClassifier on (0, 1) expect 1     # Red
    classify XorClassifier on (1, 1) expect 0     # Blue
}
"""

D4_SOURCE = """
# Test Case 2: dihedral group D4 as 2x2 matrices.
group D4 {
    gen r = rotation(90)                       # 90 degree rotation
    gen s = matrix([[-1, 0], [0, 1]])          # horizontal reflection (x -> -x)

    subgroup D = span(r, s)                    # the whole group, 8 elements
    subgroup H = span(s)                       # H = {e, s}

    # Hyperplane on vec(M) = (m00, m01, m10, m11): m11 - 0.5 > 0  <=>  M in H.
    separate InH { weight = [0, 0, 0, 1], bias = -0.5 }

    eval r * s * r                             # compile the word w = r s r
    eval r * s * r on (1, 2)                   # geometric action on a point
    classify InH of r * s * r expect 1         # r s r = s, so it lies in H
    classify InH of r expect 0
    classify InH of s * r expect 0
    verify InH = H over D                      # check all 8 elements of D4
}
"""


def count_linear_solutions_in_2d() -> int:
    """Brute-force grid search for a 2-D perceptron (w1, w2, b) solving XOR."""
    data = [((0, 0), False), ((1, 0), True), ((0, 1), True), ((1, 1), False)]
    grid = np.arange(-2.0, 2.0001, 0.25)
    hits = 0
    for w1 in grid:
        for w2 in grid:
            for b in grid:
                if all((w1 * x + w2 * y + b > 0) == c for (x, y), c in data):
                    hits += 1
    return hits


def main() -> int:
    interp = SymNetInterpreter(sharpness=10.0)
    total = Report()

    print("=" * 78)
    print("TEST CASE 1: XOR / Klein 4-group with non-linear lift z = |x - y|")
    print("=" * 78)
    print(f"Grid search for a single 2-D hyperplane solving XOR "
          f"(3 params in [-2, 2], step 0.25): {count_linear_solutions_in_2d()} solutions found")
    print("-> XOR is not linearly separable in R^2; lifting to R^3 is required.\n")
    r1 = interp.run(KLEIN_XOR_SOURCE)
    print("\nClass colouring: decision False = Blue {(0,0),(1,1)}, True = Red {(1,0),(0,1)}")
    total.passed += r1.passed
    total.failed += r1.failed

    print("\n" + "=" * 78)
    print("TEST CASE 2: Dihedral group D4 - word r*s*r and subgroup membership")
    print("=" * 78)
    r2 = interp.run(D4_SOURCE)
    total.passed += r2.passed
    total.failed += r2.failed

    print("\n" + "=" * 78)
    print(f"SUMMARY: {total.passed} checks passed, {total.failed} failed")
    print("=" * 78)
    return 0 if total.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
