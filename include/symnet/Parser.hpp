// Parser.hpp - recursive-descent parser producing the AST.
//
// Grammar (identical to symnet.py)
//   program   := group*
//   group     := 'group' IDENT '{' statement* '}'
//   statement := gen | lift | subgroup | separate | sector | space | synthesize
//              | eval | classify | verify | check | sweep
//   word      := factor ('*' factor)*      factor := IDENT ('(' expr ')')? ('^' '-'? NUMBER)?
//   expr      := term (('+'|'-') term)*    term := unary (('*'|'/') unary)*
//   unary     := '-' unary | atom          atom := NUMBER | IDENT | IDENT '(' expr ')' | '(' expr ')'
#pragma once

#include "AST.hpp"
#include "Lexer.hpp"

#include <string>
#include <vector>

namespace symnet {

class Parser {
public:
    explicit Parser(std::vector<Token> tokens) : toks_(std::move(tokens)) {}

    std::vector<GroupDecl> parseProgram() {
        std::vector<GroupDecl> groups;
        while (peek().kind != TokKind::Eof) groups.push_back(parseGroup());
        if (groups.empty()) throw ParseError("empty program: expected at least one 'group' block");
        return groups;
    }

private:
    std::vector<Token> toks_;
    std::size_t i_ = 0;

    // ---- token helpers ----------------------------------------------------------------
    const Token& peek() const { return toks_[i_]; }
    Token advance() {
        Token t = toks_[i_];
        if (t.kind != TokKind::Eof) ++i_;
        return t;
    }
    static std::string shown(const Token& t) { return pyRepr(t.text.empty() ? "end of input" : t.text); }

    bool atSymbol(const char* s) const { return peek().kind == TokKind::Symbol && peek().text == s; }
    bool acceptSymbol(const char* s) {
        if (atSymbol(s)) { advance(); return true; }
        return false;
    }
    Token expectSymbol(const char* s) {
        if (!atSymbol(s))
            throw ParseError(sfmt("line %d: expected '%s' but found %s", peek().line, s, shown(peek()).c_str()));
        return advance();
    }
    bool atIdent(const char* word = nullptr) const {
        return peek().kind == TokKind::Ident && (word == nullptr || peek().text == word);
    }
    Token expectIdent(const char* word = nullptr) {
        if (!atIdent(word)) {
            const std::string wanted = word ? "'" + std::string(word) + "'" : "an identifier";
            throw ParseError(sfmt("line %d: expected %s but found %s", peek().line, wanted.c_str(), shown(peek()).c_str()));
        }
        return advance();
    }
    double expectNumber() {
        const bool neg = acceptSymbol("-");
        const Token t = peek();
        if (t.kind != TokKind::Number)
            throw ParseError(sfmt("line %d: expected a number but found %s", t.line, shown(t).c_str()));
        advance();
        const double v = std::stod(t.text);
        return neg ? -v : v;
    }

    // ---- program / group --------------------------------------------------------------
    GroupDecl parseGroup() {
        const Token kw = expectIdent("group");
        const std::string name = expectIdent().text;
        expectSymbol("{");
        std::vector<Statement> stmts;
        while (!atSymbol("}")) {
            if (peek().kind == TokKind::Eof)
                throw ParseError(sfmt("line %d: unterminated group '%s'", peek().line, name.c_str()));
            stmts.push_back(parseStatement());
        }
        expectSymbol("}");
        return GroupDecl{name, std::move(stmts), kw.line};
    }

    Statement parseStatement() {
        const Token t = peek();
        if (t.kind != TokKind::Ident)
            throw ParseError(sfmt("line %d: expected a statement keyword but found %s", t.line, pyRepr(t.text).c_str()));
        const std::string& k = t.text;
        if (k == "gen") return parseGen();
        if (k == "lift") return parseLift();
        if (k == "subgroup") return parseSubgroup();
        if (k == "separate") return parseSeparate();
        if (k == "sector") return parseSector();
        if (k == "space") return parseSpace();
        if (k == "synthesize") return parseSynthesize();
        if (k == "eval") return parseEval();
        if (k == "classify") return parseClassify();
        if (k == "verify") return parseVerify();
        if (k == "check") return parseCheck();
        if (k == "sweep") return parseSweep();
        throw ParseError(sfmt("line %d: unknown statement '%s'", t.line, k.c_str()));
    }

    GenDecl parseGen() {
        const Token kw = expectIdent("gen");
        GenDecl g;
        g.line = kw.line;
        g.name = expectIdent().text;
        if (acceptSymbol("(")) {
            g.param = expectIdent().text;
            expectSymbol(")");
        }
        expectSymbol("=");
        g.kind = expectIdent().text;
        if (g.param && g.kind == "matrix")
            throw ParseError(sfmt("line %d: a parametrised generator needs an angle kind (phasor/rotation/reflection), not matrix", kw.line));
        expectSymbol("(");
        if (g.kind == "matrix") {
            expectSymbol("[");
            g.rows.push_back(parseExprRow());
            while (acceptSymbol(",")) g.rows.push_back(parseExprRow());
            expectSymbol("]");
        } else if (g.kind == "rotation" || g.kind == "reflection" || g.kind == "phasor") {
            g.angle = parseExpr();
        } else {
            throw ParseError(sfmt("line %d: unknown generator kind '%s'", kw.line, g.kind.c_str()));
        }
        expectSymbol(")");
        return g;
    }

    std::vector<ExprPtr> parseExprRow() {
        expectSymbol("[");
        std::vector<ExprPtr> items{parseExpr()};
        while (acceptSymbol(",")) items.push_back(parseExpr());
        expectSymbol("]");
        return items;
    }

    LiftDecl parseLift() {
        const Token kw = expectIdent("lift");
        std::string name = expectIdent().text;
        expectSymbol("=");
        return LiftDecl{std::move(name), parseExpr(), kw.line};
    }

    SubgroupDecl parseSubgroup() {
        const Token kw = expectIdent("subgroup");
        std::string name = expectIdent().text;
        expectSymbol("=");
        expectIdent("span");
        expectSymbol("(");
        std::vector<std::string> gens{expectIdent().text};
        while (acceptSymbol(",")) gens.push_back(expectIdent().text);
        expectSymbol(")");
        return SubgroupDecl{std::move(name), std::move(gens), kw.line};
    }

    SeparateDecl parseSeparate() {
        const Token kw = expectIdent("separate");
        const std::string name = expectIdent().text;
        expectSymbol("{");
        std::vector<ExprPtr> weight;
        bool haveWeight = false;
        ExprPtr bias;
        while (!atSymbol("}")) {
            const std::string key = expectIdent().text;
            expectSymbol("=");
            if (key == "weight") {
                expectSymbol("[");
                weight.clear();
                weight.push_back(parseExpr());
                while (acceptSymbol(",")) weight.push_back(parseExpr());
                expectSymbol("]");
                haveWeight = true;
            } else if (key == "bias") {
                bias = parseExpr();
            } else {
                throw ParseError(sfmt("line %d: unknown field '%s' in separate block", kw.line, key.c_str()));
            }
            acceptSymbol(",");
        }
        expectSymbol("}");
        if (!haveWeight || !bias)
            throw ParseError(sfmt("line %d: separate '%s' needs both weight and bias", kw.line, name.c_str()));
        return SeparateDecl{name, std::move(weight), std::move(bias), kw.line};
    }

    SectorDecl parseSector() {
        const Token kw = expectIdent("sector");
        std::string name = expectIdent().text;
        expectSymbol("=");
        expectSymbol("[");
        ExprPtr lo = parseExpr();
        expectSymbol(",");
        ExprPtr hi = parseExpr();
        expectSymbol("]");
        return SectorDecl{std::move(name), std::move(lo), std::move(hi), kw.line};
    }

    SpaceDecl parseSpace() {
        const Token kw = expectIdent("space");
        SpaceDecl sp;
        sp.line = kw.line;
        sp.name = expectIdent().text;
        expectSymbol("{");
        while (!atSymbol("}")) {
            sp.points.push_back(parsePoint());
            expectIdent("class");
            sp.labels.push_back(static_cast<int>(expectNumber()));
            acceptSymbol(",");
        }
        expectSymbol("}");
        if (sp.points.empty()) throw ParseError(sfmt("line %d: space '%s' has no points", kw.line, sp.name.c_str()));
        return sp;
    }

    SynthesizeStmt parseSynthesize() {
        const Token kw = expectIdent("synthesize");
        SynthesizeStmt s;
        s.line = kw.line;
        s.name = expectIdent().text;
        if (atIdent("on")) {
            advance();
            s.space = expectIdent().text;
            return s;
        }
        expectIdent("of");
        s.subgroup = expectIdent().text;
        expectIdent("over");
        s.ambient = expectIdent().text;
        if (atIdent("coset")) {
            advance();
            s.coset = parseWord();
        }
        return s;
    }

    std::vector<ExprPtr> parsePoint() {
        expectSymbol("(");
        std::vector<ExprPtr> pts{parseExpr()};
        while (acceptSymbol(",")) pts.push_back(parseExpr());
        expectSymbol(")");
        return pts;
    }

    Word parseWord() {
        Word items;
        while (true) {
            Factor f;
            f.name = expectIdent().text;
            if (atSymbol("(")) {
                advance();
                f.arg = parseExpr();
                expectSymbol(")");
            }
            if (acceptSymbol("^")) {
                const double value = expectNumber();
                if (value != std::floor(value))
                    throw ParseError(sfmt("line %d: exponent must be an integer", peek().line));
                f.exp = static_cast<int>(value);
            }
            items.push_back(std::move(f));
            if (!acceptSymbol("*")) return items;
        }
    }

    EvalStmt parseEval() {
        const Token kw = expectIdent("eval");
        EvalStmt s;
        s.line = kw.line;
        s.word = parseWord();
        if (atIdent("on")) {
            advance();
            s.point = parsePoint();
        }
        return s;
    }

    ClassifyStmt parseClassify() {
        const Token kw = expectIdent("classify");
        ClassifyStmt s;
        s.line = kw.line;
        s.classifier = expectIdent().text;
        if (atIdent("on")) {
            advance();
            s.point = parsePoint();
        } else if (atIdent("of")) {
            advance();
            s.word = parseWord();
            if (atIdent("on")) {
                advance();
                s.point = parsePoint();
            }
        } else {
            throw ParseError(sfmt("line %d: expected 'on (...)' or 'of <word>' after classifier name", kw.line));
        }
        if (atIdent("expect")) {
            advance();
            s.expect = static_cast<int>(expectNumber());
        }
        return s;
    }

    VerifyStmt parseVerify() {
        const Token kw = expectIdent("verify");
        VerifyStmt s;
        s.line = kw.line;
        s.classifier = expectIdent().text;
        expectSymbol("=");
        s.subgroup = expectIdent().text;
        expectIdent("over");
        s.ambient = expectIdent().text;
        return s;
    }

    CheckStmt parseCheck() {
        const Token kw = expectIdent("check");
        CheckStmt s;
        s.line = kw.line;
        s.lhs = parseWord();
        expectSymbol("=");
        expectSymbol("=");
        s.rhs = parseWord();
        return s;
    }

    SweepStmt parseSweep() {
        const Token kw = expectIdent("sweep");
        SweepStmt s;
        s.line = kw.line;
        s.classifier = expectIdent().text;
        expectIdent("over");
        s.gen = expectIdent().text;
        expectIdent("from");
        s.start = parseExpr();
        expectIdent("to");
        s.stop = parseExpr();
        expectIdent("steps");
        s.steps = static_cast<int>(expectNumber());
        if (s.steps < 1) throw ParseError(sfmt("line %d: steps must be >= 1", kw.line));
        return s;
    }

    // ---- expressions ------------------------------------------------------------------
    ExprPtr parseExpr() {
        ExprPtr node = parseTerm();
        while (atSymbol("+") || atSymbol("-")) {
            const char op = advance().text[0];
            node = mkBin(op, node, parseTerm());
        }
        return node;
    }
    ExprPtr parseTerm() {
        ExprPtr node = parseUnary();
        while (atSymbol("*") || atSymbol("/")) {
            const char op = advance().text[0];
            node = mkBin(op, node, parseUnary());
        }
        return node;
    }
    ExprPtr parseUnary() {
        if (acceptSymbol("-")) return mkNeg(parseUnary());
        return parseAtom();
    }
    ExprPtr parseAtom() {
        const Token t = peek();
        if (t.kind == TokKind::Number) {
            advance();
            return mkNum(std::stod(t.text));
        }
        if (t.kind == TokKind::Ident) {
            advance();
            if (atSymbol("(")) {
                if (!isKnownFunc(t.text)) throw ParseError(sfmt("line %d: unknown function '%s'", t.line, t.text.c_str()));
                advance();
                ExprPtr arg = parseExpr();
                expectSymbol(")");
                return mkCall(t.text, arg);
            }
            return mkVar(t.text);
        }
        if (acceptSymbol("(")) {
            ExprPtr node = parseExpr();
            expectSymbol(")");
            return node;
        }
        throw ParseError(sfmt("line %d: unexpected %s in expression", t.line, shown(t).c_str()));
    }
};

}  // namespace symnet
