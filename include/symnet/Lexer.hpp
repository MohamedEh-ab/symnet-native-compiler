// Lexer.hpp - regex-based tokenizer for the SymNet DSL (same token grammar as symnet.py).
#pragma once

#include "Common.hpp"

#include <regex>
#include <string>
#include <vector>

namespace symnet {

enum class TokKind { Number, Ident, Symbol, Eof };

struct Token {
    TokKind kind;
    std::string text;
    int line;
};

// Convert DSL source text into a token list terminated by an EOF token.
inline std::vector<Token> tokenize(const std::string& source) {
    // groups: 1 COMMENT, 2 SKIP, 3 NUMBER, 4 IDENT, 5 SYMBOL, 6 MISMATCH
    static const std::regex re(
        R"re(([#][^\n]*)|([ \t\r\n]+)|(\d+\.\d*(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?|\d+(?:[eE][+-]?\d+)?)|([A-Za-z_][A-Za-z_0-9]*)|([{}()\[\],=+\-*/^])|(.))re",
        std::regex::ECMAScript | std::regex::optimize);

    std::vector<Token> tokens;
    int line = 1;
    for (std::sregex_iterator it(source.begin(), source.end(), re), end; it != end; ++it) {
        const std::smatch& m = *it;
        const std::string text = m.str();
        if (m[1].matched || m[2].matched) {
            line += static_cast<int>(std::count(text.begin(), text.end(), '\n'));
        } else if (m[3].matched) {
            tokens.push_back({TokKind::Number, text, line});
        } else if (m[4].matched) {
            tokens.push_back({TokKind::Ident, text, line});
        } else if (m[5].matched) {
            tokens.push_back({TokKind::Symbol, text, line});
        } else {
            throw LexError(sfmt("line %d: unexpected character %s", line, pyRepr(text).c_str()));
        }
    }
    tokens.push_back({TokKind::Eof, "", line});
    return tokens;
}

}  // namespace symnet
