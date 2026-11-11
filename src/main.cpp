// main.cpp - symnet_cli: run SymNet DSL programs from the command line.
//
//   symnet_cli                   run the built-in Phase 1 + Phase 2 demonstrations
//   symnet_cli prog.sym [...]    run one or more DSL files
//   --verbose, -v                print the full transcript (lexer, compiler, per-layer trace)
//   --quick,   -q                print only the final summary line
//   --help,    -h
//
// Exit status: 0 if every check passed, 1 if any check failed, 2 on a usage/parse/compile error.
#include <symnet/Compiler.hpp>
#include <symnet/Examples.hpp>
#include <symnet/Interpreter.hpp>

#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

namespace {

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " [--verbose|-v] [--quick|-q] [program.sym ...]\n"
              << "  With no files, runs the built-in demonstration programs.\n";
}

bool readFile(const std::string& path, std::string& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    bool verbose = false, quick = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--verbose") || !std::strcmp(argv[i], "-v")) verbose = true;
        else if (!std::strcmp(argv[i], "--quick") || !std::strcmp(argv[i], "-q")) quick = true;
        else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        else if (argv[i][0] == '-') { usage(argv[0]); return 2; }
        else files.emplace_back(argv[i]);
    }

    std::vector<std::pair<std::string, std::string>> programs;  // (name, source)
    if (files.empty()) {
        for (const auto& ex : symnet::examples::kExamples) programs.emplace_back(ex.name, ex.source);
    } else {
        for (const auto& path : files) {
            std::string text;
            if (!readFile(path, text)) {
                std::cerr << "error: cannot read '" << path << "'\n";
                return 2;
            }
            programs.emplace_back(path, std::move(text));
        }
    }

    symnet::SymNetInterpreter interp(10.0, verbose ? &std::cout : nullptr);
    symnet::Report total;
    try {
        for (const auto& [name, source] : programs) {
            const symnet::Report r = interp.run(source);
            total += r;
            if (!quick && !verbose)
                std::cout << (r.failed == 0 && r.xFailed == 0 ? "[PASS] " : "[FAIL] ") << name << ": " << r.passed
                          << " checks passed, " << r.failed << " failed\n";
        }
    } catch (const symnet::SymNetError& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 2;
    }

    const bool ok = total.failed == 0 && total.xFailed == 0;
    std::cout << "SUMMARY: " << total.passed << " checks passed, " << total.failed << " failed ("
              << total.xChecked << " folded-vs-reference equivalence checks, " << total.xFailed << " mismatches)\n";
    return ok ? 0 : 1;
}
