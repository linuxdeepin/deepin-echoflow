// #328 — sherpa diarization hung forever on Windows CUDA.
//
// The subprocess command line was built with POSIX single quotes
// (`--segmentation.pyannote-model='models\seg.onnx'`). cmd.exe does not
// interpret those, so sherpa received literal apostrophes inside its model
// paths, failed to open them, and wedged; the command also appended
// `2>/dev/null`, which is not Windows redirection. Fixed in 98ba5c25 by
// spawning through CreateProcessA with MSVC quoting.
//
// The reason it shipped is the interesting part, and it is what this file is
// for. Note what was NOT the reason: ci.yml's `windows` job builds
// crispasr-cli, which pulls in crispasr_diarize_cli.cpp and therefore this
// header, so the Windows quoter did compile on every push. It compiled green
// while emitting command lines cmd.exe could not parse.
//
// Nothing ever ran it. That job's ctest is filtered to
// `speaker|cluster|centroid|whisper_params`, and there was no test of the
// quoting anywhere at all. Meanwhile the `#ifdef _WIN32` kept the Windows rules
// from even compiling on Linux or macOS, so the developer machines could not
// have caught it either. A compile proves the syntax; only a parse proves the
// argument survives.
//
// Both quoters are now compiled unconditionally, which makes them ordinary pure
// functions. These cases round-trip them against reference *parsers* — an
// implementation of CommandLineToArgvW's documented algorithm, and a POSIX
// single-quote unquoter — rather than against hand-written expected strings.
// Asserting on a literal would only prove the quoter still does what it did;
// round-tripping proves the receiving process sees the argument we meant.
#include "crispasr_subprocess.h"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

using crispasr_cli_process::quote_arg_posix;
using crispasr_cli_process::quote_arg_windows;

namespace {

// Reference implementation of the parsing CommandLineToArgvW performs, per
// Microsoft's "Parsing C++ Command-Line Arguments". This is the consumer our
// Windows quoter must satisfy; writing it out is what makes the round-trip a
// real check rather than a restatement of the quoter.
std::vector<std::string> win_parse(const std::string& cmdline) {
    std::vector<std::string> out;
    size_t i = 0;
    const size_t n = cmdline.size();
    while (i < n) {
        while (i < n && (cmdline[i] == ' ' || cmdline[i] == '\t'))
            i++;
        if (i >= n)
            break;
        std::string cur;
        bool in_quotes = false;
        while (i < n) {
            if (!in_quotes && (cmdline[i] == ' ' || cmdline[i] == '\t'))
                break;
            if (cmdline[i] == '\\') {
                size_t bs = 0;
                while (i < n && cmdline[i] == '\\') {
                    bs++;
                    i++;
                }
                if (i < n && cmdline[i] == '"') {
                    cur.append(bs / 2, '\\');
                    if (bs % 2 == 0) {
                        in_quotes = !in_quotes; // the quote is a delimiter
                    } else {
                        cur.push_back('"'); // escaped, so literal
                    }
                    i++;
                } else {
                    cur.append(bs, '\\'); // not before a quote: all literal
                }
                continue;
            }
            if (cmdline[i] == '"') {
                in_quotes = !in_quotes;
                i++;
                continue;
            }
            cur.push_back(cmdline[i++]);
        }
        out.push_back(cur);
    }
    return out;
}

// Reference unquoter for POSIX single quoting, i.e. what /bin/sh does with
// 'abc'\''def'.
std::string posix_parse(const std::string& quoted) {
    std::string out;
    size_t i = 0;
    while (i < quoted.size()) {
        if (quoted[i] == '\'') {
            i++;
            while (i < quoted.size() && quoted[i] != '\'')
                out.push_back(quoted[i++]);
            i++; // closing quote
        } else if (quoted[i] == '\\' && i + 1 < quoted.size()) {
            out.push_back(quoted[i + 1]);
            i += 2;
        } else {
            out.push_back(quoted[i++]);
        }
    }
    return out;
}

// The inputs that matter: real Windows paths, the flag shape from #328, and
// every backslash/quote arrangement the MSVC rules treat specially.
const std::vector<std::string> kCases = {
    "simple",
    "",
    "has space",
    R"(models\sherpa-segmentation.onnx)",
    R"(C:\Program Files\models\seg.onnx)",
    R"(--segmentation.pyannote-model=models\seg.onnx)",
    R"(--segmentation.pyannote-model=C:\Program Files\seg.onnx)",
    R"(trailing\)",
    R"(trailing\\)",
    R"(with space and trailing\)",
    R"(quote"inside)",
    R"(both "quote" and space)",
    R"(back\slash"quote)",
    R"(\\server\share\model.onnx)",
    R"(\\server\my share\model.onnx)",
    "tab\there",
    "it's got an apostrophe",
    R"(it's got an apostrophe and a \)",
};

} // namespace

TEST_CASE("windows quoting: every argument survives CommandLineToArgvW", "[subprocess][quoting][issue-328]") {
    for (const auto& arg : kCases) {
        const std::string line = quote_arg_windows(arg);
        const std::vector<std::string> parsed = win_parse(line);
        INFO("arg=[" << arg << "] quoted=[" << line << "]");
        REQUIRE(parsed.size() == 1);
        REQUIRE(parsed[0] == arg);
    }
}

TEST_CASE("posix quoting: every argument survives the shell", "[subprocess][quoting][issue-328]") {
    for (const auto& arg : kCases) {
        const std::string q = quote_arg_posix(arg);
        INFO("arg=[" << arg << "] quoted=[" << q << "]");
        REQUIRE(posix_parse(q) == arg);
    }
}

TEST_CASE("windows quoting: a full command line splits back into its arguments", "[subprocess][quoting][issue-328]") {
    // The #328 invocation. Splitting is the property that actually failed:
    // a path that re-splits is a path sherpa cannot open.
    const std::vector<std::string> argv = {
        R"(C:\Program Files\sherpa\sherpa-onnx-offline-speaker-diarization.exe)",
        R"(--segmentation.pyannote-model=models\sherpa-segmentation.onnx)",
        R"(--embedding.model=models\3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx)",
        "--clustering.num-clusters=2",
        R"(C:\Users\Some User\audio.wav)",
    };
    std::string line;
    for (const auto& a : argv) {
        if (!line.empty())
            line.push_back(' ');
        line += quote_arg_windows(a);
    }
    const std::vector<std::string> parsed = win_parse(line);
    INFO("line=[" << line << "]");
    REQUIRE(parsed == argv);
}

TEST_CASE("windows quoting: POSIX quotes would NOT have survived", "[subprocess][quoting][issue-328]") {
    // The bug itself, pinned. cmd.exe has no single-quote handling, so the
    // apostrophes stay in the string and land inside the model path — which is
    // exactly the argument sherpa could not open.
    const std::string arg = R"(--segmentation.pyannote-model=models\seg.onnx)";
    const std::string wrong = quote_arg_posix(arg); // what we used to emit
    const std::vector<std::string> parsed = win_parse(wrong);
    REQUIRE(parsed.size() == 1);
    REQUIRE(parsed[0] != arg);
    REQUIRE(parsed[0].find('\'') != std::string::npos); // literal quotes, as reported

    // And a path with a space would additionally re-split into two arguments.
    const std::string spaced = R"(C:\Program Files\seg.onnx)";
    REQUIRE(win_parse(quote_arg_posix(spaced)).size() == 2);
    REQUIRE(win_parse(quote_arg_windows(spaced)).size() == 1);
}

TEST_CASE("quoting: an empty argument stays one argument", "[subprocess][quoting][issue-328]") {
    // Dropping an empty argument shifts every positional after it.
    REQUIRE(win_parse(quote_arg_windows("") + " next").size() == 2);
    REQUIRE(win_parse(quote_arg_windows("")).size() == 1);
    REQUIRE(posix_parse(quote_arg_posix("")).empty());
}
