// Writes the synthetic tokenizer test corpus (tests/corpus/synthetic/).
// SPDX-License-Identifier: Apache-2.0
//
// Every case is one file of raw bytes. The cases target the pre-tokenizer
// and BPE edge cases: CJK, emoji, digits, whitespace runs and indentation,
// newlines, contractions, special tokens, combining marks, Unicode spaces,
// code in several languages (zh comments + English code), invalid UTF-8.
// Non-ASCII characters are written as code points so the source stays ASCII.

#include "support.h"

#include "whirl/common.h"

#include <cstdio>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace whirl::parity {

namespace {

// UTF-8 of the given code points
std::string U(std::initializer_list<std::uint32_t> cps) {
    std::string s;
    for (std::uint32_t c : cps) appendUtf8(s, c);
    return s;
}

std::string B(std::initializer_list<int> bytes) {
    std::string s;
    for (int b : bytes) s.push_back(static_cast<char>(b));
    return s;
}

std::string rep(const std::string& s, int n) {
    std::string r;
    for (int i = 0; i < n; ++i) r += s;
    return r;
}

std::string join(const std::vector<std::string>& v, const std::string& sep) {
    std::string r;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) r += sep;
        r += v[i];
    }
    return r;
}

std::string replaceFirst(std::string s, const std::string& from, const std::string& to) {
    const std::size_t p = s.find(from);
    if (p != std::string::npos) s.replace(p, from.size(), to);
    return s;
}

const std::string ZH_HELLO = U({0x4F60, 0x597D});
const std::string ZH_WORLD = U({0x4E16, 0x754C});
const std::string ZH_RETURN = U({0x56DE, 0x50B3});
const std::string ZH_COMMA = U({0xFF0C}), ZH_STOP = U({0x3002}), ZH_EXCL = U({0xFF01});
const std::string ZH_SENT = ZH_HELLO + ZH_COMMA + ZH_WORLD + ZH_EXCL;
const std::string ZH_EXPLAIN = U({0x8ACB, 0x7528, 0x7E41, 0x9AD4, 0x4E2D, 0x6587, 0x89E3, 0x91CB, 0x9019, 0x6BB5,
                                  0x7A0B, 0x5F0F, 0x78BC});
const std::string JP = U({0x3053, 0x3093, 0x306B, 0x3061, 0x306F, 0x30AB, 0x30BF, 0x30AB, 0x30CA});
const std::string KO = U({0xC548, 0xB155, 0xD558, 0xC138, 0xC694});
const std::string EMOJI = U({0x1F600, 0x1F44D, 0x1F3FD, 0x1F1F9, 0x1F1FC});
const std::string ZWJ_FAMILY = U({0x1F468, 0x200D, 0x1F469, 0x200D, 0x1F467});
const std::string COMBINING = "e" + U({0x0301}) + "a" + U({0x0300, 0x0323}) + " n" + U({0x0303});
const std::string DEVANAGARI = U({0x0928, 0x092E, 0x0938, 0x094D, 0x0924, 0x0947});
const std::string THAI = U({0x0E2A, 0x0E27, 0x0E31, 0x0E2A, 0x0E14, 0x0E35});
const std::string ARABIC = U({0x0645, 0x0631, 0x062D, 0x0628, 0x0627});
const std::string HEBREW = U({0x05E9, 0x05DC, 0x05D5, 0x05DD});
const std::string CYRILLIC = U({0x041F, 0x0440, 0x0438, 0x0432, 0x0435, 0x0442});
const std::string GREEK = U({0x0391, 0x03B8, 0x03AE, 0x03BD, 0x03B1, 0x03C2});
const std::string GREEK_UPPER = U({0x0391, 0x0398, 0x0389, 0x039D, 0x0391, 0x03A3});  // str.upper()
const std::string GREEK_LOWER = U({0x03B1, 0x03B8, 0x03AE, 0x03BD, 0x03B1, 0x03C2});  // str.lower()
const std::vector<std::string> U_SPACES = {U({0x00A0}), U({0x3000}), U({0x2028}), U({0x2029}),
                                           U({0x0085}), U({0x2003}), U({0x202F}), U({0x1680})};
const std::vector<std::string> ZERO_WIDTH = {U({0x200B}), U({0xFEFF}), U({0x200C}), U({0x200D}), U({0x2060})};
const std::string NEW_IN_16 = U({0x0897, 0x1C89, 0x10D50, 0x16D40, 0x1FAE9, 0x13460});  // assigned in Unicode 16.0
const std::string UNASSIGNED = U({0x0378, 0x0590, 0xE0080, 0x10FFFD});
const std::string PRIVATE = U({0xE000, 0xF8FF, 0xF0000});
const std::string FULLWIDTH = U({0xFF21, 0xFF22, 0xFF11, 0xFF12, 0xFF08, 0xFF09});
const std::string MATH = U({0x2211, 0x222B}) + " x" + U({0x00B2}) + " " + U({0x2264}) + " " + U({0x221E}) + " " +
                         U({0x00BD, 0x2153});
const std::string SUPERSCRIPT = "x" + U({0x00B2, 0x00B3, 0x2074}) + " H" + U({0x2082}) + "O " +
                                U({0x2460, 0x2461, 0x216B});

const std::vector<std::string> SPECIALS = {
    "<|endoftext|>",  "<|im_start|>",   "<|im_end|>",     "<|object_ref_start|>", "<|box_start|>",
    "<|quad_start|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",     "<|image_pad|>",
    "<|video_pad|>",  "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",       "<|fim_pad|>",
    "<|repo_name|>",  "<|file_sep|>",   "<tool_call>",    "</tool_call>",         "<tool_response>",
    "</tool_response>", "<think>",      "</think>"};

const std::string PY_CODE = R"CODE(import os
from typing import List


def fib(n: int) -> List[int]:
    """Return the first n Fibonacci numbers."""
    out = [0, 1]
    while len(out) < n:
        out.append(out[-1] + out[-2])
    return out[:n]


class Cache:
    def __init__(self, size=128):
        self._d = {}
        self.size = size

    def get(self, k, default=None):
        return self._d.get(k, default)


if __name__ == "__main__":
    print(fib(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
)CODE";

const std::string CPP_CODE =
    "#include <vector>\n"
    "#include <cstdint>\n"
    "\n"
    "template <typename T>\n"
    "static T clamp(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }\n"
    "\n"
    "int main(int argc, char** argv) {\n"
    "\tstd::vector<std::uint32_t> xs{1u, 2u, 3u};\n"
    "\tfor (auto& x : xs) x <<= 2;   // shift\n"
    "\tif (argc > 1 && argv[1][0] == '-') { return -1; }\n"
    "\treturn static_cast<int>(xs.size()) - 3;\n"
    "}\n";

const std::string RUST_CODE = R"CODE(fn main() {
    let v: Vec<i64> = (1..=10).map(|x| x * x).collect();
    let s: i64 = v.iter().sum();
    println!("sum = {s}, len = {}", v.len());
    match s {
        0 => unreachable!(),
        n if n > 100 => println!("big"),
        _ => {}
    }
}
)CODE";

const std::string JS_CODE = R"CODE(const fetchJson = async (url, opts = {}) => {
  const res = await fetch(url, { ...opts, headers: { "Content-Type": "application/json" } });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
};
export default fetchJson;
)CODE";

const std::string GO_CODE =
    "package main\n"
    "\n"
    "import \"fmt\"\n"
    "\n"
    "func main() {\n"
    "\tch := make(chan int, 3)\n"
    "\tgo func() { for i := 0; i < 3; i++ { ch <- i }; close(ch) }()\n"
    "\tfor v := range ch {\n"
    "\t\tfmt.Println(v)\n"
    "\t}\n"
    "}\n";

const std::string SQL_CODE = R"CODE(SELECT u.id, u.name, COUNT(o.id) AS n_orders
FROM users AS u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at >= '2024-01-01'
GROUP BY u.id, u.name
HAVING COUNT(o.id) > 3
ORDER BY n_orders DESC;
)CODE";

const std::string ZIG_CODE = R"CODE(const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const a = gpa.allocator();
    const buf = try a.alloc(u8, 16);
    defer a.free(buf);
    @memset(buf, 0);
}
)CODE";

const std::string BASH_CODE = R"CODE(#!/usr/bin/env bash
set -euo pipefail
for f in *.txt; do
  [[ -s "$f" ]] || continue
  wc -l "$f" | awk '{print $1}' >> counts.log 2>&1
done
echo "done: $(date +%s)"
)CODE";

const std::string JSON_TEXT =
    "{\"name\": \"whirl\", \"version\": 1.5, \"tags\": [\"hip\", \"rdna\"], \"nested\": {\"a\": null, \"b\": true, "
    "\"c\": [1e-5, -0.0]}}\n";

const std::string MARKDOWN = "# Title\n\n## " + ZH_EXPLAIN +
                             "\n\n- item 1\n- item **bold** and `code`\n\n| col | " + ZH_HELLO +
                             " |\n|---|---|\n| 1 | 2 |\n\n```python\nprint('hi')\n```\n\n> quote\n";

const std::string ZH_CODE_PROMPT = ZH_EXPLAIN + ZH_COMMA + "def add(a, b):\n    # " + ZH_RETURN +
                                   " a + b\n    return a + b\n" + ZH_STOP + "\n";

using Cases = std::vector<std::pair<std::string, std::string>>;

Cases textCases() {
    Cases c;
    auto add = [&](const char* n, std::string v) { c.emplace_back(n, std::move(v)); };
    add("empty", "");
    add("single_space", " ");
    add("single_newline", "\n");
    add("hello_world", "Hello world");
    add("hello_world_trailing_ws", "Hello world   ");
    add("leading_spaces", "   leading");
    add("zh_sentence", ZH_SENT);
    add("zh_explain_code", ZH_CODE_PROMPT);
    add("zh_en_mixed", "WHIRL " + ZH_HELLO + "R9700" + ZH_WORLD + " gfx1201" + ZH_STOP + "ok");
    add("ja_ko", JP + " " + KO);
    add("emoji", EMOJI + " ok " + ZWJ_FAMILY + EMOJI);
    add("combining_marks", COMBINING + " " + DEVANAGARI + " " + THAI);
    add("rtl", ARABIC + " " + HEBREW + " 123");
    add("cyrillic_greek", CYRILLIC + " " + GREEK + " " + GREEK_UPPER + " " + GREEK_LOWER);
    {
        std::string s;
        for (const auto& u : U_SPACES) s += u;
        add("unicode_spaces", "a" + join(U_SPACES, "b") + "c" + s + "d");
    }
    add("unicode_spaces_runs", "x" + rep(U({0x3000}), 3) + "y" + U({0x00A0}) + " " + U({0x00A0}) + "z " +
                                   U({0x2028}) + "\n" + U({0x0085}) + "w");
    add("zero_width", join(ZERO_WIDTH, "a") + "b" + U({0xFEFF}) + "start");
    add("new_in_unicode16", "x" + NEW_IN_16 + " y " + U({0x0897}) + "a" + U({0x1C89}) + "b");
    add("unassigned_private", UNASSIGNED + " " + PRIVATE + " " + U({0x0378}) + "a");
    add("fullwidth_math", FULLWIDTH + " " + MATH + " " + SUPERSCRIPT);
    add("digits", "123 4567 89012345 3.14159 1,234,567 0x1F 1e-9 -42 +7 007");
    add("digits_cjk", U({0xFF11, 0xFF12, 0xFF13}) + " " + U({0x4E00, 0x4E8C, 0x4E09}) + " 2024" + U({0x5E74}));
    add("contractions", "I'm you're we've they'll he'd it's don't DON'T I'M We'Ve 's 'll' 'x 'S");
    add("contractions_edge", "'''s ''re 'r 'v 'l a'b ' s" + U({0x2019}) + "s it" + U({0x2019}) + "s '" +
                                 U({0x017F}) + " '" + U({0x0130}));
    add("apostrophe_end", "rock 'n' roll '");
    add("punctuation_runs", "a!!! b??? c... d--- e;;; (x) [y] {z} <w> ~@#$%^&*_+=|\\/`");
    add("punct_newline", "end.\n\nnext;\r\n\r\nlast!\n");
    add("space_punct", "x ,y  .z \t!w");
    add("newlines_mixed", "a\nb\r\nc\rd\n\n\ne\r\r\nf\n \n \n");
    add("spaces_before_newline", "line   \n   indented\n\t\n  \t  \nend");
    add("tabs", "\tone\n\t\ttwo\n\t\t\tthree\t\t\n");
    add("indent_return", "def f():\n    return 1\n");
    add("indent_return_only", "    return");
    add("indent_variants", "  return\n   return\n    return\n        return\n\treturn\n \treturn\n\t return\n");
    add("indent_blank_lines", "if x:\n\n    y = 1\n\n\n    return y\n");
    add("trailing_spaces_eof", "code    ");
    add("only_spaces", std::string(17, ' '));
    add("only_newlines", std::string(9, '\n'));
    add("only_tabs", std::string(5, '\t'));
    add("long_word", std::string(3000, 'a'));
    add("long_spaces_mid", "x" + std::string(2500, ' ') + "y");
    add("long_digits", std::string(1200, '9'));
    add("long_zh", rep(ZH_SENT, 300));
    add("long_emoji", rep(EMOJI, 200));
    add("url_email",
        "see https://github.com/ggml-org/llama.cpp/blob/master/README.md#build or mail a.b+c@example.co.uk");
    add("paths_windows", R"(C:\Projects\whirl\build\Release\whirl-tool.exe --file C:\tmp\a b.txt)");
    add("json", JSON_TEXT);
    add("markdown", MARKDOWN);
    add("code_python", PY_CODE);
    add("code_cpp", CPP_CODE);
    add("code_rust", RUST_CODE);
    add("code_js", JS_CODE);
    add("code_go", GO_CODE);
    add("code_sql", SQL_CODE);
    add("code_zig", ZIG_CODE);
    add("code_bash", BASH_CODE);
    add("code_zh_comments",
        replaceFirst(replaceFirst(PY_CODE, "Return the first n Fibonacci numbers.", ZH_EXPLAIN), "# [0, 1",
                     "# " + ZH_RETURN + " [0, 1"));
    // special tokens
    add("special_all", join(SPECIALS, ""));
    add("special_spaced", join(SPECIALS, " ") + " ");
    add("special_chat", "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\n" + ZH_SENT +
                            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nOK<|im_end|>");
    add("special_tool_call", "<tool_call>\n<function=get_weather>\n<parameter=city>\n" + ZH_WORLD +
                                 "\n</parameter>\n</function>\n</tool_call>");
    add("special_partial",
        "<|im_start <|im_end| <think <|endoftext|x |>< <<think>> <|<|im_end|>|> </think</think>");
    add("special_adjacent_text", "abc<think>def</think>ghi<|im_end|>jkl");
    add("special_pad_like", "[PAD248100] [PAD] <|PAD|> <|unused|>");
    add("special_lookalike_fullwidth",
        U({0xFF1C}) + "|im_start|" + U({0xFF1E}) + " <|im" + U({0x200B}) + "_start|>");
    return c;
}

Cases byteCases() {
    Cases b;
    auto add = [&](const char* n, std::string v) { b.emplace_back(n, std::move(v)); };
    add("invalid_stray_continuation", "ab" + B({0x80, 0xBF}) + " cd " + B({0x9F}) + "e");
    add("invalid_truncated_2", "x " + B({0xC3}) + " y " + B({0xC3}));
    add("invalid_truncated_3", "x " + B({0xE4, 0xB8}) + "y" + B({0xE4}));
    add("invalid_truncated_4", "x " + B({0xF0, 0x9F, 0x98}) + " y");
    add("invalid_overlong", "a" + B({0xC0, 0xAF}) + "b" + B({0xE0, 0x80, 0xAF}) + "c" + B({0xC1, 0xBF}));
    add("invalid_surrogates", "a" + B({0xED, 0xA0, 0x80}) + "b" + B({0xED, 0xBF, 0xBF}) + " c");
    add("invalid_f8_ff", "a" + B({0xF8, 0x88, 0x80, 0x80, 0x80}) + "b" + B({0xFE, 0xFF}) + " c");
    add("invalid_lone_lead_end", "end" + B({0xE4}));
    add("invalid_mixed_with_zh", ZH_SENT + B({0xFF}) + ZH_HELLO.substr(0, 4) + " ok");
    add("invalid_in_code", "def f():\n    return '" + B({0xA9}) + "'\n");
    add("invalid_bad_continuation", B({0xE4, 0x41, 0xB8}) + "x" + B({0xC3, 0x28}));
    add("invalid_space_then_bad", "a " + B({0x80}) + "  " + B({0x81}) + "\n" + B({0x82}));
    add("nul_and_controls", "a" + B({0}) + "b" + B({1, 2, 0x1B}) + "[0m c" + B({0x7F}) + "d" + B({0x0B, 0x0C}) + "e");
    add("crlf_only", rep("\r\n", 7));
    add("cr_only", rep("\r", 5) + "x\r");
    add("bom_start", B({0xEF, 0xBB, 0xBF}) + "BOM text\n");
    // F5..F7 lead bytes decode to code points above U+10FFFF: llama.cpp throws
    // (llama-tokenize aborts) and WHIRL reports an error; both count as ERROR
    add("invalid_f5_f7_lead", "ab " + B({0xF5, 0x80, 0x80, 0x80}) + " cd");
    return b;
}

}  // namespace

int cmdMakeCorpus(const std::string& out_dir) {
    namespace fs = std::filesystem;
    makeDirs(out_dir);
    for (const auto& e : fs::directory_iterator(fs::path(widen(out_dir))))
        if (e.is_regular_file() && e.path().extension() == L".txt") fs::remove(e.path());
    int n = 0;
    for (const Cases& cases : {textCases(), byteCases()}) {
        for (const auto& [name, data] : cases) {
            char fn[128];
            std::snprintf(fn, sizeof fn, "%03d_%s.txt", n, name.c_str());
            writeFile(joinPath(out_dir, fn), data);
            ++n;
        }
    }
    std::printf("wrote %d cases to %s\n", n, out_dir.c_str());
    return 0;
}

}  // namespace whirl::parity
