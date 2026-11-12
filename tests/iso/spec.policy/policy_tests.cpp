// whirl-iso-spec-policy-tests: src/spec/policy.cpp built with only its declared dependencies (whirl::widen / narrow of
// src/common/util.cpp and the platform layer it calls: args, fs, proc) (ARCH-1 8.5 isolation, arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "spec/policy.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

using namespace whirl;
using namespace whirl::qwen35;

namespace {

void testDraftVocab() {
    ISO_CHECK(!draftVocabFile("off", "D").has_value());
    const std::optional<std::string> f = draftVocabFile("48k", "D");
    ISO_CHECK(f.has_value() && f->size() >= 14 && f->substr(f->size() - 14) == "subset_48k.bin");
    ISO_CHECK(draftVocabChoice(std::nullopt, "").by_default);
    ISO_CHECK(draftVocabDefaultFits(false, draft_vocab_embedded_n_vocab));
    ISO_CHECK(!draftVocabDefaultFits(true, draft_vocab_embedded_n_vocab));
    const std::array<std::uint32_t, 2> ids{1, 3};
    const std::array<std::uint32_t, 1> req{2};
    std::uint32_t added = 0;
    const std::vector<std::uint32_t> v = draftVocabIds(ids, 10, req, &added);
    ISO_CHECK(v.size() == 3 && added == 1);
    ISO_THROWS(readDraftVocab("iso_no_such_vocab_file.bin"), std::exception);
}

// <len> characters: base + "\" + segments of up to 100 characters (the last one longer by one when a separator would end it)
std::wstring dirOfLen(const std::wstring& base, std::size_t len, wchar_t c) {
    std::wstring d = base;
    while (d.size() < len) {
        const std::size_t room = len - d.size() - 1;
        if (room == 0) {
            d += c;
            break;
        }
        d += L"\\" + std::wstring(std::min<std::size_t>(room, 100), c);
    }
    return d;
}

// <folder>\draft_vocab\subset_48k.bin with the ids 1 2, made through the \\?\ form (the folder may be one only that form
// can make); false when it could not be written
bool makeVocabNextTo(const std::wstring& folder) {
    std::error_code ec;
    const std::filesystem::path d(L"\\\\?\\" + folder + L"\\draft_vocab");
    std::filesystem::create_directories(d, ec);
    if (ec) return false;
    std::ofstream f(d / L"subset_48k.bin", std::ios::binary);
    const unsigned char b[8] = {1, 0, 0, 0, 2, 0, 0, 0};
    f.write(reinterpret_cast<const char*>(b), sizeof b);
    return static_cast<bool>(f);
}

// WHIRL_DRAFT_VOCAB=48k as whirl-server resolves it (server_main.cpp: draftVocabChoice(settings, exeDirectory()) ->
// draftVocabSubset -> readDraftVocab) for an exe started as `exe`; "ok" or what went wrong
std::string vocabNextToExe(const std::wstring& exe) {
    const DraftVocabChoice c = draftVocabChoice(std::string("48k"), exeDirectoryOf(exe));
    if (c.kind != DraftVocabChoice::Kind::file) return "no file chosen";
    try {
        const std::vector<std::uint32_t> ids = readDraftVocab(c.file);
        return ids == std::vector<std::uint32_t>{1, 2} ? "ok" : "wrong ids";
    } catch (const std::exception& ex) {
        return std::string("throws: ") + ex.what();
    }
}

void testExeDirectory() {
    const std::string d = exeDirectory();
    ISO_CHECK(!d.empty());
    ISO_CHECK(exeDirectoryOf(L"C:\\whirl\\whirl-server.exe") == "C:\\whirl");
    ISO_CHECK(exeDirectoryOf(L"\\\\srv\\share\\w\\x.exe") == "\\\\srv\\share\\w");
    ISO_CHECK(exeDirectoryOf(L"") == "");
    // arch025-lpath2 (review F1): GetModuleFileNameW returns the path as the program was started. Started through a \\?\
    // path the folder keeps the prefix, at every length: the files next to the exe are opened under it (draftVocabFile
    // joins with backslashes), which works for every folder length and name. Taking the prefix off (cdd3519, for folders
    // under 248 characters) lost the files of folders of 233-247 characters and of a folder whose name ends in a dot
    // ("draft vocabulary file not found"); a rule on the folder alone cannot be safe, because the appended names are not
    // bounded (WHIRL_DRAFT_VOCAB=<N>k).
    ISO_CHECK(exeDirectoryOf(L"\\\\?\\C:\\whirl\\whirl-server.exe") == "\\\\?\\C:\\whirl");
    ISO_CHECK(exeDirectoryOf(L"\\\\?\\UNC\\srv\\share\\w\\x.exe") == "\\\\?\\UNC\\srv\\share\\w");
    const std::wstring seg(120, L'q');
    const std::wstring longdir = L"C:\\" + seg + L"\\" + seg + L"\\" + seg;
    const std::string q120(120, 'q');
    ISO_CHECK(exeDirectoryOf(L"\\\\?\\" + longdir + L"\\x.exe") == "\\\\?\\C:\\" + q120 + "\\" + q120 + "\\" + q120);
    // SC-2 item 1 (progress\arch025-d7.HANDOFF.md:47): a '/' join would break the \\?\ form. The draft-vocabulary path
    // under a \\?\ exe folder keeps the prefix and has backslashes only (also for an exe in a drive root and a trailing-dot
    // folder, names only \\?\ can carry)
    for (const wchar_t* exe : {L"\\\\?\\C:\\whirl\\whirl-server.exe", L"\\\\?\\C:\\whirl-server.exe", L"\\\\?\\UNC\\srv\\share\\w\\x.exe",
                               L"\\\\?\\C:\\tools\\whirl.\\whirl-server.exe"}) {
        const std::optional<std::string> f = draftVocabFile("48k", exeDirectoryOf(exe));
        const std::string tail = "\\draft_vocab\\subset_48k.bin";
        ISO_CHECK(f && f->rfind("\\\\?\\", 0) == 0 && f->find('/') == std::string::npos && f->size() > tail.size() &&
                  f->compare(f->size() - tail.size(), tail.size(), tail) == 0);
    }
    ISO_CHECK(draftVocabFile("48k", exeDirectoryOf(L"\\\\?\\C:\\whirl-server.exe")) == std::string("\\\\?\\C:\\draft_vocab\\subset_48k.bin"));
    ISO_CHECK(draftVocabFile("48k", exeDirectoryOf(L"\\\\?\\C:\\tools\\whirl.\\whirl-server.exe")) ==
              std::string("\\\\?\\C:\\tools\\whirl.\\draft_vocab\\subset_48k.bin"));
}

// arch025-lpath2 (review F1, the review's counter-test A / A3): the file really is read. Folders made under the current
// directory through \\?\; this test program has no manifest, so it is not long-path aware, as whirl-server.exe on a
// default Windows (LongPathsEnabled = 0). 6a9f6be read every case; cdd3519 threw for 233 / 240 / 247 and the trailing dot.
void testExeDirectoryFiles() {
    std::error_code ec;
    const std::wstring base = std::filesystem::current_path(ec).wstring() + L"\\iso_lpath2";
    std::filesystem::remove_all(std::filesystem::path(L"\\\\?\\" + base), ec);
    ISO_CHECK(base.size() <= 190);  // room for the 200-character folder below
    if (base.size() > 190) return;
    for (const std::size_t len : {200u, 232u, 233u, 240u, 247u, 248u, 260u, 300u}) {
        const std::wstring folder = dirOfLen(base + L"\\v", len, L'v');
        ISO_CHECK(folder.size() == len && makeVocabNextTo(folder));
        const std::string r = vocabNextToExe(L"\\\\?\\" + folder + L"\\whirl-server.exe");
        if (r != "ok") std::printf("  exe_directory_files: \\\\?\\ start, folder of %zu characters: %s\n", len, r.c_str());
        ISO_CHECK(r == "ok");
    }
    const std::wstring dot = base + L"\\t\\whirl.";  // a folder name ending in a dot (Win32 parsing would drop the dot)
    ISO_CHECK(makeVocabNextTo(dot));
    const std::string r = vocabNextToExe(L"\\\\?\\" + dot + L"\\whirl-server.exe");
    if (r != "ok") std::printf("  exe_directory_files: \\\\?\\ start, folder name ending in a dot: %s\n", r.c_str());
    ISO_CHECK(r == "ok");
    ISO_CHECK(vocabNextToExe(base + L"\\v\\whirl-server.exe") != "ok");  // control: no vocabulary file in that folder
    std::filesystem::remove_all(std::filesystem::path(L"\\\\?\\" + base), ec);
}

// arch025-n2 (PLAN SC-2b, progress\arch025-lpath2.HANDOFF.md:57 N2): a PLAIN start (no \\?\) from a folder of 233+
// characters: <folder>\draft_vocab\subset_48k.bin is 260+ characters, past MAX_PATH for this test program (no manifest,
// as whirl-server.exe with LongPathsEnabled = 0). ab74257 threw "draft vocabulary file not found" for 233 / 240 / 247 /
// 300; readDraftVocab now opens through platform::fs::longPathForm (Rust std get_long_path at the open). 200 / 232: the
// joined path is under MAX_PATH and was read before as well.
void testExeDirectoryPlainLong() {
    std::error_code ec;
    const std::wstring base = std::filesystem::current_path(ec).wstring() + L"\\iso_n2";
    std::filesystem::remove_all(std::filesystem::path(L"\\\\?\\" + base), ec);
    ISO_CHECK(base.size() <= 190);  // room for the 200-character folder below
    if (base.size() > 190) return;
    for (const std::size_t len : {200u, 232u, 233u, 240u, 247u, 300u}) {
        const std::wstring folder = dirOfLen(base + L"\\p", len, L'p');
        ISO_CHECK(folder.size() == len && makeVocabNextTo(folder));
        const std::string r = vocabNextToExe(folder + L"\\whirl-server.exe");
        if (r != "ok") std::printf("  exe_directory_plain_long: plain start, folder of %zu characters: %s\n", len, r.c_str());
        ISO_CHECK(r == "ok");
    }
    // the error still names the path as given (not its \\?\ form) when the file is missing
    const std::wstring none = dirOfLen(base + L"\\n", 300, L'n');
    const std::string r = vocabNextToExe(none + L"\\whirl-server.exe");
    ISO_CHECK(r.rfind("throws: draft vocabulary file not found: ", 0) == 0 && r.find("\\\\?\\") == std::string::npos);
    std::filesystem::remove_all(std::filesystem::path(L"\\\\?\\" + base), ec);
}

void testMtpDefaults() {
    const MtpDefaults a = mtpDefaults(false);
    const MtpDefaults b = mtpDefaults(true);
    ISO_CHECK(a.drafts > 0 && b.drafts > 0);
    ISO_CHECK(a.ngram_min >= 1);
}

// v0252-e1def (PLAN V1-93): the WHIRL_E1 value of a server that does not set it, per code object and model class
// (device::Profile::e1_dense / e1_moe): the one-cycle lookahead (2) for MoE models on the R9700, 0 everywhere else
void testE1Default() {
    ISO_CHECK(e1Default(true, hip::Arch::gfx1201) == 2u);
    ISO_CHECK(e1Default(false, hip::Arch::gfx1201) == 0u);
    ISO_CHECK(e1Default(true, hip::Arch::gfx1151) == 0u);
    ISO_CHECK(e1Default(false, hip::Arch::gfx1151) == 0u);
    ISO_CHECK(e1Default(true) == e1Default(true, device::k_default_arch) && e1Default(false) == 0u);
    for (const hip::Arch a : {hip::Arch::gfx1201, hip::Arch::gfx1151})
        ISO_CHECK(e1Default(true, a) == device::profile(a).e1_moe && e1Default(false, a) == device::profile(a).e1_dense);
}

void testNgram() {
    Ngram ng;
    ng.reset();
    const std::vector<std::uint32_t> toks{5, 6, 7, 8, 9, 5, 6, 7};
    std::array<std::uint32_t, 8> out{};
    const Ngram::Match m = ng.lookup(toks, 3, out);
    ISO_CHECK(m.n <= out.size());
    if (m.n > 0) ISO_CHECK(out[0] == 8);  // the continuation of the earlier 5 6 7
}

// arch025-N10 (lane 1 leg 6): whirl-server's defaultBatchDrafts and its WHIRL_MTP_BATCH_DRAFTS parse (536d470
// server_main.cpp:231-235, :803-823) -> defaultBatchDrafts / batchDraftsFrom. The references are the old texts.
void testBatchDrafts() {
    using BD = std::array<std::uint32_t, gdn_max_seg + 1>;
    const auto oldDefault = [](bool moe, bool wide) -> BD {
        (void)wide;
        if (moe) return {0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
        return {0, 8, 7, 4, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    };
    const auto oldParse = [](BD batch_drafts, const std::string& s) {
        std::size_t k = 1, p = 0;
        while (p < s.size() && k <= gdn_max_seg) {
            std::size_t e = s.find(',', p);
            if (e == std::string::npos) e = s.size();
            if (e > p) {
                std::uint32_t x = 1;
                try {
                    x = static_cast<std::uint32_t>(std::stoul(s.substr(p, e - p)));
                } catch (const std::exception&) {
                }
                batch_drafts[k++] = std::min(x, qwen35::max_drafts);
            }
            p = e + 1;
        }
        for (; k <= gdn_max_seg; ++k) batch_drafts[k] = batch_drafts[k - 1];
        return batch_drafts;
    };
    std::size_t n = 0, bad = 0;
    for (bool moe : {false, true})
        for (bool wide : {false, true}) {
            ++n;
            if (defaultBatchDrafts(moe) != oldDefault(moe, wide)) ++bad;
        }
    const char* texts[] = {"", ",", "3", "8,7,4", "1,,2", "x,5", "99,2", "-1,4", "4294967296,3", " 5,6", "1,2,3,4,5,6,7,8,9,10,1,2,3,4,5,6,7,8,9",
                           "0", "2,", ",,,7", "3x,2", "+4,1", "10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10"};
    for (bool moe : {false, true})
        for (std::uint32_t drafts : {1u, 3u, 8u})
            for (const char* t : texts) {
                BD base = defaultBatchDrafts(moe);
                base[1] = drafts;
                ++n;
                if (batchDraftsFrom(base, t) != oldParse(base, t)) ++bad;
            }
    std::printf("batch drafts: %zu cases vs the old server_main text, %zu differ\n", n, bad);
    ISO_CHECK(n == 4 + 2 * 3 * (sizeof(texts) / sizeof(texts[0])));
    ISO_CHECK(bad == 0);
}

// arch025-N10 / N6 (lane 1 leg 6): the draft-head vocabulary decision both front ends made (536d470 server_main.cpp:190-227,
// cli/build_model.cpp:15-45) -> draftVocabGate. The references are the two old decision paths, outcome only (messages stay
// with each front end): server = what it logged / whether it went on to the subset; CLI = whether it printed the 64k
// line, returned quietly, or went on.
void testDraftVocabGate() {
    using K = DraftVocabChoice::Kind;
    enum Srv { s_return, s_info_fit, s_warn_fit, s_info_d2, s_warn_d2, s_subset };
    const auto oldServer = [](const DraftVocabChoice& dv, bool moe, std::uint32_t n_vocab, bool d2) {
        if (dv.kind != K::embedded_64k && dv.kind != K::file) return s_return;
        if (dv.kind == K::embedded_64k) {
            const bool fits = dv.by_default ? draftVocabDefaultFits(moe, n_vocab) : n_vocab == draft_vocab_embedded_n_vocab;
            if (!fits) return dv.by_default ? s_info_fit : s_warn_fit;
            if (!d2) return dv.by_default ? s_info_d2 : s_warn_d2;
        }
        return s_subset;
    };
    enum Cli { c_first_n, c_return, c_print64k, c_subset };
    const auto oldCli = [](const DraftVocabChoice& dv, bool moe, std::uint32_t n_vocab, bool d2) {
        if (dv.kind == K::first_n) return c_first_n;
        if (dv.kind == K::full) return c_return;
        if (dv.kind == K::embedded_64k) {
            const bool fits = dv.by_default ? draftVocabDefaultFits(moe, n_vocab) : n_vocab == draft_vocab_embedded_n_vocab;
            if (!fits || (dv.by_default && !d2)) return dv.by_default ? c_return : c_print64k;
        }
        return c_subset;
    };
    // the new decision, read the way each front end now reads it
    const auto newServer = [](const DraftVocabChoice& dv, DraftVocabGate g) {
        switch (g) {
            case DraftVocabGate::none: return s_return;
            case DraftVocabGate::not_fit: return dv.by_default ? s_info_fit : s_warn_fit;
            case DraftVocabGate::no_d2: return dv.by_default ? s_info_d2 : s_warn_d2;
            case DraftVocabGate::subset: break;
        }
        return s_subset;
    };
    const auto newCli = [](const DraftVocabChoice& dv, DraftVocabGate g) {
        if (dv.kind == K::first_n) return c_first_n;
        if (g == DraftVocabGate::none) return c_return;
        if (g == DraftVocabGate::not_fit) return dv.by_default ? c_return : c_print64k;
        if (g == DraftVocabGate::no_d2 && dv.by_default) return c_return;
        return c_subset;
    };
    std::size_t n = 0, bad = 0;
    for (K kind : {K::full, K::first_n, K::embedded_64k, K::file})
        for (bool by_default : {false, true})
            for (bool moe : {false, true})
                for (std::uint32_t n_vocab : {draft_vocab_embedded_n_vocab, 151936u})
                    for (bool d2 : {false, true}) {
                        DraftVocabChoice dv;
                        dv.kind = kind;
                        dv.by_default = by_default;
                        dv.n = 1000;
                        dv.file = "f.bin";
                        const DraftVocabGate g = draftVocabGate(dv, moe, n_vocab, d2);
                        ++n;
                        if (newServer(dv, g) != oldServer(dv, moe, n_vocab, d2)) ++bad;
                        if (newCli(dv, g) != oldCli(dv, moe, n_vocab, d2)) ++bad;
                    }
    std::printf("draft vocab gate: %zu cases x 2 front ends vs the old texts, %zu differ\n", n, bad);
    ISO_CHECK(n == 64);
    ISO_CHECK(bad == 0);
}

// v0252-n2 (DISCOVERY v0252 N1): the MoE row of the default. moe_ok (device::Profile::draft64k_moe) opens the embedded
// 64k subset for a MoE model with the 248320-token vocabulary; without it (the 8060S row, and every caller that does not
// pass it) the decision is the old one; dense models and explicit settings do not read it.
void testDraftVocabMoeRow() {
    using K = DraftVocabChoice::Kind;
    const std::uint32_t nv = draft_vocab_embedded_n_vocab;
    ISO_CHECK(draftVocabDefaultFits(true, nv, true));
    ISO_CHECK(!draftVocabDefaultFits(true, nv, false) && !draftVocabDefaultFits(true, nv));
    ISO_CHECK(!draftVocabDefaultFits(true, 151936, true));
    ISO_CHECK(draftVocabDefaultFits(false, nv, true) && draftVocabDefaultFits(false, nv, false));
    ISO_CHECK(!draftVocabDefaultFits(false, 151936, true));
    const DraftVocabChoice def = draftVocabChoice(std::nullopt, "");
    ISO_CHECK(def.kind == K::embedded_64k && def.by_default);
    // the default on a MoE model: subset with the row, not_fit without it (the 0.2.5.1 decision), no_d2 without the 2-bit head
    ISO_CHECK(draftVocabGate(def, true, nv, true, true) == DraftVocabGate::subset);
    ISO_CHECK(draftVocabGate(def, true, nv, true, false) == DraftVocabGate::not_fit);
    ISO_CHECK(draftVocabGate(def, true, nv, true) == DraftVocabGate::not_fit);
    ISO_CHECK(draftVocabGate(def, true, nv, false, true) == DraftVocabGate::no_d2);
    ISO_CHECK(draftVocabGate(def, true, 151936, true, true) == DraftVocabGate::not_fit);
    // dense: the row changes nothing
    for (const bool ok : {false, true}) {
        ISO_CHECK(draftVocabGate(def, false, nv, true, ok) == DraftVocabGate::subset);
        ISO_CHECK(draftVocabGate(def, false, 151936, true, ok) == DraftVocabGate::not_fit);
    }
    // explicit settings: the row changes nothing (64k needs only the vocabulary; off = none)
    const DraftVocabChoice e64 = draftVocabChoice(std::string("64k"), "");
    const DraftVocabChoice off = draftVocabChoice(std::string("off"), "");
    for (const bool ok : {false, true}) {
        ISO_CHECK(draftVocabGate(e64, true, nv, true, ok) == DraftVocabGate::subset);
        ISO_CHECK(draftVocabGate(off, true, nv, true, ok) == DraftVocabGate::none);
    }
}

// the required ids (specials + byte tokens), the ids and the label (the part both old texts wrote alike)
struct FakeTok {
    std::vector<int> sp{-1, 7, 3, 7};
    std::vector<int> specials() const { return sp; }
    int byteToken(int b) const { return b % 5 == 0 ? -1 : 20 + b; }
};
std::vector<std::uint32_t> fakeEmbedded() { return {1, 2, 5, 9, 40, 100}; }
void testDraftVocabSubset() {
    const FakeTok tok;
    std::vector<std::uint32_t> old_req;
    for (const int t : tok.specials())
        if (t >= 0) old_req.push_back(static_cast<std::uint32_t>(t));
    for (int b = 0; b < 256; ++b)
        if (tok.byteToken(b) >= 0) old_req.push_back(static_cast<std::uint32_t>(tok.byteToken(b)));
    const std::vector<std::uint32_t> req = draftVocabRequired(tok);
    ISO_CHECK(req == old_req);
    for (bool by_default : {false, true}) {
        DraftVocabChoice dv;
        dv.kind = DraftVocabChoice::Kind::embedded_64k;
        dv.by_default = by_default;
        std::uint32_t added = 0;
        const std::vector<std::uint32_t> ids = draftVocabIds(fakeEmbedded(), 400, old_req, &added);
        const std::string label = std::string(dv.by_default ? "64k (embedded, default)" : "64k (embedded)");
        const DraftVocabSubset s = draftVocabSubset(dv, &fakeEmbedded, 400, req);
        ISO_CHECK(s.ids == ids && s.added == added && s.label == label);
    }
    {
        DraftVocabChoice dv;
        dv.kind = DraftVocabChoice::Kind::file;
        dv.file = "iso_no_such_vocab_file.bin";
        ISO_THROWS(draftVocabSubset(dv, &fakeEmbedded, 400, req), std::exception);  // readDraftVocab's throw, as before
    }
}

// arch025-lane1 leg 8: the automatic MTP draft count, which whirl-server and the CLI each wrote (5728296
// src/server/server_main.cpp:744 and src/cli/decode.cpp:458-463)
bool old_srv_auto(const std::optional<std::string>& adapt_env, bool def_automatic, bool drafts_user) {
    return adapt_env ? (*adapt_env == "auto") : (def_automatic && !drafts_user);
}
bool old_cli_auto(const std::optional<std::string>& adapt_env, bool def_automatic, bool dv) {
    bool a = def_automatic && !dv;
    if (adapt_env) a = *adapt_env == "auto";  // (MtpAdapt::set / automatic)
    return a;
}

void testMtpAutoMode() {
    std::size_t n = 0;
    for (const std::optional<std::string>& v : {std::optional<std::string>(), std::optional<std::string>("auto"), std::optional<std::string>("0"),
                                                std::optional<std::string>("2"), std::optional<std::string>(""), std::optional<std::string>("Auto")})
        for (const bool def : {false, true})
            for (const bool given : {false, true}) {
                const bool a = mtpAutoMode(def, given, v.has_value(), v && *v == "auto");
                ISO_CHECK(a == old_srv_auto(v, def, given));
                ISO_CHECK(a == old_cli_auto(v, def, given));
                ++n;
            }
    std::printf("mtp auto mode: %zu cases == the server's and the CLI's 5728296 rule\n", n);
}

// arch025-lane1 leg 9: --draft dflash and the MTP-only settings, verbatim from 729c080 src/server/server_main.cpp
// ServerRun::makeEngine :962-969 (relaxacc) and :991-998 (WHIRL_SLOT_DRAFTS, WHIRL_DRAFT_VOCAB / _WINDOW); the log calls
// recorded as text, numerics::skipItem as its reason
struct OldDflash {
    bool relax_on = false;
    std::string relax_note, skip_why;
    std::uint32_t slot_drafts = 0;
    std::string warn, note;
};
OldDflash old_dflash(bool use_dflash, bool relax_on, std::uint32_t slot_drafts, bool head_env) {
    OldDflash o;
    o.relax_on = relax_on;
    o.slot_drafts = slot_drafts;
    if (use_dflash && o.relax_on) {
        o.relax_note = "relaxacc: not used with --draft dflash (DFlash2 / n-gram drafts keep exact acceptance)";
        o.relax_on = false;
        o.skip_why = "--draft dflash: DFlash2 / n-gram drafts keep exact acceptance";
    }
    if (use_dflash) {
        if (o.slot_drafts != 0) o.warn = "WHIRL_SLOT_DRAFTS ignored with --draft dflash (it allocates MTP drafts)";
        o.slot_drafts = 0;
        if (head_env) o.note = "WHIRL_DRAFT_VOCAB / WHIRL_DRAFT_WINDOW not used with --draft dflash (MTP draft head only)";
    }
    return o;
}
void testDflashRules() {
    std::size_t n = 0;
    for (const bool ud : {false, true})
        for (const bool ro : {false, true})
            for (const std::uint32_t sd : {0u, 1u, 2u, 7u})
                for (const bool he : {false, true}) {
                    const OldDflash o = old_dflash(ud, ro, sd, he);
                    const DflashRelax r = dflashRelax(ud, ro);
                    ISO_CHECK((r.off ? false : ro) == o.relax_on);
                    ISO_CHECK(r.note == o.relax_note && r.skip_why == o.skip_why);
                    std::uint32_t sd_new = sd;
                    std::string warn, note;
                    if (ud) {
                        const DflashMtpOnly m = dflashMtpOnly(sd, he);
                        sd_new = m.slot_drafts;
                        warn = m.warn;
                        note = m.note;
                    }
                    ISO_CHECK(sd_new == o.slot_drafts && warn == o.warn && note == o.note);
                    ++n;
                }
    std::printf("dflash rules: %zu cases == 729c080 makeEngine\n", n);
}

}  // namespace

// ---- leg 10: verbatim c1ca7ce src/server/server_main.cpp:1440 (reportPool: the n-gram drafts the startup line prints)
std::uint32_t old_ngShown(std::uint32_t ngram_max) {
    return ngram_max > 0 ? std::min(ngram_max, whirl::qwen35::max_ng_drafts) : whirl::qwen35::max_ng_drafts;
}
void testNgramShown() {
    std::size_t n = 0;
    for (std::uint32_t x = 0; x <= 2 * whirl::qwen35::max_ng_drafts + 2; ++x) {
        ISO_CHECK(whirl::qwen35::ngramDraftsShown(x) == old_ngShown(x));
        ++n;
    }
    for (const std::uint32_t x : {100u, 65535u, 0x7fffffffu, 0xffffffffu}) {
        ISO_CHECK(whirl::qwen35::ngramDraftsShown(x) == old_ngShown(x));
        ++n;
    }
    std::printf("n-gram drafts shown: %zu cases == c1ca7ce\n", n);
}

// ---- leg 10 part B: verbatim c1ca7ce src/server/server_main.cpp:727 (buildModel: --mtp-drafts over WHIRL_MTP_DRAFTS)
std::optional<std::uint32_t> old_draftsUser(const std::optional<std::uint32_t>& cli, const std::optional<std::uint32_t>& env) {
    return cli ? cli : env;
}
void testMtpDraftsGiven() {
    std::size_t n = 0;
    const std::vector<std::optional<std::uint32_t>> v = {std::nullopt, 0u, 1u, 3u, 8u, 0xffffffffu};
    for (const auto& c : v)
        for (const auto& e : v) {
            ISO_CHECK(whirl::qwen35::mtpDraftsGiven(c, e) == old_draftsUser(c, e));
            ++n;
        }
    std::printf("mtp drafts given: %zu cases == c1ca7ce\n", n);
}

int main(int argc, char** argv) {
    return iso::run("spec.policy",
                    {{"draft_vocab", testDraftVocab}, {"exe_directory", testExeDirectory}, {"exe_directory_files", testExeDirectoryFiles},
                     {"exe_directory_plain_long", testExeDirectoryPlainLong},
                     {"mtp_defaults", testMtpDefaults}, {"e1_default", testE1Default}, {"ngram", testNgram},
                     {"batch_drafts", testBatchDrafts}, {"draft_vocab_gate", testDraftVocabGate}, {"draft_vocab_moe_row", testDraftVocabMoeRow}, {"draft_vocab_subset", testDraftVocabSubset}, {"mtp_auto_mode", testMtpAutoMode},
                     {"dflash_rules", testDflashRules}, {"ngram_shown", testNgramShown}, {"mtp_drafts_given", testMtpDraftsGiven}},
                    argc, argv);
}
