// whirl-iso-spec-policy-tests: src/spec/policy.cpp built with only its declared dependencies (whirl::widen / narrow of
// src/common/util.cpp and the platform layer it calls: args, fs, proc) (ARCH-1 8.5 isolation, arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "spec/policy.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
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

void testExeDirectory() {
    const std::string d = exeDirectory();
    ISO_CHECK(!d.empty());
}

void testMtpDefaults() {
    const MtpDefaults a = mtpDefaults(false);
    const MtpDefaults b = mtpDefaults(true);
    ISO_CHECK(a.drafts > 0 && b.drafts > 0);
    ISO_CHECK(a.ngram_min >= 1);
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

}  // namespace

int main(int argc, char** argv) {
    return iso::run("spec.policy",
                    {{"draft_vocab", testDraftVocab}, {"exe_directory", testExeDirectory}, {"mtp_defaults", testMtpDefaults}, {"ngram", testNgram}},
                    argc, argv);
}
