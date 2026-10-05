// Embedded default draft-head vocabulary subset (data/draft_vocab/subset_64k.bin via whirl-bin2c).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/model.h"

extern const unsigned char whirl_draft_vocab_64k[];
extern const unsigned long long whirl_draft_vocab_64k_size;

namespace whirl::qwen35 {

std::vector<std::uint32_t> embeddedDraftVocab64k() {
    const std::size_t n = static_cast<std::size_t>(whirl_draft_vocab_64k_size / 4);
    std::vector<std::uint32_t> ids(n);
    const unsigned char* b = whirl_draft_vocab_64k;
    for (std::size_t i = 0; i < n; ++i)
        ids[i] = static_cast<std::uint32_t>(b[4 * i]) | static_cast<std::uint32_t>(b[4 * i + 1]) << 8 |
                 static_cast<std::uint32_t>(b[4 * i + 2]) << 16 | static_cast<std::uint32_t>(b[4 * i + 3]) << 24;
    return ids;
}

}  // namespace whirl::qwen35
