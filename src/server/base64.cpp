// Base64 decoding of data: URL images (declared in vision_iface.h), moved verbatim from engine_vision.cpp
// (arch025-lane2 leg 5 S6): its only product caller is the HTTP front end (src/server/http.cpp buildVis, module
// protocol), which then links without the engine.
// SPDX-License-Identifier: Apache-2.0

#include "vision_iface.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace whirl::server {

// Base64, standard alphabet; padded input must be a multiple of 4 long, unpadded
// input must not contain '='; unused trailing bits must be zero (as Zig's
// std.base64 standard / standard_no_pad decoders the prototype used).
bool base64Decode(std::string_view in, std::vector<std::uint8_t>& out) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    out.clear();
    std::size_t n = in.size();
    if (n % 4 == 0) {
        // padded: up to two '=' at the end
        std::size_t pad = 0;
        while (pad < 2 && n > 0 && in[n - 1] == '=') {
            --n;
            ++pad;
        }
    }
    if (n % 4 == 1) return false;
    out.reserve(n / 4 * 3 + 2);
    std::uint32_t acc = 0;
    int bits = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int v = val(in[i]);
        if (v < 0) return false;
        acc = (acc << 6) | static_cast<std::uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<std::uint8_t>(acc >> bits));
            acc &= (1u << bits) - 1;
        }
    }
    if (acc != 0) return false;  // non-zero leftover bits
    return true;
}

}  // namespace whirl::server
