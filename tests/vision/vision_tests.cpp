// WHIRL vision host tests (no GPU, no model files): preprocessing size rules,
// bicubic resize invariants, hashes, placeholder expansion, M-RoPE positions.
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-vision-tests [RGB_DIR]
// With RGB_DIR (optional, external data): for every NAME.rgbref there
// (header "w h tw th\n" then the prototype's preprocessed bytes) and the source
// image NAME.png / NAME.jpg next to it, checks decode + resize bit-exactly.

#include "whirl/common.h"
#include "whirl/vision.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace whirl;

static int g_fail = 0, g_pass = 0;

#define CHECK(cond)                                                       \
    do {                                                                  \
        if (cond) {                                                       \
            ++g_pass;                                                     \
        } else {                                                          \
            ++g_fail;                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
        }                                                                 \
    } while (0)

static std::span<const std::uint8_t> sv(const char* s) { return {reinterpret_cast<const std::uint8_t*>(s), std::strlen(s)}; }

static std::string hex(std::span<const std::uint8_t> b) {
    std::string s;
    char t[3];
    for (std::uint8_t c : b) {
        std::snprintf(t, sizeof(t), "%02x", c);
        s += t;
    }
    return s;
}

static void testSha() {
    const std::span<const std::uint8_t> one[1] = {sv("abc")};
    CHECK(hex(vision::sha256(one)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const std::span<const std::uint8_t> none[1] = {sv("")};
    CHECK(hex(vision::sha256(none)) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    const std::span<const std::uint8_t> two[2] = {sv("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"), sv("")};
    CHECK(hex(vision::sha256(two)) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // split across parts == one part
    const std::span<const std::uint8_t> split[3] = {sv("abcdbcdecdefdefgefghfghighij"), sv("hijkijkljklmklmnlmnomnopnopq"), sv("")};
    CHECK(hex(vision::sha256(split)) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // SHA-NI path (when the CPU has it) == scalar path, odd lengths and split parts
    std::vector<std::uint8_t> big(1 << 20);
    std::uint64_t z = 1;
    for (auto& b : big) {
        z = z * 6364136223846793005ull + 1442695040888963407ull;
        b = static_cast<std::uint8_t>(z >> 56);
    }
    bool same = true;
    for (std::size_t n : {0u, 1u, 55u, 56u, 63u, 64u, 65u, 127u, 1000u, 4096u + 7, 1u << 20}) {
        const std::span<const std::uint8_t> a[2] = {std::span<const std::uint8_t>(big.data(), n / 3), std::span<const std::uint8_t>(big.data() + n / 3, n - n / 3)};
        same = same && vision::sha256Impl(a, true) == vision::sha256Impl(a, false);
    }
    CHECK(same);
}

static void testWyhash() {
    // wyhash final v4 reference vectors (also Zig std.hash.Wyhash's test vectors)
    CHECK(vision::wyhash(0, sv("")) == 0x0409638ee2bde459ull);
    CHECK(vision::wyhash(1, sv("a")) == 0xa8412d091b5fe0a9ull);
    CHECK(vision::wyhash(2, sv("abc")) == 0x32dd92e4b2915153ull);
    CHECK(vision::wyhash(3, sv("message digest")) == 0x8619124089a3a16bull);
    CHECK(vision::wyhash(4, sv("abcdefghijklmnopqrstuvwxyz")) == 0x7a43afb61d7f5f40ull);
    CHECK(vision::wyhash(5, sv("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")) == 0xff42329b90e50d58ull);
    CHECK(vision::wyhash(6, sv("12345678901234567890123456789012345678901234567890123456789012345678901234567890")) == 0xc39cab13b115aad3ull);
}

static void testTargetSize() {
    const vision::SizeOpt o{32, 8 * 1024, 4096 * 1024};
    auto ts = vision::targetSize(640, 360, o);
    CHECK(ts[0] == 640 && ts[1] == 352);
    ts = vision::targetSize(512, 512, o);
    CHECK(ts[0] == 512 && ts[1] == 512);
    ts = vision::targetSize(1920, 1080, o);
    CHECK(ts[0] * ts[1] <= 4096u * 1024u && ts[0] % 32 == 0 && ts[1] % 32 == 0);
    ts = vision::targetSize(10, 10, o);  // min tokens
    CHECK(ts[0] * ts[1] >= 8u * 1024u);
}

static void testResize() {
    // identity size: copy; constant image stays constant under bicubic
    vision::Rgb a;
    a.w = 37;
    a.h = 21;
    a.px.assign(37 * 21 * 3, 0);
    for (std::size_t i = 0; i < a.px.size(); ++i) a.px[i] = static_cast<std::uint8_t>(i % 3 == 0 ? 200 : (i % 3 == 1 ? 17 : 99));
    auto b = vision::resizePadCeil(a, 37, 21);
    CHECK(b.px == a.px);
    auto c = vision::resizePadCeil(a, 64, 32);
    CHECK(c.w == 64 && c.h == 32 && c.px.size() == 64u * 32u * 3u);
    // the resized area (centered) keeps the constant colour, the padding is black
    bool ok = true;
    const std::uint32_t nw = 57, ox = (64 - nw) / 2;  // ceil(37 * 32/21 -> min(64/37, 32/21)) = ceil(37 * 1.5238) = 57
    for (std::uint32_t y = 0; y < 32 && ok; ++y)
        for (std::uint32_t x = 0; x < 64; ++x) {
            const std::uint8_t* p = c.px.data() + (static_cast<std::size_t>(y) * 64 + x) * 3;
            const bool in = x >= ox && x < ox + nw;
            if (in && (p[0] != 200 || p[1] != 17 || p[2] != 99)) ok = false;
            if (!in && (p[0] | p[1] | p[2]) != 0) ok = false;
        }
    CHECK(ok);
}

static void testExpandRope() {
    vision::Prepared p1, p2;
    p1.nx = 4;
    p1.ny = 2;  // 8 tokens
    p2.nx = 3;
    p2.ny = 3;  // 9 tokens
    p1.hash.fill(1);
    p2.hash.fill(2);
    const std::uint32_t pad = 999;
    const std::vector<std::uint32_t> toks = {1, 2, pad, 3, pad, 4};
    const vision::Prepared* imgs[2] = {&p1, &p2};
    qwen35::VisSpan spans[2];
    const auto out = vision::expand(toks, pad, imgs, spans);
    CHECK(out.size() == 6 - 2 + 8 + 9);
    CHECK(spans[0].start == 2 && spans[0].n == 8 && spans[0].delta0 == 0);
    CHECK(spans[1].start == 11 && spans[1].n == 9 && spans[1].delta0 == 8 - 4);
    for (std::uint32_t i = 0; i < 8; ++i) CHECK(out[2 + i] == vision::tokenId(p1.hash, i) && out[2 + i] >= qwen35::image_id_base);
    CHECK(out[10] == 3);
    // same image -> same ids, different image -> different ids
    CHECK(vision::tokenId(p1.hash, 0) != vision::tokenId(p2.hash, 0));
    const qwen35::VisMap vm{std::span<const qwen35::VisSpan>(spans, 2)};
    auto r = vm.rope(1);
    CHECK(r[0] == 1 && r[1] == 1 && r[2] == 1);
    r = vm.rope(2 + 5);  // image 1 token 5: row 1, col 1, t = 2
    CHECK(r[0] == 2 && r[1] == 3 && r[2] == 3);
    r = vm.rope(10);  // text after image 1: 10 - (8 - 4) = 6
    CHECK(r[0] == 6 && r[1] == 6 && r[2] == 6);
    r = vm.rope(11 + 8);  // image 2 token 8: t = 11 - 4 = 7, row 2, col 2
    CHECK(r[0] == 7 && r[1] == 9 && r[2] == 9);
    CHECK(vm.deltaEnd() == 4 + 9 - 3);
    r = vm.rope(20);
    CHECK(r[0] == 20 - 10);
    CHECK(vm.spanAt(3) == &spans[0] && vm.spanAt(10) == nullptr && vm.spanAt(19) == &spans[1]);
    bool threw = false;
    try {
        const std::vector<std::uint32_t> bad = {pad};
        (void)vision::expand(bad, pad, imgs, spans);
    } catch (const vision::VisionError& e) {
        threw = e.code() == "ImagePlaceholderMismatch";
    }
    CHECK(threw);
}

static void testRgbRefs(const std::string& dir) {
    namespace fs = std::filesystem;
    for (const auto& ent : fs::directory_iterator(dir)) {
        if (ent.path().extension() != ".rgbref") continue;
        const std::string ref = readFile(ent.path().string());
        unsigned w = 0, h = 0, tw = 0, th = 0;
        int off = 0;
        if (std::sscanf(ref.c_str(), "%u %u %u %u\n%n", &w, &h, &tw, &th, &off) != 4) continue;
        fs::path img;
        for (const char* ext : {".png", ".jpg", ".jpeg", ".bmp"}) {
            fs::path c = ent.path();
            c.replace_extension(ext);
            if (fs::exists(c)) img = c;
        }
        if (img.empty()) continue;
        const std::string bytes = readFile(img.string());
        const auto src = vision::decodeImage(std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
        const auto ts = vision::targetSize(src.w, src.h, vision::SizeOpt{32, 8 * 1024, 4096 * 1024});
        const auto got = vision::resizePadCeil(src, ts[0], ts[1]);
        const bool same = src.w == w && src.h == h && ts[0] == tw && ts[1] == th && got.px.size() == ref.size() - off &&
                          std::memcmp(got.px.data(), ref.data() + off, got.px.size()) == 0;
        std::printf("  rgb %s: %ux%u -> %ux%u %s\n", img.filename().string().c_str(), src.w, src.h, ts[0], ts[1], same ? "bit-exact" : "DIFFERENT");
        CHECK(same);
    }
}

int main(int argc, char** argv) {
    testSha();
    testWyhash();
    testTargetSize();
    testResize();
    testExpandRope();
    if (argc > 1) testRgbRefs(argv[1]);
    std::printf("vision tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
