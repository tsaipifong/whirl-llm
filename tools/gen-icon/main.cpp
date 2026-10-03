// whirl-gen-icon: renders the WHIRL application icon (an abstract three-arm
// vortex on a dark disc) and writes it as a multi-size Windows .ico file.
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-gen-icon OUT.ico
//
// Written for WHIRL (original design). Every size is rendered directly from
// the geometry below with 8x8 supersampling, so the small sizes stay crisp
// instead of being scaled down from the large one. Entries are 32-bit BGRA
// DIBs with an (all-zero) AND mask, which every Windows version and rc.exe
// accept, including the 256 px entry.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr double k_pi = 3.14159265358979323846;

struct Rgb {
    double r, g, b;
};

Rgb mix(Rgb a, Rgb b, double t) {
    t = std::clamp(t, 0.0, 1.0);
    return {a.r + (b.r - a.r) * t, a.g + (b.g - a.g) * t, a.b + (b.b - a.b) * t};
}

double smooth(double e0, double e1, double x) {
    const double t = std::clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// One sample of the design at (x, y) in [-1, 1]^2 (y up). Returns premultiplied
// colour and coverage. `size` adapts stroke weights so 16 px stays legible.
void sample(double x, double y, int size, Rgb& col, double& cov) {
    const double r = std::sqrt(x * x + y * y);
    const double disc_r = 0.96;
    if (r > disc_r) {
        col = {0, 0, 0};
        cov = 0;
        return;
    }
    // background disc: deep indigo, slightly lighter towards the rim
    const Rgb bg_in{0.055, 0.067, 0.165}, bg_out{0.110, 0.137, 0.306};
    Rgb c = mix(bg_in, bg_out, r / disc_r);

    // three logarithmic-spiral arms: phase = angle - twist * ln(r)
    const int arms = 3;
    const double twist = size <= 24 ? 1.55 : size <= 48 ? 1.9 : 2.25;
    const double period = 2.0 * k_pi / arms;
    const double ang = std::atan2(y, x);
    const double rr = std::max(r, 1e-4);
    double ph = std::fmod(ang - twist * std::log(rr / disc_r), period);
    if (ph < 0) ph += period;
    // distance from the arm's centre line, as a fraction of the period
    const double d = std::fabs(ph / period - 0.5) * 2.0;  // 0 = arm centre, 1 = gap centre
    // arms taper towards the centre; small icons get heavier arms
    const double w_out = size <= 24 ? 0.62 : size <= 48 ? 0.52 : 0.42;
    const double w = w_out * smooth(0.0, 0.85, r / disc_r) + 0.10;
    const double arm = 1.0 - smooth(w - 0.04, w + 0.04, d);
    // arm colour: cyan at the rim to violet at the core
    const Rgb cyan{0.184, 0.827, 1.000}, violet{0.553, 0.361, 1.000};
    const Rgb ac = mix(violet, cyan, smooth(0.10, 0.95, r / disc_r));
    // fade the arms into the eye of the vortex
    const double fade = smooth(0.06, 0.30, r);
    c = mix(c, ac, arm * fade);
    // bright eye
    const double eye = 1.0 - smooth(0.07, 0.13, r);
    c = mix(c, Rgb{0.92, 0.97, 1.0}, eye);
    // thin rim highlight
    const double rim = smooth(disc_r - 0.06, disc_r - 0.02, r);
    c = mix(c, Rgb{0.30, 0.55, 0.95}, rim * 0.55);
    col = c;
    cov = 1.0;
}

std::vector<std::uint8_t> render(int size) {
    const int ss = 8;
    std::vector<std::uint8_t> bgra(static_cast<std::size_t>(size) * size * 4);
    for (int py = 0; py < size; ++py) {
        for (int px = 0; px < size; ++px) {
            double ar = 0, ag = 0, ab = 0, aa = 0;
            for (int sy = 0; sy < ss; ++sy) {
                for (int sx = 0; sx < ss; ++sx) {
                    const double fx = (px + (sx + 0.5) / ss) / size * 2.0 - 1.0;
                    const double fy = 1.0 - (py + (sy + 0.5) / ss) / size * 2.0;
                    Rgb c;
                    double cov;
                    sample(fx, fy, size, c, cov);
                    ar += c.r * cov;
                    ag += c.g * cov;
                    ab += c.b * cov;
                    aa += cov;
                }
            }
            const double n = static_cast<double>(ss) * ss;
            const double a = aa / n;
            auto to8 = [](double v) { return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 1.0) * 255.0)); };
            // straight (non-premultiplied) alpha, rows top-down here
            const std::size_t o = (static_cast<std::size_t>(py) * size + px) * 4;
            bgra[o + 0] = a > 0 ? to8(ab / aa) : 0;
            bgra[o + 1] = a > 0 ? to8(ag / aa) : 0;
            bgra[o + 2] = a > 0 ? to8(ar / aa) : 0;
            bgra[o + 3] = to8(a);
        }
    }
    return bgra;
}

void put16(std::vector<std::uint8_t>& v, std::uint32_t x) {
    v.push_back(static_cast<std::uint8_t>(x));
    v.push_back(static_cast<std::uint8_t>(x >> 8));
}

void put32(std::vector<std::uint8_t>& v, std::uint32_t x) {
    put16(v, x & 0xffff);
    put16(v, x >> 16);
}

// BITMAPINFOHEADER + bottom-up BGRA rows + AND mask (1 bpp, rows padded to 4 bytes)
std::vector<std::uint8_t> dibEntry(int size, const std::vector<std::uint8_t>& bgra) {
    std::vector<std::uint8_t> v;
    const std::uint32_t mask_row = static_cast<std::uint32_t>(((size + 31) / 32) * 4);
    const std::uint32_t img_bytes = static_cast<std::uint32_t>(size * size * 4) + mask_row * static_cast<std::uint32_t>(size);
    put32(v, 40);
    put32(v, static_cast<std::uint32_t>(size));
    put32(v, static_cast<std::uint32_t>(size * 2));  // colour + mask
    put16(v, 1);
    put16(v, 32);
    put32(v, 0);  // BI_RGB
    put32(v, img_bytes);
    put32(v, 0);
    put32(v, 0);
    put32(v, 0);
    put32(v, 0);
    for (int y = size - 1; y >= 0; --y)
        v.insert(v.end(), bgra.begin() + static_cast<std::ptrdiff_t>(y) * size * 4, bgra.begin() + static_cast<std::ptrdiff_t>(y + 1) * size * 4);
    v.insert(v.end(), static_cast<std::size_t>(mask_row) * size, 0);
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: whirl-gen-icon OUT.ico\n");
        return 2;
    }
    const int sizes[] = {16, 20, 24, 32, 40, 48, 64, 128, 256};
    std::vector<std::vector<std::uint8_t>> entries;
    for (int s : sizes) entries.push_back(dibEntry(s, render(s)));
    std::vector<std::uint8_t> ico;
    put16(ico, 0);
    put16(ico, 1);  // icon
    put16(ico, static_cast<std::uint32_t>(entries.size()));
    std::uint32_t offset = 6 + 16 * static_cast<std::uint32_t>(entries.size());
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const int s = sizes[i];
        ico.push_back(static_cast<std::uint8_t>(s >= 256 ? 0 : s));
        ico.push_back(static_cast<std::uint8_t>(s >= 256 ? 0 : s));
        ico.push_back(0);  // palette colours
        ico.push_back(0);
        put16(ico, 1);   // planes
        put16(ico, 32);  // bits per pixel
        put32(ico, static_cast<std::uint32_t>(entries[i].size()));
        put32(ico, offset);
        offset += static_cast<std::uint32_t>(entries[i].size());
    }
    for (const auto& e : entries) ico.insert(ico.end(), e.begin(), e.end());
    std::FILE* f = std::fopen(argv[1], "wb");
    if (!f) {
        std::fprintf(stderr, "whirl-gen-icon: cannot write %s\n", argv[1]);
        return 1;
    }
    const bool ok = std::fwrite(ico.data(), 1, ico.size(), f) == ico.size();
    ok ? (void)0 : (void)std::fprintf(stderr, "whirl-gen-icon: short write\n");
    std::fclose(f);
    return ok ? 0 : 1;
}
