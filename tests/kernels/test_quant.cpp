// whirl-kernel-test: quant block decoding (get_rows_*, dequant_f16_*),
// int8 activation quantization (quantize_q8) and the MXFP4 repack.
// SPDX-License-Identifier: Apache-2.0

#include "cpu_ref.h"

namespace kt {

void testQuant(Ctx& c) {
    c.rep.family = "quant";
    const std::vector<HostMat> mats = sampleMats(c, c.quick ? 64 : 512);
    for (const HostMat& m : mats) {
        const std::string sfx = wk::typeSuffix(m.type);
        Buf w(m.data);
        // get_rows: dequantize 8 random rows to f32 (exact: every format decodes exactly in f32)
        {
            const int n = std::min(8, m.nrows);
            std::vector<int> ids = c.randi(static_cast<std::size_t>(n), 0, m.nrows - 1);
            Buf dids(ids), out(static_cast<std::size_t>(n) * m.ncols * 4);
            out.fill(0xff);
            hip::launch(c.fn("get_rows_" + sfx), {8, static_cast<unsigned>(n), 1}, {128, 1, 1}, 0, c.s, w.p(),
                        m.row_bytes, dids.p(), out.p(), m.ncols);
            c.sync();
            std::vector<float> ref(static_cast<std::size_t>(n) * m.ncols);
            for (int t = 0; t < n; ++t)
                ref::dequantRow(m.type, m.data.data() + static_cast<std::size_t>(ids[t]) * m.row_bytes, m.ncols,
                                ref.data() + static_cast<std::size_t>(t) * m.ncols);
            c.rep.add(cmpExact("get_rows_" + sfx + " " + m.name, out.down<float>(ref.size()), ref));
        }
        // dequant_f16: whole (sub)matrix to f16
        if (m.type != QType::f16) {
            const int nr = m.nrows;
            Buf out(static_cast<std::size_t>(nr) * m.ncols * 2);
            out.fill(0xff);
            const std::uint64_t groups = static_cast<std::uint64_t>(m.ncols / 8) * nr;
            hip::launch(c.fn("dequant_f16_" + sfx), {static_cast<unsigned>(std::min<std::uint64_t>(cdiv(groups, 256), 65535)), 1, 1},
                        {256, 1, 1}, 0, c.s, w.p(), m.row_bytes, out.p(), m.ncols, nr);
            c.sync();
            std::vector<std::uint16_t> ref(static_cast<std::size_t>(nr) * m.ncols);
            for (int r = 0; r < nr; ++r)
                ref::dequantRowF16(m.type, m.data.data() + static_cast<std::size_t>(r) * m.row_bytes, m.ncols,
                                   ref.data() + static_cast<std::size_t>(r) * m.ncols);
            c.rep.add(cmpExact("dequant_f16_" + sfx + " " + m.name, out.down<std::uint16_t>(ref.size()), ref));
        }
    }

    // quantize_q8 on several distributions (exact: one f32 division, one reciprocal, rint)
    for (int pass = 0; pass < 3; ++pass) {
        const int n = 17408 * 4;
        std::vector<float> x = pass == 0 ? c.randn(n) : (pass == 1 ? c.randu(n, -1e-3f, 1e-3f) : c.randn(n, 30.f));
        if (pass == 2)
            for (int i = 0; i < 32; ++i) x[static_cast<std::size_t>(64 + i)] = 0.f;  // an all-zero block
        Buf dx(x), dq(n), dd(static_cast<std::size_t>(n / 32) * 4);
        hip::launch(c.k.quantize_q8, {cdiv(n, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dq.p(), dd.p(), n);
        c.sync();
        std::vector<std::int8_t> rq(static_cast<std::size_t>(n));
        std::vector<float> rd(static_cast<std::size_t>(n / 32));
        ref::quantizeQ8(x.data(), n, rq.data(), rd.data());
        c.rep.add(cmpExact("quantize_q8 xq pass " + std::to_string(pass), dq.down<std::int8_t>(rq.size()), rq));
        c.rep.add(cmpExact("quantize_q8 xd pass " + std::to_string(pass), dd.down<float>(rd.size()), rd));
    }

    // MXFP4 repack: decoding the GGUF layout directly == decoding the repacked rows
    if (const auto* f = c.open(c.models.mx)) {
        const auto* t = f->tensor("blk.0.attn_gate.weight");
        if (t != nullptr && t->type == whirl::gguf::GgmlType::mxfp4) {
            const int ncols = static_cast<int>(t->ne[0]);
            const auto raw = f->tensorData(*t);
            const std::uint64_t rb = t->rowBytes();
            std::size_t bad = 0, n = 0;
            for (int r = 0; r < 16; ++r) {
                const std::uint8_t* src = raw.data() + static_cast<std::size_t>(r) * rb;
                std::vector<float> direct(static_cast<std::size_t>(ncols));
                for (int b = 0; b < ncols / 32; ++b) {  // GGUF block: e, qs[16]
                    static const int kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
                    const std::uint8_t* blk = src + static_cast<std::size_t>(b) * 17;
                    const float d = std::ldexp(1.0f, static_cast<int>(blk[0]) - 128);
                    for (int j = 0; j < 16; ++j) {
                        direct[static_cast<std::size_t>(32 * b + j)] = d * kv[blk[1 + j] & 0xF];
                        direct[static_cast<std::size_t>(32 * b + j + 16)] = d * kv[blk[1 + j] >> 4];
                    }
                }
                std::vector<std::uint8_t> rp(rb);
                wk::repackMxfp4Row(src, rp.data(), ncols, nullptr);
                std::vector<float> via(static_cast<std::size_t>(ncols));
                ref::dequantRow(QType::mxfp4, rp.data(), ncols, via.data());
                n += via.size();
                for (std::size_t i = 0; i < via.size(); ++i) bad += via[i] != direct[i];
            }
            Result r;
            r.name = "repackMxfp4Row == GGUF block decode (16 rows)";
            r.kind = Kind::exact;
            r.n = n;
            r.mismatches = bad;
            r.pass = bad == 0;
            c.rep.add(r);
        }
    }
}

}  // namespace kt
