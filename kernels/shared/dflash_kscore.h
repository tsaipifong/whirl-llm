        const float *kr;  // shared DFlash key score (ADR-0005); see the end of this file
        bool vis;
        if (k < W) {
            const int p = cpos[k];
            vis = p >= 0 && p < qpos && (window == 0 || my_pos - p < window);
            kr = kc + ((size_t)k * nkv + g) * hd;
        } else {
            vis = true;
            kr = kb + ((size_t)(k - W) * nkv + g) * hd;
        }
        float s = -INFINITY;
        if (vis) {
            float d = 0.f;
            for (int i = 0; i < hd; ++i) d += qs[i] * kr[i];
            s = d * scale;
        }

// kernels/shared/dflash_kscore.h -- score s of key k for the DFlash draft attention (kernels/dflash.hip, both
// kernels): keys k < W are context keys (kc, visible if cached position cpos[k] is before qpos and inside the
// sliding window), the others the draft block's own keys (kb, always visible); s = scale * dot(qs, k row) or
// -inf. #included inside both key loops; ADR-0005 / arch025-sc11d. The includer provides k, W, cpos, qpos,
// window, my_pos, kc, kb, nkv, g, hd, qs, scale.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
