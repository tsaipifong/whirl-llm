    struct X {  // shared 4-bit WMMA unit activation pieces (ADR-0005); see the end of this file
        int2 a0, a1, c0, c1;
    };
    static __device__ __forceinline__ void load_x(const int8_t *xt, int u, int h, X &x) {
        const int base = (u >> 2) * 256 + 64 * (u & 3) + 8 * h;
        x.a0 = *(const int2 *)(xt + base);
        x.a1 = *(const int2 *)(xt + base + 16);
        x.c0 = *(const int2 *)(xt + base + 32);
        x.c1 = *(const int2 *)(xt + base + 48);
    }

// kernels/shared/gemvw_unit4_x.h -- activation pieces of a 4-bit WMMA unit (IQ4_XS, MXFP4 in kernels/gemv_w.hip):
// sub-block 2p: values 8h..8h+8 (low nibbles) and 16+8h.. (high); same for sub-block 2p+1 -> four 8-byte x pieces.
// #included inside the bodies of WMU<QT_IQ4_XS> and WMU<QT_MXFP4>; ADR-0005 / arch025-sc11d.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
