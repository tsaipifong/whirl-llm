// kvram719 probe: does a GPU kernel fault reading pinned host memory once the
// process has pinned more than X MiB (hipHostMalloc, 512 MiB chunks like the tier)?
// usage: probe.exe <first_mb> <total_mb> [chunk_mb=512] [read=each|first|last]
//   first_mb: one pinned buffer allocated first (stands for the 1288 MiB embedding)
//   total_mb: tier-like pinned arena allocated after it, in chunk_mb pieces
// Reads one u32 per 4 KiB of each buffer from a kernel, syncs after each buffer.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("ERR %s -> %d %s\n", #x, (int)e_, hipGetErrorString(e_)); std::fflush(stdout); return 1; } } while (0)


static hipFunction_t g_fn;
static int readBuf(const char* tag, void* buf, size_t bytes, unsigned* d_out) {
    std::printf("%s: memset\n", tag); std::fflush(stdout);
    CK(hipMemset(d_out, 0, 4));
    std::printf("%s: launch\n", tag); std::fflush(stdout);
    struct { const unsigned* p; unsigned long long n; unsigned long long st; unsigned* o; } a{(const unsigned*)buf, (unsigned long long)(bytes / 4), 1024ull, d_out};
    size_t sz = sizeof a;
    void* cfg[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &a, HIP_LAUNCH_PARAM_BUFFER_SIZE, &sz, HIP_LAUNCH_PARAM_END};
    hipError_t le = hipModuleLaunchKernel(g_fn, 256, 1, 1, 256, 1, 1, 0, 0, nullptr, cfg);
    hipError_t se = hipDeviceSynchronize();
    unsigned h = 0;
    hipError_t ce = hipMemcpy(&h, d_out, 4, hipMemcpyDeviceToHost);
    std::printf("%s ptr=%p bytes=%zu launch=%d sync=%d(%s) d2h=%d sum=%u\n", tag, buf, bytes, (int)le, (int)se,
                hipGetErrorString(se), (int)ce, h);
    std::fflush(stdout);
    return (le == hipSuccess && se == hipSuccess && ce == hipSuccess) ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: probe first_mb total_mb [chunk_mb] [each|first|last]\n"); return 2; }
    const unsigned long long first_mb = std::strtoull(argv[1], nullptr, 10);
    const unsigned long long total_mb = std::strtoull(argv[2], nullptr, 10);
    const unsigned long long chunk_mb = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 512;
    const char* mode = argc > 4 ? argv[4] : "each";
    const size_t MiB = 1ull << 20;
    int dev = 0;
    int n = 0;
    CK(hipGetDeviceCount(&n));
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t pr;
        CK(hipGetDeviceProperties(&pr, i));
        std::printf("device %d: %s %s\n", i, pr.name, pr.gcnArchName);
        if (std::strstr(pr.gcnArchName, "gfx1201")) dev = i;
    }
    CK(hipSetDevice(dev));
    hipModule_t mod;
    CK(hipModuleLoad(&mod, "C:\\ClaudeWorkspace\\whirl-work\\scratch\\kvram719\\probe_k.co"));
    CK(hipModuleGetFunction(&g_fn, mod, "readk"));
    std::printf("using device %d\n", dev);
    unsigned* d_out = nullptr;
    CK(hipMalloc(&d_out, 4));
    void* first = nullptr;
    if (first_mb) {
        CK(hipHostMalloc(&first, first_mb * MiB, 0));
        std::memset(first, 1, first_mb * MiB);
        if (readBuf("first(before arena)", first, first_mb * MiB, d_out)) return 1;
    }
    std::vector<void*> ch;
    std::vector<size_t> len;
    unsigned long long left = total_mb;
    while (left) {
        unsigned long long m = left < chunk_mb ? left : chunk_mb;
        void* p = nullptr;
        hipError_t e = hipHostMalloc(&p, m * MiB, 0);
        if (e != hipSuccess) { std::printf("hipHostMalloc chunk %zu (%llu MiB) failed %d %s\n", ch.size(), m, (int)e, hipGetErrorString(e)); return 1; }
        std::memset(p, 2, m * MiB);
        ch.push_back(p);
        len.push_back(m * MiB);
        left -= m;
    }
    std::printf("arena: %zu chunks, %llu MiB (+ first %llu MiB) pinned\n", ch.size(), total_mb, first_mb);
    std::fflush(stdout);
    std::system("C:\\ClaudeWorkspace\\whirl-work\\scratch\\kvram719\\segs.exe");
    std::fflush(stdout);
    int bad = 0;
    if (first) bad |= readBuf("first(after arena)", first, first_mb * MiB, d_out);
    if (bad) return 1;
    char tag[64];
    for (size_t i = 0; i < ch.size(); ++i) {
        if (std::strcmp(mode, "first") == 0 && i != 0) continue;
        if (std::strcmp(mode, "last") == 0 && i + 1 != ch.size()) continue;
        std::snprintf(tag, sizeof tag, "chunk %zu", i);
        if (readBuf(tag, ch[i], len[i], d_out)) return 1;
    }
    std::printf("OK all reads\n");
    return 0;
}
