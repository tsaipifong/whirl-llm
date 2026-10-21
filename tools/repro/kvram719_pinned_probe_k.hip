#include <hip/hip_runtime.h>
extern "C" __global__ void readk(const unsigned* p, unsigned long long n_words, unsigned long long stride, unsigned* out) {
    unsigned long long i = (blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x) * stride;
    unsigned s = 0;
    for (; i < n_words; i += (unsigned long long)gridDim.x * blockDim.x * stride) s += p[i];
    atomicAdd(out, s);
}
