#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>


/*-----------------------
Welford's algorithm
-------------------------*/
struct WelfordData {
    int64_t n;
    float mean;
    float m2;
};

// online update
__device__ inline void welford_update(WelfordData &a, float x) {
    a.n += 1;
    float delta = x - a.mean;
    a.mean += delta / a.n;
    float delta2 = x - a.mean;
    a.m2 += delta * delta2;
}

// combine two welfords
__device__ inline WelfordData welford_combine(const WelfordData &a, const WelfordData &b) {
    if (a.n == 0) return b;
    if (b.n == 0) return a;
    WelfordData out;
    out.n = a.n + b.n;
    float delta = b.mean - a.mean;
    out.mean = a.mean + delta * (static_cast<float>(b.n) / static_cast<float>(out.n));
    out.m2 = a.m2 + b.m2 + delta * delta * (static_cast<float>(a.n) * static_cast<float>(b.n) / static_cast<float>(out.n));
    return out;
}

// tree reduction
__device__ inline WelfordData welford_shfl_down(WelfordData v, int offset) {
    unsigned mask = 0xffffffffu;
    WelfordData other;
    other.n     = __shfl_down_sync(mask, (int)v.n, offset);
    other.mean  = __shfl_down_sync(mask, v.mean, offset);
    other.m2    = __shfl_down_sync(mask, v.m2, offset);
    return welford_combine(v, other);
}