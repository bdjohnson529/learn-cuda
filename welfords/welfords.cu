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

// initialize
__device__ __forceinline__ WelfordData welford_init() {
    return WelfordData{0, 0.0f, 0.0f};
}


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

// test kernel for welfords
template <typename scalar_t>
__global__ void welford_test_kernel(
    const scalar_t* __restrict__ x,
    float* __restrict__ mean_out,
    float* __restrict__ var_out,
    int H
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const int warpSize_ = 32;

    // move pointer to offset
    const scalar_t* row_x = x + (size_t)row * H;

    // online accumulation
    // offest by number of threads in the block
    WelfordData wd = welford_init();
    for (int col = tid; col < H; col += blockDim.x) {
        float v = static_cast<float>(row_x[col]);
        welford_update(wd, v);
    }

    // warp reduce
    // tree reduction pattern
    for (int offset = warpSize_ / 2; offset > 0; offset /= 2) {
        wd = welford_shfl_down(wd, offset);
    }

    // shared memory to combine warp partials
    // only the first thread in each warp stores combined data
    __shared__ WelfordData smem[32];
    int warp_id     = tid / warpSize_;
    int lane        = tid % warpSize_;
    if (lane == 0) smem[warp_id] = wd;

    // final combination by the first warp
    WelfordData total = welford_init();
    if (warp_id == 0) {
        WelfordData acc = welford_init();
        int num_warps = blockDim.x / warpSize_;
        if (lane < num_warps) acc = smem[lane];
        for (int offset = warpSize_ / 2; offset > 0; offset /= 2) {
            acc = welford_shfl_down(acc, offset);
        }
        if (lane == 0) total = acc;
    }

    // load into shared memroy
    __shared__ float s_mean, s_var;
    if (tid == 0) {
        float mean = total.mean;
        float var = (total.n > 0) ? (total.m2 / static_cast<float>(total.n)) : 0.f;
        s_mean = mean;
        s_var = var;
        mean_out[row] = mean;
        var_out[row] = var;
    }
}

/*------- Launcher ----------*/
std::vector<torch::Tensor> welford_test_cuda(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    // tensor dimensions
    int64_t H = x.size(-1);
    int64_t M = x.numel() / H;
    auto x2d = x.view({M,H});

    auto mean = torch::empty({M}, x.options().dtype(torch::kFloat));
    auto var = torch::empty({M}, x.options().dtype(torch::kFloat));

    const int threads = (H >= 512 ? 512 : (H >= 256 ? 256 : 128));
    const dim3 block(threads);
    const dim3 grid(M);

    // at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "welford_test_cuda", [&] {
        welford_test_kernel<scalar_t><<<grid, block, 0, stream>>>(
            x2d.data_ptr<scalar_t>(),
            mean.data_ptr<float>(),
            var.data_ptr<float>(),
            (int)H
        );
    });

    return {mean, var};

}