import torch
from torch.utils.cpp_extension import load

# JIT compile
ext = load(name="welford_test_ext",
            sources=["binding.cpp", "welfords.cu"],
            extra_cflags=["03"],
            extra_cuda_cflags=["03", "--use_fast_math"],
            verbose=False)

M, H = 1024, 4096
dtype = torch.float16
x = torch.randn(M, H, device="cuda", dtype=dtype)

# run welfords
mean_cuda, var_cuda = ext.welford_test(x)

# pytorch reference
mean_ref = x.float().mean(dim=-1)
var_ref = x.float().var(dim=-1, unbiased=False)

mae_mean = (mean_ref - mean_cuda).abs().max().item()
mae_var = (var_ref - var_cuda).abs().max().item()

print(f"max |D mean| = {mae_mean:.3e}")
print(f"max |D var| = {mae_var:.3e}")
