#include <torch/extension.h>
std::vector<torch::Tensor> welford_test_cuda(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("welford_test", &welford_test_cuda, "Welford mean/var test (CUDA)");
}