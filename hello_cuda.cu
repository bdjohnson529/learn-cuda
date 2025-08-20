#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    printf("CUDA Hello World Example\n");
    
    // Launch hello kernel with 2 blocks of 4 threads each
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    
    printf("\nVector Addition Example\n");
    
    // Simple vector addition
    const int N = 8;
    float h_a[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    float h_b[N] = {8, 7, 6, 5, 4, 3, 2, 1};
    float h_c[N];
    
    float *d_a, *d_b, *d_c;
    
    // Allocate GPU memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    // Copy data to GPU
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel with 1 block of 8 threads
    vector_add<<<1, 8>>>(d_a, d_b, d_c, N);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print results
    for (int i = 0; i < N; i++) {
        printf("%.1f + %.1f = %.1f\n", h_a[i], h_b[i], h_c[i]);
    }
    
    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}