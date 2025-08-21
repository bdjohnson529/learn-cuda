## CUDA Notes
From reading the [C programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide)

* Grid → made of blocks.
* Block cluster (OPTIONAL) → made of blocks
* Block → made of threads.
* Thread → executes your kernel code.

Thread blocks are required to execute independently.
Threads within a block can cooperate by sharing data through some shared memory and by synchronizing execution.

#### Memory

Thread memory - specific to a thread
Block memory - shared by all threads in a block
Distributed shared memory - shared by all thread blocks in a cluster
Global memory - shared by all kernels

Global memory is allocated usin `cudaMalloc()`, for example
```
cudaMalloc(&d_a, N * sizeof(float));
```

Memory can then be copied from the host to device
```
cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
```

#### Kernels

Kernels are C++ functions which are executed by threads, in parallel.

#### Heterogenous Programming

* Host - CPU is manager. Serial code executes here.
* Device - GPU is worker. Parallel code executes here.

Host and device have separate memory spaces.
Unified memory is managed memory to bridge between device and host.

#### Synchronicity

Threads are asynchronous.
Threads that trigger an async operation does not need to be the thread which collects the operation.

#### Memory

Device memory can be allocated either as linear memory or as CUDA arrays.
* Linear memory is 1-dimensional block of addresses
* CUDA arrays are multi-dimensional

Variables need to be copied from host memory to device memory.

L2 cache can be used for persistent memory.

`hitRatio` can be used to reduce the amount of the access window stored in the L2 cache.

Types of global memory access:
* cudaAccessPropertyStreaming (not likely to persist in L2 cache)
* cudaAccessPropertyPersisting (likely to persist in L2 cache)
* cudaAccessPropertyNormal (removes persisting status of previous memory)


Shared memory is faster than global memory because it is on chip. Shared memory is allocated per thread block, so all threads in the block have access to the same memory. Declare shared memory using `__shared__`.

Page-locked (pinned) host memory advantages (versus regular pageable host memory):
* copies from host to device can be concurrent with kernel execution
* can be mapped directly to device address, eliminating copy
* bandwidth is higher