# Resources

1. [Cuda C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)


## Notes

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


#### Heterogenous Programming

* Host - CPU is manager. Serial code executes here.
* Device - GPU is worker. Parallel code executes here.

Host and device have separate memory spaces.
Unified memory is managed memory to bridge between device and host.

#### Synchronicity

Threads are asynchronous.
Threads that trigger an async operation does not need to be the thread which collects the operation.