NVCC = nvcc
CFLAGS = -O2 -arch=sm_50

hello_cuda: hello_cuda.cu
	$(NVCC) $(CFLAGS) -o hello_cuda hello_cuda.cu

clean:
	rm -f hello_cuda

.PHONY: clean