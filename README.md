# Resources

1. [Cuda C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)


# Python configuration
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Usage
Compile
```
nvcc hello_cuda.cu -o hello_cuda
```

Run
```
./hello_cuda
```
