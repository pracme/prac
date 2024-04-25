#include <iostream>
#include <chrono>

#define N 1000000

void vectorMulSequential(int *a, int *b, int *c) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] * b[i];
    }
}

__global__ void vectorMul(int *a, int *b, int *c) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < N) {
        c[tid] = a[tid] * b[tid];
    }
}

int main() {
    int *a, *b, *c;            // host copies of a, b, c
    int *d_a, *d_b, *d_c;    // device copies of a, b, c
    int size = N * sizeof(int);

    // Allocate space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate space for host copies of a, b, c and setup input values
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Sequential version
    auto start_seq = std::chrono::high_resolution_clock::now();
    vectorMulSequential(a, b, c);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_seq = end_seq - start_seq;
    std::cout << "Sequential Execution Time: " << duration_seq.count() << " ms" << std::endl;

    // Print sequential output
    std::cout << "Sequential Output:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " * " << b[i] << " = " << c[i] << std::endl;
    }
    std::cout << "..." << std::endl;

    // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // CUDA version
    auto start_cuda = std::chrono::high_resolution_clock::now();

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorMul<<<numBlocks, blockSize>>>(d_a, d_b, d_c);
    
    cudaDeviceSynchronize(); // Wait for the GPU to finish

    auto end_cuda = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration_cuda = end_cuda - start_cuda;
    std::cout << "CUDA Execution Time: " << duration_cuda.count() << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print CUDA output
    std::cout << "CUDA Output:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << a[i] << " * " << b[i] << " = " << c[i] << std::endl;
    }
    std::cout << "..." << std::endl;

    // Cleanup
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

