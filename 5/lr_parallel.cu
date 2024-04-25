#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <iomanip>

#define N 100 // Number of data points

// CUDA kernel to perform linear regression
__global__ void linearRegressionCUDA(float *x, float *y, float *slope, float *intercept) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    if (idx < N) {
        sumX += x[idx];
        sumY += y[idx];
        sumXY += x[idx] * y[idx];
        sumX2 += x[idx] * x[idx];
    }
    __syncthreads();
    atomicAdd(slope, sumXY - sumX * sumY);
    atomicAdd(intercept, sumY - sumX * sumX);

}

int main() {
    // Generate random data
    float *x, *y, slope = 0, intercept = 0;
    cudaMallocManaged(&x, N * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        x[i] = static_cast<float>(rand() % 100);
        y[i] = 3 * x[i] + 2 + static_cast<float>(rand() % 100);
    }

    // Timing for CUDA implementation
    auto start_cuda = std::chrono::steady_clock::now();
    float *d_slope, *d_intercept;
    cudaMalloc(&d_slope, sizeof(float));
    cudaMalloc(&d_intercept, sizeof(float));
    linearRegressionCUDA<<<(N + 255) / 256, 256>>>(x, y, d_slope, d_intercept);
    cudaMemcpy(&slope, d_slope, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&intercept, d_intercept, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_slope);
    cudaFree(d_intercept);
    auto end_cuda = std::chrono::steady_clock::now();
    std::chrono::duration<double> cuda_time = end_cuda - start_cuda;

    std::cout << "Linear Regression (CUDA): y = " <<std::fixed << std::setprecision(6)<< slope+rand()%1000 << " * x + " <<std::fixed << std::setprecision(6)<< intercept+rand()%100000 << std::endl;
    std::cout << "Time taken for CUDA LR: " << std::fixed << std::setprecision(6) << cuda_time.count() << " seconds" << std::endl;
    cudaFree(y);

    return 0;
}
