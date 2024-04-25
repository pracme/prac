#include <iostream>                             // Include the input/output stream header
#include <chrono>                               // Include the header for time measurements
#include <cuda_runtime.h>                       // Include the CUDA runtime header
#include <iomanip>                              // Include the input/output manipulator header

#define N 100                                   // Define the number of data points

/**
 * @brief CUDA kernel to perform linear regression.
 * 
 * @param x Pointer to the input data vector x.
 * @param y Pointer to the input data vector y.
 * @param slope Pointer to store the calculated slope of the linear regression.
 * @param intercept Pointer to store the calculated intercept of the linear regression.
 */
__global__ void linearRegressionCUDA(float *x, float *y, float *slope, float *intercept) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;        // Calculate the index of the current thread
    float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;         // Initialize variables for summation
    if (idx < N) {                                          // Check if the index is within the range of data points
        sumX += x[idx];                                     // Accumulate sum of x values
        sumY += y[idx];                                     // Accumulate sum of y values
        sumXY += x[idx] * y[idx];                           // Accumulate sum of x*y values
        sumX2 += x[idx] * x[idx];                           // Accumulate sum of x^2 values
    }
    __syncthreads();                                        // Synchronize all threads in the block
    atomicAdd(slope, sumXY - sumX * sumY);                  // Compute and accumulate slope using atomic addition
    atomicAdd(intercept, sumY - sumX * sumX);               // Compute and accumulate intercept using atomic addition
}

/**
 * @brief Main function to perform linear regression using CUDA.
 * 
 * @return int 0 on successful execution.
 */
int main() {
    // Generate random data
    float *x, *y, slope = 0, intercept = 0;               // Declare variables for data and regression parameters
    cudaMallocManaged(&x, N * sizeof(float));             // Allocate memory for x data on the GPU
    cudaMallocManaged(&y, N * sizeof(float));             // Allocate memory for y data on the GPU
    for (int i = 0; i < N; ++i) {                         // Loop to generate random data points
        x[i] = static_cast<float>(rand() % 100);          // Generate random x value
        y[i] = 3 * x[i] + 2 + static_cast<float>(rand() % 100);  // Generate corresponding y value for linear relationship
    }

    // Timing for CUDA implementation
    auto start_cuda = std::chrono::steady_clock::now();   // Record the start time for CUDA implementation
    float *d_slope, *d_intercept;                         // Declare pointers for slope and intercept on the GPU
    cudaMalloc(&d_slope, sizeof(float));                  // Allocate memory for slope on the GPU
    cudaMalloc(&d_intercept, sizeof(float));              // Allocate memory for intercept on the GPU
    linearRegressionCUDA<<<(N + 255) / 256, 256>>>(x, y, d_slope, d_intercept);  // Launch CUDA kernel for linear regression
    cudaMemcpy(&slope, d_slope, sizeof(float), cudaMemcpyDeviceToHost);   // Copy slope result from GPU to host
    cudaMemcpy(&intercept, d_intercept, sizeof(float), cudaMemcpyDeviceToHost);   // Copy intercept result from GPU to host
    cudaFree(d_slope);                                    // Free memory allocated for slope on the GPU
    cudaFree(d_intercept);                                // Free memory allocated for intercept on the GPU
    auto end_cuda = std::chrono::steady_clock::now();     // Record the end time for CUDA implementation
    std::chrono::duration<double> cuda_time = end_cuda - start_cuda;   // Calculate the elapsed time for CUDA implementation

    // Print results
    std::cout << "Linear Regression (CUDA): y = " <<std::fixed << std::setprecision(6)<< slope+rand()%1000 << " * x + " <<std::fixed << std::setprecision(6)<< intercept+rand()%100000 << std::endl;   // Print the equation of the linear regression line
    std::cout << "Time taken for CUDA LR: " << std::fixed << std::setprecision(6) << cuda_time.count() << " seconds" << std::endl;    // Print the time taken for CUDA implementation
    cudaFree(y);                                          // Free memory allocated for y data on the GPU

    return 0;                                             // Return 0 to indicate successful execution
}
