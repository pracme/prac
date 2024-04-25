#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define N 10000    // Number of data points
#define D 10       // Number of features
#define BLOCK_SIZE 256

/**
 * @brief Kernel function for logistic regression.
 * 
 * @param X Pointer to the input data matrix.
 * @param y Pointer to the target labels.
 * @param theta Pointer to the parameter vector.
 * @param cost Pointer to store the calculated cost.
 * @param num_features Number of features.
 */
__global__ void logisticRegression(float *X, float *y, float *theta, float *cost, int num_features) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float dot_product = 0;
        // Compute dot product of feature vector and parameter vector
        for (int j = 0; j < num_features; ++j) {
            dot_product += X[idx * num_features + j] * theta[j];
        }
        // Compute hypothesis using logistic function
        float hypothesis = 1.0f / (1.0f + exp(-dot_product));
        // Compute cost using logistic regression cost function
        cost[idx] = -y[idx] * log(hypothesis) - (1 - y[idx]) * log(1 - hypothesis);
    }
}

/**
 * @brief Main function to perform logistic regression on GPU.
 * 
 * @return int 0 on successful execution.
 */
int main() {
    // Initialize random data
    float *X, *y, *theta, *cost;
    cudaMallocManaged(&X, N * D * sizeof(float));
    cudaMallocManaged(&y, N * sizeof(float));
    cudaMallocManaged(&theta, D * sizeof(float));
    cudaMallocManaged(&cost, N * sizeof(float));

    // Initialize random number generator
    srand(time(NULL));

    // Initialize X, y, and theta with random values
    for (int i = 0; i < N * D; ++i) {
        X[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }
    for (int i = 0; i < N; ++i) {
        y[i] = rand() % 2;  // Random binary classification labels (0 or 1)
    }
    for (int i = 0; i < D; ++i) {
        theta[i] = static_cast<float>(rand()) / RAND_MAX; // Random float between 0 and 1
    }

    // Start time measurement
    clock_t start = clock();

    // Run logistic regression on GPU
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    logisticRegression<<<num_blocks, BLOCK_SIZE>>>(X, y, theta, cost, D);
    cudaDeviceSynchronize();

    // End time measurement
    clock_t end = clock();
    double cuda_time = double(end - start) / CLOCKS_PER_SEC;

    // Calculate total cost
    float total_cost = 0;
    for (int i = 0; i < N; ++i) {
        total_cost += cost[i];
    }
    total_cost /= N;
    std::cout << "Total cost (CUDA): " << total_cost << std::endl;
    std::cout << "Time taken (CUDA): " << cuda_time << " seconds" << std::endl;

    // Free allocated memory
    cudaFree(X);
    cudaFree(y);
    cudaFree(theta);
    cudaFree(cost);

    return 0;
}
