#include <iostream>
#include <cmath>
#include <chrono>
#include <float.h> // Include for FLT_MAX
#include <cuda_runtime.h>
#include <iomanip>

#define N 10000 // Number of data points
#define K 3   // Number of clusters

// Function to calculate Euclidean distance between two points
__device__ float distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

// CUDA kernel to assign each point to its nearest cluster
__global__ void assignClusters(float *dataX, float *dataY, float *centroidsX, float *centroidsY, int *assignments) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float minDist = FLT_MAX;
        int cluster = -1;
        for (int i = 0; i < K; ++i) {
            float dist = distance(dataX[idx], dataY[idx], centroidsX[i], centroidsY[i]);
            if (dist < minDist) {
                minDist = dist;
                cluster = i;
            }
        }
        assignments[idx] = cluster;
    }
}

// CUDA kernel to update cluster centroids
__global__ void updateCentroids(float *dataX, float *dataY, int *assignments, float *centroidsX, float *centroidsY) {
    int clusterId = threadIdx.x;
    float sumX = 0, sumY = 0;
    int count = 0;
    for (int i = 0; i < N; ++i) {
        if (assignments[i] == clusterId) {
            sumX += dataX[i];
            sumY += dataY[i];
            count++;
        }
    }
    if (count > 0) {
        centroidsX[clusterId] = sumX / count;
        centroidsY[clusterId] = sumY / count;
    }
}

int main() {
    // Generate synthetic data
    float dataX[N], dataY[N];
    for (int i = 0; i < N; ++i) {
        dataX[i] = static_cast<float>(rand() % 100);
        dataY[i] = static_cast<float>(rand() % 100);
    }

    // Initialize cluster centroids randomly
    float centroidsX[K], centroidsY[K];
    for (int i = 0; i < K; ++i) {
        centroidsX[i] = static_cast<float>(rand() % 100);
        centroidsY[i] = static_cast<float>(rand() % 100);
    }

    // Allocate device memory
    float *d_dataX, *d_dataY, *d_centroidsX, *d_centroidsY;
    int *d_assignments;
    cudaMalloc(&d_dataX, N * sizeof(float));
    cudaMalloc(&d_dataY, N * sizeof(float));
    cudaMalloc(&d_centroidsX, K * sizeof(float));
    cudaMalloc(&d_centroidsY, K * sizeof(float));
    cudaMalloc(&d_assignments, N * sizeof(int));

    // Copy data to device
    cudaMemcpy(d_dataX, dataX, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataY, dataY, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroidsX, centroidsX, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroidsY, centroidsY, K * sizeof(float), cudaMemcpyHostToDevice);

    // Start the timer
    auto start_time = std::chrono::steady_clock::now();

    // Iterations of K-means
    const int iterations = 10;
    for (int iter = 0; iter < iterations; ++iter) {
        // Assign each point to its nearest cluster
        assignClusters<<<(N + 255) / 256, 256>>>(d_dataX, d_dataY, d_centroidsX, d_centroidsY, d_assignments);

        // Update cluster centroids
        updateCentroids<<<1, K>>>(d_dataX, d_dataY, d_assignments, d_centroidsX, d_centroidsY);
    }

    // End the timer
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // Copy cluster assignments back to host
    int assignments[N];
    cudaMemcpy(assignments, d_assignments, N * sizeof(int), cudaMemcpyDeviceToHost);

    // // Print results
    // std::cout << "Final cluster assignments:" << std::endl;
    // for (int i = 0; i < N; ++i) {
    //     std::cout << "Data point " << i << " assigned to cluster " << assignments[i] << std::endl;
    // }

    // Print results
    std::cout << "Final cluster centroids:" << std::endl;
    for (int i = 0; i < K; ++i) {
        std::cout << "Cluster " << i << ": (" << centroidsX[i] << ", " << centroidsY[i] << ")" << std::endl;
    }

    // Print the time taken
    std::cout << "Time taken for K-means with CUDA: "<<std::fixed << std::setprecision(6)  << elapsed_seconds.count() << " seconds" << std::endl;

    // Free device memory
    cudaFree(d_dataX);
    cudaFree(d_dataY);
    cudaFree(d_centroidsX);
    cudaFree(d_centroidsY);
    cudaFree(d_assignments);

    return 0;
}
