#include <iostream>
#include <cuda_runtime.h>
#include <float.h> // Include for FLT_MAX
#include <chrono>
#include <iomanip>

#define N 50 // Number of data points
#define K 5    // Number of nearest neighbors to consider

// Function to calculate Euclidean distance between two points
__device__ float distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

// Kernel to find K nearest neighbors for each point
__global__ void knn_cuda(float *trainX, float *trainY, float *testX, float *testY, int *labels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        float minDist[K];
        int minIdx[K];

        // Initialize arrays with large values
        for (int i = 0; i < K; ++i) {
            minDist[i] = FLT_MAX;
            minIdx[i] = -1;
        }

        // Calculate distances to training points
        for (int i = 0; i < N; ++i) {
            float dist = distance(testX[idx], testY[idx], trainX[i], trainY[i]);
            // Update nearest neighbors
            for (int j = 0; j < K; ++j) {
                if (dist < minDist[j]) {
                    for (int k = K - 1; k > j; --k) {
                        minDist[k] = minDist[k - 1];
                        minIdx[k] = minIdx[k - 1];
                    }
                    minDist[j] = dist;
                    minIdx[j] = i;
                    break;
                }
            }
        }

        // Count labels of nearest neighbors
        int counts[3] = {0};
        for (int i = 0; i < K; ++i) {
            int label = labels[minIdx[i]];
            counts[label]++;
        }

        // Determine majority label
        int majorityLabel = 0;
        int maxCount = counts[0];
        for (int i = 1; i < 3; ++i) {
            if (counts[i] > maxCount) {
                maxCount = counts[i];
                majorityLabel = i;
            }
        }

        // Assign the majority label to the test point
        labels[idx] = majorityLabel;
    }
}

int main() {
    // Generate synthetic data
    float trainX[N], trainY[N], testX[N], testY[N];
    int labels[N];
    for (int i = 0; i < N; ++i) {
        trainX[i] = rand() % 100;
        trainY[i] = rand() % 100;
        testX[i] = rand() % 100;
        testY[i] = rand() % 100;
        labels[i] = rand() % 3; // Random label (0, 1, or 2)
    }

    // Allocate memory on GPU
    float *dev_trainX, *dev_trainY, *dev_testX, *dev_testY;
    int *dev_labels;
    cudaMalloc((void **)&dev_trainX, N * sizeof(float));
    cudaMalloc((void **)&dev_trainY, N * sizeof(float));
    cudaMalloc((void **)&dev_testX, N * sizeof(float));
    cudaMalloc((void **)&dev_testY, N * sizeof(float));
    cudaMalloc((void **)&dev_labels, N * sizeof(int));

    // Copy data from host to GPU
    cudaMemcpy(dev_trainX, trainX, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_trainY, trainY, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_testX, testX, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_testY, testY, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_labels, labels, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block size
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Timing for CUDA implementation
    auto start_cuda = std::chrono::steady_clock::now();
    knn_cuda<<<numBlocks, blockSize>>>(dev_trainX, dev_trainY, dev_testX, dev_testY, dev_labels);
    cudaDeviceSynchronize();
    auto end_cuda = std::chrono::steady_clock::now();
    std::chrono::duration<double> cuda_time = end_cuda - start_cuda;

    
    // Copy result from GPU to host
    cudaMemcpy(labels, dev_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results (omitted for brevity)
  // Print results
  std::cout << "Classification labels assigned by KNN CUDA:" << std::endl;
  for (int i = 0; i < N; ++i) {
      std::cout << "Data point " << i << ": " << labels[i] << std::endl;
  }

  std::cout << "Time taken for CUDA KNN: " << std::fixed << std::setprecision(6) << cuda_time.count() << " seconds" << std::endl;

    // Free memory on GPU
    cudaFree(dev_trainX);
    cudaFree(dev_trainY);
    cudaFree(dev_testX);
    cudaFree(dev_testY);
    cudaFree(dev_labels);

    return 0;
}
