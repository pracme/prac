#include <iostream>
#include <cmath>
#include <chrono>

#define N 50 // Number of data points
#define K 5    // Number of nearest neighbors to consider

// Function to calculate Euclidean distance between two points
float distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

// Function to find K nearest neighbors for each point
void knn_serial(float *trainX, float *trainY, float *testX, float *testY, int *labels) {
    for (int i = 0; i < N; ++i) {
        float minDist[K];
        int minIdx[K];

        // Initialize arrays with large values
        for (int j = 0; j < K; ++j) {
            minDist[j] = std::numeric_limits<float>::max();
            minIdx[j] = -1;
        }

        // Calculate distances to training points
        for (int j = 0; j < N; ++j) {
            float dist = distance(testX[i], testY[i], trainX[j], trainY[j]);
            // Update nearest neighbors
            for (int k = 0; k < K; ++k) {
                if (dist < minDist[k]) {
                    for (int l = K - 1; l > k; --l) {
                        minDist[l] = minDist[l - 1];
                        minIdx[l] = minIdx[l - 1];
                    }
                    minDist[k] = dist;
                    minIdx[k] = j;
                    break;
                }
            }
        }

        // Count labels of nearest neighbors
        int counts[3] = {0};
        for (int j = 0; j < K; ++j) {
            int label = labels[minIdx[j]];
            counts[label]++;
        }

        // Determine majority label
        int majorityLabel = 0;
        int maxCount = counts[0];
        for (int j = 1; j < 3; ++j) {
            if (counts[j] > maxCount) {
                maxCount = counts[j];
                majorityLabel = j;
            }
        }

        // Assign the majority label to the test point
        labels[i] = majorityLabel;
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

    // Timing for serial implementation
    auto start_serial = std::chrono::steady_clock::now();
    knn_serial(trainX, trainY, testX, testY, labels);
    auto end_serial = std::chrono::steady_clock::now();
    std::chrono::duration<double> serial_time = end_serial - start_serial;
    
    
// Print results
std::cout << "Classification labels assigned by KNN CUDA:" << std::endl;
for (int i = 0; i < N; ++i) {
    std::cout << "Data point " << i << ": " << labels[i] << std::endl;
}

std::cout << "Time taken for serial KNN: " << serial_time.count() << " seconds" << std::endl;
    return 0;
}


