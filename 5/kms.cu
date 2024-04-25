#include <iostream>
#include <cmath>
#include <chrono>
#include <limits>
#include <iomanip>

#define N 10000 // Number of data points
#define K 3     // Number of clusters

// Function to calculate Euclidean distance between two points
float distance(float x1, float y1, float x2, float y2) {
    return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
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

    // Assignments array declaration
    int assignments[N];

    // Start the timer
    auto start_time = std::chrono::steady_clock::now();

    // Iterations of K-means
    const int iterations = 10;
    for (int iter = 0; iter < iterations; ++iter) {
        // Assign each point to its nearest cluster
        for (int i = 0; i < N; ++i) {
            float minDist = std::numeric_limits<float>::max();
            int cluster = -1;
            for (int j = 0; j < K; ++j) {
                float dist = distance(dataX[i], dataY[i], centroidsX[j], centroidsY[j]);
                if (dist < minDist) {
                    minDist = dist;
                    cluster = j;
                }
            }
            assignments[i] = cluster;
        }

        // Update cluster centroids
        for (int clusterId = 0; clusterId < K; ++clusterId) {
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
    }

    // End the timer
    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    // // Print results
    // std::cout << "Final cluster assignments:" << std::endl;
    // for (int i = 0; i < N; ++i) {
    //     std::cout << "Data point " << i << " assigned to cluster " << assignments[i] << std::endl;
    // }

    std::cout << "Final cluster centroids:" << std::endl;
    for (int i = 0; i < K; ++i) {
        std::cout << "Cluster " << i << ": (" << centroidsX[i] << ", " << centroidsY[i] << ")" << std::endl;
    }

    // Print the time taken
    std::cout << "Time taken for K-means: " <<std::fixed << std::setprecision(6)  << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
