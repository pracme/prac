#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

#define N 10000    // Number of data points
#define D 10       // Number of features

/**
 * @brief Function for logistic regression.
 * 
 * @param X Pointer to the input data matrix.
 * @param y Pointer to the target labels.
 * @param theta Pointer to the parameter vector.
 * @param cost Pointer to store the calculated cost.
 * @param num_features Number of features.
 */
void logisticRegression(float *X, float *y, float *theta, float *cost, int num_features) {
    for (int i = 0; i < N; ++i) {
        float dot_product = 0;
        // Compute dot product of feature vector and parameter vector
        for (int j = 0; j < num_features; ++j) {
            dot_product += X[i * num_features + j] * theta[j];
        }
        // Compute hypothesis using logistic function
        float hypothesis = 1.0f / (1.0f + exp(-dot_product));
        // Compute cost using logistic regression cost function
        cost[i] = -y[i] * log(hypothesis) - (1 - y[i]) * log(1 - hypothesis);
    }
}

/**
 * @brief Main function to perform logistic regression sequentially.
 * 
 * @return int 0 on successful execution.
 */
int main() {
    // Initialize random data
    float *X, *y, *theta, *cost;
    X = new float[N * D];
    y = new float[N];
    theta = new float[D];
    cost = new float[N];

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

    // Run logistic regression sequentially
    logisticRegression(X, y, theta, cost, D);

    // End time measurement
    clock_t end = clock();
    double sequential_time = double(end - start) / CLOCKS_PER_SEC;

    // Calculate total cost
    float total_cost = 0;
    for (int i = 0; i < N; ++i) {
        total_cost += cost[i];
    }
    total_cost /= N;
    std::cout << "Total cost (sequential): " << total_cost << std::endl;
    std::cout << "Time taken (sequential): " << sequential_time << " seconds" << std::endl;

    // Free allocated memory
    delete[] X;
    delete[] y;
    delete[] theta;
    delete[] cost;

    return 0;
}
