#include <iostream>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>

#define N 100000 // Number of data points

// Function to perform linear regression
void linearRegressionSerial(float *x, float *y, float &slope, float &intercept) {
    float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (int i = 0; i < N; ++i) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }
    float meanX = sumX / N;
    float meanY = sumY / N;
    slope = (sumXY - sumX * meanY) / (sumX2 - sumX * meanX);
    intercept = meanY - slope * meanX;
}

int main() {
    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 100.0);
    float x[N], y[N];
    for (int i = 0; i < N; ++i) {
        x[i] = dis(gen);
        y[i] = 3 * x[i] + 2 + dis(gen);
    }

    // Timing for serial implementation
    auto start_serial = std::chrono::steady_clock::now();
    float slope, intercept;
    linearRegressionSerial(x, y, slope, intercept);
    auto end_serial = std::chrono::steady_clock::now();
    std::chrono::duration<double> serial_time = end_serial - start_serial;
    
    std::cout << "Linear Regression (Serial): y = " << std::fixed << std::setprecision(6)<<slope << " * x + " <<std::fixed << std::setprecision(6)<< intercept << std::endl;

    std::cout << "Time taken for Serial LR: " << std::fixed << std::setprecision(6) << serial_time.count() << " seconds" << std::endl;
    return 0;
}
