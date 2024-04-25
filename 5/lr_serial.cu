#include <iostream>                             // Include the input/output stream header
#include <chrono>                               // Include the header for time measurements
#include <random>                               // Include the header for random number generation
#include <cmath>                                // Include the header for mathematical functions
#include <iomanip>                              // Include the input/output manipulator header

#define N 100000                                // Define the number of data points

/**
 * @brief Function to perform linear regression.
 * 
 * @param x Pointer to the input data vector x.
 * @param y Pointer to the input data vector y.
 * @param slope Reference to store the calculated slope of the linear regression.
 * @param intercept Reference to store the calculated intercept of the linear regression.
 */
void linearRegressionSerial(float *x, float *y, float &slope, float &intercept) {
    float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;     // Initialize variables for summation
    for (int i = 0; i < N; ++i) {                       // Loop through each data point
        sumX += x[i];                                   // Accumulate sum of x values
        sumY += y[i];                                   // Accumulate sum of y values
        sumXY += x[i] * y[i];                           // Accumulate sum of x*y values
        sumX2 += x[i] * x[i];                           // Accumulate sum of x^2 values
    }
    float meanX = sumX / N;                             // Calculate mean of x values
    float meanY = sumY / N;                             // Calculate mean of y values
    slope = (sumXY - sumX * meanY) / (sumX2 - sumX * meanX);   // Compute slope of the linear regression
    intercept = meanY - slope * meanX;                   // Compute intercept of the linear regression
}

/**
 * @brief Main function to perform linear regression serially.
 * 
 * @return int 0 on successful execution.
 */
int main() {
    // Generate random data
    std::random_device rd;                             // Create a random device
    std::mt19937 gen(rd());                             // Create a Mersenne Twister random number generator
    std::uniform_real_distribution<float> dis(0.0, 100.0); // Create a uniform real distribution between 0 and 100
    float x[N], y[N];                                   // Declare arrays to store x and y data
    for (int i = 0; i < N; ++i) {                       // Loop to generate random data points
        x[i] = dis(gen);                                // Generate random x value
        y[i] = 3 * x[i] + 2 + dis(gen);                 // Generate corresponding y value for linear relationship
    }

    // Timing for serial implementation
    auto start_serial = std::chrono::steady_clock::now();   // Record the start time for serial implementation
    float slope, intercept;                                 // Declare variables to store slope and intercept
    linearRegressionSerial(x, y, slope, intercept);          // Call function to perform linear regression
    auto end_serial = std::chrono::steady_clock::now();     // Record the end time for serial implementation
    std::chrono::duration<double> serial_time = end_serial - start_serial;   // Calculate the elapsed time for serial implementation
    
    // Print results
    std::cout << "Linear Regression (Serial): y = " << std::fixed << std::setprecision(6)<<slope << " * x + " <<std::fixed << std::setprecision(6)<< intercept << std::endl;   // Print the equation of the linear regression line
    std::cout << "Time taken for Serial LR: " << std::fixed << std::setprecision(6) << serial_time.count() << " seconds" << std::endl;    // Print the time taken for serial implementation
    return 0;                                             // Return 0 to indicate successful execution
}
