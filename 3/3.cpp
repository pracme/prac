#include <iostream>
#include <iomanip>
#include <vector>
#include <limits>
#include <numeric>
#include <chrono>
#include <random>
#include <omp.h>
using namespace std;

// Struct to hold reduction data (minimum, maximum, sum)
struct ReductionData
{
  double min_val; // Minimum value
  double max_val; // Maximum value
  double sum_val; // Sum of all values
};

// Function to generate a random vector of specified size and print it
std::vector<double> generate_random_data(int size)
{
  std::random_device rd;                                  // Random device for seed generation
  std::mt19937 gen(rd());                                 // Mersenne Twister engine seeded with rd
  std::uniform_real_distribution<double> dis(1.0, 100.0); // Uniform distribution between 1.0 and 100.0

  std::vector<double> data(size); // Vector to store random data
  std::cout << "\n-> Generated Data:" << std::endl;
  // Generating random data and printing it
  for (int i = 0; i < size; ++i)
  {
    data[i] = dis(gen);                                                // Generating a random number
    std::cout << std::fixed << std::setprecision(7) << data[i] << " "; // Printing the generated number
  }
  std::cout << std::endl;
  return data; // Returning the vector of random data
}

// Function to perform parallel reduction for finding minimum, maximum, and sum
#pragma omp declare reduction(ReductionDataOp:ReductionData : omp_out.min_val = std::min(omp_out.min_val, omp_in.min_val), omp_out.max_val = std::max(omp_out.max_val, omp_in.max_val), omp_out.sum_val += omp_in.sum_val)

// Function to perform parallel reduction operation on an array of doubles
ReductionData parallel_reduction(const std::vector<double> &arr)
{
  // Initialize the result struct with initial values
  ReductionData result = {std::numeric_limits<double>::max(),    // Initialize minimum value to maximum possible value
                          std::numeric_limits<double>::lowest(), // Initialize maximum value to lowest possible value
                          0.0};                                  // Initialize sum to 0.0

#pragma omp parallel // Start parallel region
  {
    ReductionData local_result = result; // Create local copy of result struct for each thread

#pragma omp for                             // Start parallel loop with automatic workload distribution
    for (size_t i = 0; i < arr.size(); ++i) // Iterate over array elements
    {
      // Update local_result struct with minimum, maximum, and sum values
      local_result.min_val = std::min(local_result.min_val, arr[i]); // Find minimum value
      local_result.max_val = std::max(local_result.max_val, arr[i]); // Find maximum value
      local_result.sum_val += arr[i];                                // Calculate sum
    }

#pragma omp critical // Ensure atomic access to shared result struct
    {
      // Update shared result struct with values from local_result struct
      result.min_val = std::min(result.min_val, local_result.min_val); // Update minimum value
      result.max_val = std::max(result.max_val, local_result.max_val); // Update maximum value
      result.sum_val += local_result.sum_val;                          // Update sum
    }
  }
  return result; // Return the final result struct
}

// Function to perform non-parallel operations for finding minimum, maximum, sum, and average
ReductionData sequential_reduction(const std::vector<double> &arr)
{
  // Initialize the result struct with initial values
  ReductionData result = {
      std::numeric_limits<double>::max(), // Initialize minimum value to maximum possible value
      std::numeric_limits<double>::min(), // Initialize maximum value to minimum possible value
      0.0                                 // Initialize sum to 0.0
  };

  // Loop through each element in the array
  for (size_t i = 0; i < arr.size(); ++i)
  {
    // Update the minimum value if the current element is smaller
    result.min_val = std::min(result.min_val, arr[i]);
    // Update the maximum value if the current element is larger
    result.max_val = std::max(result.max_val, arr[i]);
    // Add the current element to the sum
    result.sum_val += arr[i];
  }
  return result; // Return the final result struct containing the minimum, maximum, and sum values
}

int main()
{
  cout << "\n ***** [ HPC ASSIGNMENT 3 ] ***** \n --------------------------------\n\n-> Generating Random Numbers......\n\n-> Enter Total Numbers to Generate/Compute: ";
  int size_of_data = 0; // Change this value to the desired number of elements
  cin >> size_of_data;
  std::vector<double> data = generate_random_data(size_of_data); // Generate random data
  auto start_parallel = std::chrono::steady_clock::now();        // Measure time for parallel reduction
  ReductionData parallel_result = parallel_reduction(data);
  auto end_parallel = std::chrono::steady_clock::now();
  std::chrono::duration<double> parallel_time = end_parallel - start_parallel;

  auto start_sequential = std::chrono::steady_clock::now(); // Measure time for sequential reduction
  ReductionData sequential_result = sequential_reduction(data);
  auto end_sequential = std::chrono::steady_clock::now();
  std::chrono::duration<double> sequential_time = end_sequential - start_sequential;
  double average = sequential_result.sum_val / data.size();

  std::cout << "\n-> Sequential Results:" << std::endl;
  std::cout << "-> Minimum Number: " << sequential_result.min_val << std::endl;
  std::cout << "-> Maximum Number: " << sequential_result.max_val << std::endl;
  std::cout << "-> Total Sum: " << sequential_result.sum_val << std::endl;
  std::cout << "-> Average of Numbers: " << std::fixed << std::setprecision(8) << average << std::endl;
  std::cout << "-> Time Taken for Sequential Reduction: " << std::fixed << std::setprecision(7) << sequential_time.count() << " seconds" << std::endl;

  std::cout << "\n-> Parallel Results:" << std::endl;
  std::cout << "-> Minimum Number: " << parallel_result.min_val << std::endl;
  std::cout << "-> Maximum Number: " << parallel_result.max_val << std::endl;
  std::cout << "-> Total Sum: " << parallel_result.sum_val << std::endl;
  std::cout << "-> Average of Numbers: " << std::fixed << std::setprecision(8) << average << std::endl;
  std::cout << "-> Time taken for Parallel Reduction: " << std::fixed << std::setprecision(7) << parallel_time.count() << " seconds\n\n"
            << std::endl;

  return 0;
}
