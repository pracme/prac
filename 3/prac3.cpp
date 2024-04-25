#include <iomanip>
#include <bits/stdc++.h>

using namespace std;

struct ReductionData {
   double min_val, max_val, sum_val;
};

vector<double> generate_random_data(int size) {
   random_device rd;
   mt19937 gen(rd());
   uniform_real_distribution<double> dis(1.0, 100.0);

   vector<double> data(size);
   for (int i = 0; i < size; ++i) {
       data[i] = dis(gen);
       cout << data[i] << " ";
   }
   cout << endl;
   return data;
}


ReductionData sequential_reduction(const vector<double> &arr) {
   ReductionData result = {numeric_limits<double>::max(), numeric_limits<double>::min(), 0.0};

   for (size_t i = 0; i < arr.size(); ++i) {
       result.min_val = min(result.min_val, arr[i]);
       result.max_val = max(result.max_val, arr[i]);
       result.sum_val += arr[i];
   }
   return result;
}

#pragma omp declare reduction(ReductionDataOp:ReductionData : omp_out.min_val = min(omp_out.min_val, omp_in.min_val), omp_out.max_val = max(omp_out.max_val, omp_in.max_val), omp_out.sum_val += omp_in.sum_val)


ReductionData parallel_reduction(const vector<double> &arr) {
   ReductionData result = {numeric_limits<double>::max(), numeric_limits<double>::lowest(), 0.0};

#pragma omp parallel
   {
       ReductionData local_result = result;

#pragma omp for
       for (size_t i = 0; i < arr.size(); ++i) {
           local_result.min_val = min(local_result.min_val, arr[i]);
           local_result.max_val = max(local_result.max_val, arr[i]);
           local_result.sum_val += arr[i];
       }

#pragma omp critical
       {
           result.min_val = min(result.min_val, local_result.min_val);
           result.max_val = max(result.max_val, local_result.max_val);
           result.sum_val += local_result.sum_val;
       }
   }
   return result;
}


int main() {
   int size_of_data;
   cout << "Enter Total Numbers to Generate/Compute: ";
   cin >> size_of_data;
   vector<double> data = generate_random_data(size_of_data);

   auto start_parallel = chrono::steady_clock::now();
   ReductionData parallel_result = parallel_reduction(data);
   auto end_parallel = chrono::steady_clock::now();
   chrono::duration<double> parallel_time = end_parallel - start_parallel;

   auto start_sequential = chrono::steady_clock::now();
   ReductionData sequential_result = sequential_reduction(data);
   auto end_sequential = chrono::steady_clock::now();
   chrono::duration<double> sequential_time = end_sequential - start_sequential;
   double average = sequential_result.sum_val / data.size();

   cout << "\nSequential Results:" << endl;
   cout << "Minimum Number: " << sequential_result.min_val << endl;
   cout << "Maximum Number: " << sequential_result.max_val << endl;
   cout << "Total Sum: " << sequential_result.sum_val << endl;
   cout << "Average of Numbers: " << average << endl;
   cout << "Time Taken for Sequential Reduction: " <<std::fixed << std::setprecision(7) << sequential_time.count() << " seconds" << endl;

   cout << "\nParallel Results:" << endl;
   cout << "Minimum Number: " << parallel_result.min_val << endl;
   cout << "Maximum Number: " << parallel_result.max_val << endl;
   cout << "Total Sum: " << parallel_result.sum_val << endl;
   cout << "Average of Numbers: " << average << endl;
   cout << "Time taken for Parallel Reduction: " << std::fixed << std::setprecision(7) << parallel_time.count() << " seconds\n" << endl;

   return 0;
}