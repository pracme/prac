#include <bits/stdc++.h>
#include <limits.h>
#include <stdlib.h>
#include <omp.h>

#include <chrono>
#include <functional>

using std :: chrono :: duration_cast;
using std :: chrono :: high_resolution_clock;
using std :: chrono :: milliseconds;
using namespace std;

// sequential functions, declarations

void sequential_average(vector<int>& array);
void sequential_max(vector<int>& array);
void sequential_min(vector<int>& array);
void sequential_sum(vector<int>& array);

// parallel computing functions, declarations

void parallel_average(vector<int>& array);
void parallel_max(vector<int>& array);
void parallel_min(vector<int>& array);
void parallel_sum(vector<int>& array);

// utility functions, declaration
template <typename T> ostream& operator << (ostream& console, const vector<T>& array);
void populating_random_values(vector<int>& array);
void initialize_array(vector<int>& array, int N);

// function to analyze the performance
int analysis(std :: function<void()> function);

int main(void) {

    int N = 0, sequential_execution = 0, parallel_execution = 0;
    vector<int> array;

    omp_set_num_threads(16);
    
    bool flag = true;
    while (flag)
    {   
        // Menu driven program ...

        cout << "-------------------------- Menu -------------------------------"  << endl;
        cout << "1. Sequential and Parallel Min Computation   "  << endl;     
        cout << "2. Sequential and Parallel Max Computation   "  << endl;
        cout << "3. Sequential and Parallel Sum Computation   "  << endl;
        cout << "4. Sequential and Parallel Average Computation " << endl;
        cout << "5. Exit "                                        << endl;
        cout << "---------------------------------------------------------------"  << endl;

        int choice = -1;
        cout << "Enter the choice : ";
        cin >> choice;

        switch(choice)
        {
            case 1:

                initialize_array(array, N);
                populating_random_values(array);

                cout << "displaying first 10 numbers " << array;

                sequential_execution = analysis([&] { sequential_min(array); });
                parallel_execution   = analysis([&] { parallel_min(array); });
                
                cout << "Sequential Min   : "   << sequential_execution << " ms" << endl;
                cout << "Parallel Min (16): "   << parallel_execution   << " ms" << endl;
                cout << "speed up           "   << (float)(sequential_execution / (float) parallel_execution) << " ms" << endl;

                break;

            case 2:

                initialize_array(array, N);
                populating_random_values(array);

                cout << "displaying first 10 numbers " << array;

                sequential_execution = analysis([&] { sequential_max(array); });
                parallel_execution   = analysis([&]{ parallel_max(array); });

                cout << "Sequential Max   : "   << sequential_execution << " ms" << endl;
                cout << "Parallel Max (16): "   << parallel_execution   << " ms" << endl;
                cout << "speed up           "   << (float)(sequential_execution / (float) parallel_execution) << " ms" << endl;
    
                break;
            
            case 3:

                initialize_array(array, N);
                populating_random_values(array);

                cout << "displaying first 10 numbers " << array;

                sequential_execution = analysis([&] { sequential_sum(array); });
                parallel_execution  = analysis([&] { parallel_sum(array); });
    
                cout << "Sequential Sum   : "   << sequential_execution << " ms" << endl;
                cout << "Parallel Sum (16): "   << parallel_execution   << " ms" << endl;
                cout << "speed up           "   << (float)(sequential_execution / (float) parallel_execution) << " ms" << endl;
    
                break;

            case 4:

                initialize_array(array, N);
                populating_random_values(array);

                cout << "displaying first 10 numbers " << array;

                sequential_execution = analysis([&] { sequential_average(array); });
                parallel_execution  = analysis([&] { parallel_average(array); });
    
                cout << "Sequential Sum   : "   << sequential_execution << " ms" << endl;
                cout << "Parallel Sum (16): "   << parallel_execution   << " ms" << endl;
                cout << "speed up           "   << (float)(sequential_execution / (float) parallel_execution) << " ms" << endl;
                   
                break;
            
            case 5:

                flag = false;
                break;

            default:

                cout << "Invalid choice !!! " << endl;
                break;
        }
    }
    

    return 0;
}

void initialize_array(vector<int>& array, int N) {

    cout << "Enter the size of the array : ";
    cin  >> N;

    array.resize(max(5, N));

    return;
}

// Min, Max, Sum, Average Sequential Functions ---------------------------

void sequential_average(vector<int>& array) {

    cout << "------ Sequential Average -----" << endl;

    long long int N = (long long int)array.size();
    long long int sum = 0;
    long long int average = 0;

    for (int i = 0; i < N; i++)
        sum += array[i];

    average = (sum / N);

    cout << "Sequential Average " << average << endl;
}

void sequential_max(vector<int>& array) {

    cout << " ----- Sequential Max ------" << endl;

    int N = array.size();
    long long int max_value = LONG_MIN;

    for (int i = 0; i < N; i++)
        if (array[i] > max_value)
            max_value = array[i];
    
    cout << "Sequential max : " << max_value << endl;
}

void sequential_min(vector<int>& array) {

    cout << " ----- Sequential Min ------" << endl;
    
    int N = array.size();
    long long int min_value = LONG_MAX;

    for (int i = 0; i < N; i++)
        if (array[i] < min_value)
            min_value = array[i];
    
    cout << "Sequential min : " << min_value << endl;
}

void sequential_sum(vector<int>& array) {

    cout << " ----- Sequential Sum ----- " << endl;

    int N = array.size();
    long long int sum = 0;

    for (int i = 0; i < N; i++)
        sum += array[i];
    
    cout << "Sequential sum : " << sum << endl;
}

// -----------------------------------------------------------------------

// ---- Min, Max, Sum, Average Parallel Functions ------------------------


void parallel_average(vector<int>& array) {

    cout << " ----- Parallel Average -----" << endl;

    long long int N = (long long int)array.size();
    long long int sum = 0;
    long long int average = 0;

    #pragma omp parallel for reduction(+:sum)num_threads(16)
    for (int i = 0; i < N; i++)
        sum += array[i];
    
    average = (sum / N);
    cout << "Parallel Average " << average << endl;
}


void parallel_max(vector<int>& array) {

    cout << " ----- Parallel Max ----- " << endl;

    int N = array.size();
    long long int max_value = LONG_MIN;

    #pragma omp parallel for reduction(max:max_value) num_threads(16)
    for (int i = 0; i < N; i++)
        if (array[i] > max_value)
            max_value = array[i];
    
    cout << "Parallel Max : " << max_value << endl; 

}


void parallel_sum(vector<int>& array) {

    cout << " ----- Parallel Sum -----" << endl;

    int N = array.size();
    long long int sum = 0;

    #pragma omp parallel for reduction(+:sum)num_threads(16)
    for (int i = 0; i < N; i++)
        sum += array[i];
    
    cout << "Parallel Sum : " << sum << endl;
}

void parallel_min(vector<int>& array) {

    cout << " ----- Parallel Min ----- " << endl;

    int N = array.size();
    long long int min_value = LONG_MAX;

    #pragma omp parallel for reduction(min:min_value) num_threads(16)
    for (int i = 0; i < N; i++)
        if (array[i] < min_value)
            min_value = array[i];
    
    cout << "Parallel Min : " << min_value << endl;

}

// -----------------------------------------------------------------------

int analysis(std :: function<void()> function) {

    // subtract end from start timepoints and cast it to required unit, nanoseconds, microseconds, milliseconds etc.
    auto start = high_resolution_clock::now();
    
    // Executing function
    function();
    auto end   = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    return duration.count();
}

// utility functions

template <typename T> ostream& operator << (ostream& console, const vector<T>& array) {

    int N = array.size();

    for (int i = 0; i < min(10, N); i++)
        console << array[i] << "  ";

    console << endl;
    return console;
}

void populating_random_values(vector<int>& array) {

    int N = array.size();

    for (int i = 0; i < N; i++)
        array[i] = 1 + (rand() % 100);                     // feeding the array values between 1 and 1000 randomly

    return;
}