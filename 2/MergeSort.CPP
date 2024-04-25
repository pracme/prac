#include <bits/stdc++.h>
#include <omp.h>

using std :: chrono :: duration_cast;
using std :: chrono :: high_resolution_clock;
using std :: chrono :: milliseconds;

using namespace std;

/*
    <chrono> is a C++ header that provides a collection of types and functions to work with time.
*/

// utility functions, declaration

template <typename T> ostream& operator << (ostream& console, const vector<T>& array);
void initialize_array(vector<int>& array, int N);
void populating_random_values(vector<int>& array);

// Merge Sort Function Declaration

void merge(vector<int>& array, vector<int>& left_subarray, vector<int>& right_subarray, int left_array_size, int right_array_size);
void merge_sort(vector<int>& array);
void merge_sort_using_threads(vector<int>& array);


// function to analyze the performance
int analysis(std :: function<void()> function);

int main(void) {

    int N = 10;
    vector<int> array, temp;

    double execution_time = 0.0f;
    double execution_time_case1 = 0.0f, execution_time_case2 = 0.0f;
    double speed_up = 0.0f;

    bool flag = true;
    while (flag)
    {   
        // Menu driven program ...

        cout << "-------------------------- Menu -------------------------------"  << endl;
        cout << "1. Merge Sort  "                                                  << endl;
        cout << "2. Merge Sort Execution Via threads, parallel computing    "      << endl;
        cout << "3. Compare threaded and non threaded execution of Merge Sort  "   << endl;
        cout << "4. Exit "                                                         << endl;
        cout << "---------------------------------------------------------------"  << endl;

        int choice = -1;
        cout << "Enter the choice : ";
        cin >> choice;

        switch(choice)
        {
            case 1:
                
                initialize_array(array, N);

                cout << "Merge Sort " << endl;

                populating_random_values(array);

                cout << "Before Sorting (displaying the first 10 numbers ) " << array;
                
                // executing the MergeSort and measuring the time required to sort the array in ms.

                execution_time = analysis([&] {merge_sort(array); });

                cout << "After  Sorting (displaying the first 10 numbers ) " << array;

                // analysis
                cout << "Time taken to sort the array : " << execution_time << " ms" << endl;

                break;

            case 2:

                initialize_array(array, N);

                cout << "Merge Sort Execution Via threads, parallel computing  " << endl;

                populating_random_values(array);

                // executing the threaded version of Merge Sort and measuring the time required to sort the array in ms

                cout << "Before Sorting (displaying the first 10 numbers ) " << array;

                omp_set_num_threads(16);  // setting the thread for parallel execution of the program.

                execution_time = analysis([&] {merge_sort_using_threads(array); });

                cout << "After  Sorting (displaying the first 10 numbers ) " << array;

                // analysis
                cout << "Time taken to sort the array : " << execution_time << " ms" << endl;

                break;
            
            case 3:

                initialize_array(array, N);

                cout << "Compare threaded and non threaded execution of Merge Sort" << endl;
                populating_random_values(array);

                temp = array;                               // making copy of array


                execution_time_case1 = analysis([&] {merge_sort(array); });
                execution_time_case2 = analysis([&] {merge_sort_using_threads(temp); });
                
                speed_up = (float)(execution_time_case1 / (float)execution_time_case2);


                cout << "Merge Sort without multi threading                      : " << execution_time_case1 << " ms" << endl;
                cout << "Merge Sort Using Multi threading , parallel Computation : " << execution_time_case2 << " ms" << endl;
                cout << "Speed up                                                : " << speed_up << endl;

                break;

            case 4:

                flag = false;
                break;
            
            default:

                cout << "Invalid choice !!! " << endl;
                break;
        }
    }

    return 0;
}

int analysis(std :: function<void()> function) {

    // subtract end from start timepoints and cast it to required unit, nanoseconds, microseconds, milliseconds etc.
    auto start = high_resolution_clock::now();
    
    // Executing function
    function();
    auto end   = high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    return duration.count();
}

// Merge and Merge Sort Function

void merge(vector<int>& array, vector<int>& left_subarray, vector<int>& right_subarray, int left_array_size, int right_array_size) {

    int i = 0, j = 0, k = 0;

    while (i < left_array_size && j < right_array_size)
        array[k++] = left_subarray[i] < right_subarray[j] ? left_subarray[i++] : right_subarray[j++];

    while (i < left_array_size)
        array[k++] = left_subarray[i++];

    while (j < right_array_size)
        array[k++] = right_subarray[j++];
}

void merge_sort(vector<int>& array) {

    int N = array.size();

    if (N <= 1)
        return;

    int mid = N / 2;
    int left_array_size = mid;
    int right_array_size = N - mid;

    vector<int> left_subarray(left_array_size);
    vector<int> right_subarray(right_array_size);

    int j = 0;
    for (int i = 0; i < N; i++)
    {
        if (i < mid)
        {
            left_subarray[i] = array[i];
        }
        else {

            right_subarray[j] = array[i];
            j ++;
        }
    }

    merge_sort(left_subarray);
    merge_sort(right_subarray);

    merge(array, left_subarray, right_subarray, left_array_size, right_array_size);
}

void merge_sort_using_threads(vector<int>& array) {

    int N = array.size();

    if (N <= 1)
        return;

    int mid = N / 2;
    int left_array_size = mid;
    int right_array_size = N - mid;

    vector<int> left_subarray(left_array_size);
    vector<int> right_subarray(right_array_size);

    int j = 0;
    for (int i = 0; i < N; i++)
    {
        if (i < mid)
        {
            left_subarray[i] = array[i];
        }
        else {

            right_subarray[j] = array[i];
            j ++;
        }
    }

    // defining parallel sections for merge sort using threading.

    #pragma omp parallel sections num_threads(16)
    {
        #pragma omp section
        {
            //cout << "l section " << omp_get_thread_num() << endl;
            merge_sort(left_subarray);
        }
        #pragma omp section
        {
            //cout << "r section " << omp_get_thread_num() << endl;
            merge_sort(right_subarray);
        }
    }

    merge(array, left_subarray, right_subarray, left_array_size, right_array_size);
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
        array[i] = 1 + (rand() % 1000);                     // feeding the array values between 1 and 1000 randomly

    return;
}

void initialize_array(vector<int>& array, int N) {

    cout << "Enter the size of the array : ";
    cin  >> N;

    array.resize(max(10, N));

    return;
}