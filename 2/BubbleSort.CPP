#include <bits/stdc++.h>
#include <omp.h>

using std :: chrono :: duration_cast;
using std :: chrono :: high_resolution_clock;
using std :: chrono :: milliseconds;

/*
    <chrono> is a C++ header that provides a collection of types and functions to work with time.
*/

using namespace std;

// utility functions, declaration
template <typename T> ostream& operator << (ostream& console, const vector<T>& array);
void populating_random_values(vector<int>& array);
void initialize_array(vector<int>& array, int N);

// Function to compute the performance of sequential and parallel execution of Bubblesort
int analysis(std :: function<void()> function);

// sequential and parallel Bubblesort Functions declaration
void sequential_bubblesort(vector<int>& array);
void parallel_bubblesort(vector<int>& array);


int main(void) {

    int N = 0;                                               // define the size of input array
    vector<int> array, temp;              
    float speed_up = 0.0f;                                  // speed_up variable to compute the performance


    bool flag = true;
    while (flag)
    {   
        // Menu driven program ...

        cout << "-------------------------- Menu -------------------------------"  << endl;
        cout << "1. Sequential Bubble Sort  "                                      << endl;
        cout << "2. Parallel Bubble Sort    "                                      << endl;
        cout << "3. Compare Sequential and Parallel Bubble Sort  "                 << endl;
        cout << "4. Exit "                                                         << endl;
        cout << "---------------------------------------------------------------"  << endl;

        int sequential_bubblesort_time = 0, parallel_bubblesort_time = 0;
        // variable to measure the performance of both versions of Bubblesort algorithm.

        int choice = -1;
        cout << "Enter the choice : ";
        cin >> choice;

        switch(choice)
        {
            case 1:
                
                initialize_array(array, N);

                cout << "----- Sequential BubbleSort ----- " << endl;

                populating_random_values(array);

                cout << "Before Sorting (displaying the first 10 numbers ) " << array;
                
                // executing the sequential bubblesort and measuring the time required to sort the array in ms
                sequential_bubblesort_time = analysis([&] {sequential_bubblesort(array); });
                
                cout << "After  Sorting (displaying the first 10 numbers ) " << array;

                // analysis
                cout << "Time taken to sort the array : " << sequential_bubblesort_time << " ms" << endl;

                break;

            case 2:

                initialize_array(array, N);

                cout << "Parallel BubbleSort  " << endl;

                populating_random_values(array);

                // executing the parallel bubblesort and measuring the time required to sort the array in ms

                cout << "Before Sorting (displaying the first 10 numbers ) " << array;

                omp_set_num_threads(16);                     // setting the thread for parallel execution of the program.
                parallel_bubblesort_time = analysis([&] {parallel_bubblesort(array); });
                
                cout << "After  Sorting (displaying the first 10 numbers ) " << array;

                // analysis
                cout << "Time taken to sort the array : " << parallel_bubblesort_time << " ms" << endl;

                break;
            
            case 3:

                initialize_array(array, N);

                cout << "Comparing Sequential and Parallel BubbleSort : " << endl;
                populating_random_values(array);

                temp = array;                               // making copy of array

                sequential_bubblesort_time = analysis([&] {sequential_bubblesort(array); });
                
                omp_set_num_threads(16);
                parallel_bubblesort_time = analysis([&] {parallel_bubblesort(temp); });
                
                speed_up = (float) (sequential_bubblesort_time / (float)parallel_bubblesort_time);
                
                cout << "Sequential BubbleSort Time : " << sequential_bubblesort_time <<  " ms" << endl;
                cout << "Parallel   BubbleSort Time : " << parallel_bubblesort_time <<    " ms" << endl;
                cout << "Speed up : " << speed_up   << endl;               

                break;

            case 4:

                flag = false;
                break;
            
            default:

                cout << "Invalid choice !!! " << endl;
                break;
        }
    }

    return 0; // terminating program
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


// Time Complexity :- O(N^2)
// odd even method of sorting
void sequential_bubblesort(vector<int>& array) {

    int N = array.size();

    for (int i = 0; i < N; i++)
    {
        int first = i % 2;

        for (int j = first; j < N - 1; j += 2)
            if (array[j] > array[j + 1])
                swap(array[j], array[j + 1]);
    }

    return;
}


// Odd–even sort is a parallel version of bubble sort.

// Time Complexity : O(N + k) , N -> size of the array, k -> time taken to divide the algorithms in different threads for parallel execution
void parallel_bubblesort(vector<int>& array) {

    int N = array.size();

    for (int i = 0; i < N; i++)
    {
        int first = i % 2;

        // defining the shared data source for parallel execution.

        #pragma omp parallel for shared (array, first) num_threads(16)

        for (int j = first; j < N - 1; j += 2)
            if (array[j] > array[j + 1])
                swap(array[j], array[j + 1]);
    }

    return;
}

// utility functions

template <typename T> ostream& operator << (ostream& console, const vector<T>& array) {
    
    // operator overloading.
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