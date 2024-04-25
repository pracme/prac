#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <omp.h>

using namespace std;

// Sequential Bubble Sort
void bubbleSort(vector<int> &arr)
{
  int n = arr.size();
  for (int i = 0; i < n - 1; ++i)
  {
    for (int j = 0; j < n - i - 1; ++j)
    {
      if (arr[j] > arr[j + 1])
      {
        swap(arr[j], arr[j + 1]);
      }
    }
  }
}

// Parallel Bubble Sort using OpenMP
void parallelBubbleSort(vector<int> &arr)
{
  int n = arr.size();
  for (int i = 0; i < n - 1; ++i)
  {
#pragma omp parallel for
    for (int j = 0; j < n - i - 1; ++j)
    {
      if (arr[j] > arr[j + 1])
      {
        swap(arr[j], arr[j + 1]);
      }
    }
  }
}

// Merge function for Merge Sort
void merge(vector<int> &arr, int left, int mid, int right)
{
  int n1 = mid - left + 1;
  int n2 = right - mid;

  vector<int> L(n1), R(n2);

  for (int i = 0; i < n1; ++i)
    L[i] = arr[left + i];
  for (int j = 0; j < n2; ++j)
    R[j] = arr[mid + 1 + j];

  int i = 0, j = 0, k = left;

  while (i < n1 && j < n2)
  {
    if (L[i] <= R[j])
    {
      arr[k] = L[i];
      ++i;
    }
    else
    {
      arr[k] = R[j];
      ++j;
    }
    ++k;
  }

  while (i < n1)
  {
    arr[k] = L[i];
    ++i;
    ++k;
  }

  while (j < n2)
  {
    arr[k] = R[j];
    ++j;
    ++k;
  }
}

// Sequential Merge Sort
void mergeSort(vector<int> &arr, int left, int right)
{
  if (left < right)
  {
    int mid = left + (right - left) / 2;

    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);

    merge(arr, left, mid, right);
  }
}

// Parallel Merge Sort using OpenMP
void parallelMergeSort(vector<int> &arr, int left, int right)
{
  if (left < right)
  {
    int mid = left + (right - left) / 2;

#pragma omp parallel sections
    {
#pragma omp section
      parallelMergeSort(arr, left, mid);

#pragma omp section
      parallelMergeSort(arr, mid + 1, right);
    }
    merge(arr, left, mid, right);
  }
}

// Function to generate a random vector
vector<int> generateRandomVector(int size)
{
  vector<int> arr(size);
  srand(time(nullptr));
  for (int i = 0; i < size; ++i)
  {
    arr[i] = rand() % size;
  }
  return arr;
}

void displayArray(const vector<int> &arr)
{
  cout << "\n-> Generated Numbers Array:\n\n[ ";
  for (int num : arr)
  {
    cout << num << " ";
  }
  cout << "]\n\n"
       << endl;
}

// Function to print the sorted array
void printSortedArray(const vector<int> &arr)
{
  cout << "-> Sorted Array:\n\n[ ";
  for (int num : arr)
  {
    cout << num << " ";
  }
  cout << "]\n\n"
       << endl;
}

int main()
{
  cout << "\n ***** [ HPC ASSIGNMENT 2 ] ***** \n --------------------------------\n\n-> Generating Random Numbers......\n\n-> Enter Total Numbers to Generate/Sort: ";

  int size = 0;
  cin >> size;
  vector<int> arr = generateRandomVector(size);
  vector<int> arrCopy = arr;
  displayArray(arr); // Display generated numbers
  double startTime, endTime;

  cout << "-> Sorting " << size << " Elements...\n"
       << endl;

  // Sequential Bubble Sort
  startTime = omp_get_wtime();
  bubbleSort(arr);
  endTime = omp_get_wtime();
  cout << "-> Sequential Bubble Sort Time: " << endTime - startTime << " seconds\n"
       << endl;

  // Parallel Bubble Sort
  startTime = omp_get_wtime();

  endTime = omp_get_wtime();
  cout << "-> Parallel Bubble Sort Time: " << endTime - startTime << " seconds\n"
       << endl;

  // Reset array for merge sort
  arr = arrCopy;

  // Sequential Merge Sort
  startTime = omp_get_wtime();
  mergeSort(arr, 0, size - 1);
  endTime = omp_get_wtime();
  cout << "-> Sequential Merge Sort Time: " << endTime - startTime << " seconds\n"
       << endl;

  // Parallel Merge Sort
  startTime = omp_get_wtime();
  parallelMergeSort(arrCopy, 0, size - 1);
  endTime = omp_get_wtime();
  cout << "-> Parallel Merge Sort Time: " << endTime - startTime << " seconds\n"
       << endl;

  printSortedArray(arr);

  return 0;
}
