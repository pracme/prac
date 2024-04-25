#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <iostream>
#include <time.h>

using namespace std;

__global__ void add(int* A, int* B, int* C, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        C[tid] = A[tid] + B[tid];
    }
}

void add_serial(int *A, int *B, int*C, int size) {
  for (int i=0; i<size; i++) {
    C[i] = A[i] + B[i];
  }
}


void initialize(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        vector[i] = rand() % 10;
    }
}

void print(int* vector, int size) {
    for (int i = 0; i < size; i++) {
        cout << vector[i] << " ";
    }
    cout << endl;
}

int main() {
  int i = 1;
  while (i == 1) {
    int N = 4;
    int* A, * B, * C;

    int vectorSize;
    cout << "\nEnter size of Vector: ";
    cin >> vectorSize;
    size_t vectorBytes = vectorSize * sizeof(int);

    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];

    bool shoulprint;
    cout << "\nDisplay Vectors? ";
    cin >> shoulprint;

    initialize(A, vectorSize);
    initialize(B, vectorSize);

    if (shoulprint) {
      cout << "\nVector A: ";
      print(A, N);
      cout << "Vector B: ";
      print(B, N);
    }

    cout << "\nCalculating Parallel..\n";
    clock_t start_time = clock();

    int* X, * Y, * Z;
    cudaMalloc(&X, vectorBytes);
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, N);

    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);

    if (shoulprint) {
      cout << "Addition: ";
      print(C, N);
    }
    cout << "Time taken: " << (double) (clock() - start_time) / CLOCKS_PER_SEC << "\n\n";

    cout << "Calculating Serial..\n";

    start_time = clock();
    add_serial(A, B, C, vectorSize);

    if (shoulprint) {
      cout << "Addition: ";
      print(C, vectorSize);
    }
    cout << "Time taken: " << (double) (clock() - start_time) / CLOCKS_PER_SEC << "\n\n";


    delete[] A;
    delete[] B;
    delete[] C;

    cudaFree(X);
    cudaFree(Y);
    cudaFree(Z);

    cout << "Enter 1 to go again: ";
    cin >> i;
  }
  return 0;
}
