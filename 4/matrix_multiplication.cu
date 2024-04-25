#include <cstdio>
#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel code
__global__ void calc_prod_cuda(int* A, int* B, int* C, int rows_a, int cols_a, int rows_b, int cols_b) {
  // get row, column from block and therd index
  int g = blockIdx.x * blockDim.x + threadIdx.x;
  int col = g / rows_a, row = g % rows_a;
    
  // calcuate prod for a cell
  C[row * rows_b + col] = 0;
  for (int i = 0; i < cols_b; i++) {
    C[row * cols_b + col] += A[row * cols_a + i]*B[i * cols_b + col];
  }
}

// serial prouduct method
void calc_prod_serial(int * A, int* B, int* C, int rows_a, int cols_a, int rows_b, int cols_b) {
  // traverse rows
  for (int i=0; i < rows_a; i++) {
    // traverse column
    for (int j=0; j < cols_b; j++) {
      // calcuate prod for a cell
      C[i * cols_b +j] = 0;
      for (int k=0; k < cols_b; k++) {
        C[i * cols_b + j] += A[i * cols_a + k] * B[k * cols_b + j];
      }
    }
  }
}

void initialize_matrix(
  int *host_a, int *host_b, int *host_prod, // Host matrices
  int rows_a, int cols_a, // dimenstin of A 
  int rows_b, int cols_b // dimensions of B
) {
  printf("Initializing matrix..\n");

  //initialize A, B
  for (int i = 0; i < rows_a * cols_a; i++) {
    host_a[i] = i;
  }
  for (int i = 0; i < rows_b * cols_b; i++) {
    host_b[i] = i+i;
  }
  
  printf("Matrix initialized\n");
  fflush(stdout);
}

// function of print matrix
void display_matrix(int *matrix, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      printf("%d ", matrix[i * cols + j]);
    }
    printf("\n");
  }
}

// gpu matrix multiplication function
void calculate_cuda(
  int *host_a, int *host_b, int *host_prod, // Host matrices
  int rows_a, int cols_a, // dimenstin of A 
  int rows_b, int cols_b, // dimensions of B 
  int rows_prod, int cols_prod, // dimensions of prod 
  bool show_product
) {

 // initialize matrix on device
  int *device_a, *device_b, *device_prod;

  printf("\nCalculating PARALLEL..\n");

  // Allocate on device
  cudaMalloc((void**) &device_a, rows_a * cols_a * sizeof(int));
  cudaMalloc((void**) &device_b, rows_b * rows_b * sizeof(int));
  cudaMalloc((void**) &device_prod, rows_prod * cols_prod * sizeof(int));

  // Copy host to device
  cudaMemcpy(
    device_a, host_a,
    rows_a * rows_b * sizeof(int),
    cudaMemcpyHostToDevice
  );
  cudaMemcpy(
    device_b, host_b,
    rows_b * cols_b * sizeof(int),
    cudaMemcpyHostToDevice
  );

  // Define grid and block dimensions
  dim3 blockDim(cols_b);
  dim3 gridDim(rows_a);
    
  clock_t start_time = clock();

  // multiply
  calc_prod_cuda <<<gridDim, blockDim>>> (
    device_a, device_b, device_prod, 
    rows_a, cols_a, 
    rows_b, cols_b
  );

  // Copy the result back to the host
  cudaMemcpy(
    host_prod, device_prod,
    rows_prod * cols_prod * sizeof(int),
    cudaMemcpyDeviceToHost
  );

  if (show_product) {
    printf("\nProduct is:\n");
    display_matrix(host_prod, rows_prod, cols_prod);
  }
  
  printf(
    "\nProduct calculated in %f seconds\n",
    (double)(clock() - start_time) / CLOCKS_PER_SEC
  );
  fflush(stdout);
  
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_prod);
}

// serial matrix multiplication function
void calculate_serial(
  int *host_a, int *host_b, int *host_prod, // Host matrices
  int rows_a, int cols_a, // dimenstin of A 
  int rows_b, int cols_b, // dimensions of B 
  int rows_prod, int cols_prod, // dimensions of prod
  bool show_product
) {
  clock_t start_time = clock();
  printf("\nCalculating Serial..\n");

  calc_prod_serial(
    host_a, host_b, host_prod,
    rows_a, rows_b,
    rows_b, cols_b
  );
  if (show_product) {
    printf("\nProduct is:\n");
    display_matrix(host_prod, rows_prod, cols_prod);
  }
  printf(
    "\nProduct calculated in %f seconds\n",
    (double)(clock() - start_time) / CLOCKS_PER_SEC
  );
  fflush(stdout);
}

void free_matrix(int *host_a, int *host_b, int *host_prod) {
  // free memory
  free(host_a);
  free(host_b);
  free(host_prod);
}

int main() {
  int i=1;
  while (true) {
    if (i==1) {
      int rows_a, cols_a, rows_b, cols_b, see_prod;
      
      printf("\nEnter dimensions of Matrix: ");
      scanf("%d", &rows_a);

      cols_a =  cols_b = rows_b = rows_a;
     
      printf("\nDo you want to see prouct? ");
      scanf("%d", &see_prod);
      printf("\n");

      int *A, *B, *prod;
      
      // matrix size
      int rows_prod = rows_a;
      int cols_prod = cols_b;

      // allocate on host
      A = (int*) malloc (rows_a * cols_a * sizeof(int));
      B = (int*) malloc (rows_b * cols_b * sizeof(int));
      prod = (int*) malloc (rows_prod * cols_prod * sizeof(int));

      initialize_matrix(
        A, B, prod, 
        rows_a, cols_a, 
        rows_b, cols_b
      );
      calculate_cuda(
        A, B, prod, 
        rows_a, cols_a, 
        rows_b, cols_b, 
        rows_prod, cols_prod, 
        see_prod
      );
      calculate_serial(
        A, B, prod,
        rows_a, cols_a,
        rows_b, cols_b,
        rows_prod, cols_prod,
        see_prod
      );

      free_matrix(A, B, prod);
    } else {
      break;
    }
    printf("Enter 1 to calculate again? ");
    scanf("%d", &i);
  }
}
