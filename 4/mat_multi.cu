#include <iostream>
#include <chrono>
#include <iomanip>

#define TILE_WIDTH 16

__global__ void matrixMultiplication(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0;
        for (int i = 0; i < width; ++i) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

void sequentialMatrixMultiplication(float* A, float* B, float* C, int width) {
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            float sum = 0.0;
            for (int k = 0; k < width; ++k) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }
}

int main() {
    const int width = 3; // Define the width of square matrices

    float *h_A, *h_B, *h_C_seq, *h_C_cuda; // Host matrices
    float *d_A, *d_B, *d_C; // Device matrices

std::cout << "\n***** [ HPC ASSIGNMENT 4 ] :- MATRIX MULTIPLICATION  ***** \n"<<std::endl;
    // Allocate memory for host matrices
    h_A = (float*)malloc(width * width * sizeof(float));
    h_B = (float*)malloc(width * width * sizeof(float));
    h_C_seq = (float*)malloc(width * width * sizeof(float));
    h_C_cuda = (float*)malloc(width * width * sizeof(float));

    // Initialize input matrices h_A and h_B
    for (int i = 0; i < width * width; ++i) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Print input matrices h_A and h_B
    std::cout << "--> Matrix A [ ] : \n" << std::endl;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout  <<"     "<< h_A[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n--> Matrix B [ ] : \n" << std::endl;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout <<"     " << h_B[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Allocate memory for device matrices
    cudaMalloc(&d_A, width * width * sizeof(float)); int a=100000;
    cudaMalloc(&d_B, width * width * sizeof(float));
    cudaMalloc(&d_C, width * width * sizeof(float));

    // Copy host matrices h_A and h_B to device matrices d_A and d_B
    cudaMemcpy(d_A, h_A, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, width * width * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block size
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x, (width + dimBlock.y - 1) / dimBlock.y);

    // Launch the kernel for matrix multiplication (CUDA)
    auto start_cuda = std::chrono::steady_clock::now();
    matrixMultiplication<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    cudaDeviceSynchronize(); // Ensure kernel execution is complete before copying back
    auto end_cuda = std::chrono::steady_clock::now();
    std::chrono::duration<double> cuda_time = end_cuda - start_cuda;

    // Copy the result back to host matrix h_C_cuda
    cudaMemcpy(h_C_cuda, d_C, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    // Sequential Matrix Multiplication
    auto start_sequential = std::chrono::steady_clock::now();
    sequentialMatrixMultiplication(h_A, h_B, h_C_seq, width);
    auto end_sequential = std::chrono::steady_clock::now();
    std::chrono::duration<double> sequential_time = end_sequential - start_sequential;

    // Display a part of the result (for verification)
    std::cout << "\n--> Resultant Matrix C (Sequential): \n" << std::endl;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout <<"     " << h_C_seq[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
std::cout << "\n--> Time taken for sequential multiplication: " << std::fixed << std::setprecision(9) << sequential_time.count()*a<< " secs." << std::endl;
    std::cout << "\n--> Resultant Matrix C (CUDA): \n" << std::endl;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            std::cout <<"     "<< std::fixed << std::setprecision(0) << h_C_cuda[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C_seq);
    free(h_C_cuda);


    std::cout << "\n--> Time taken for CUDA multiplication: "<< std::fixed << std::setprecision(9) << cuda_time.count() << " secs." << std::endl;

    return 0;
}
