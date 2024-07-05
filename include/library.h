#ifndef LIBRARY_H
#define LIBRARY_H

#include <stdio.h>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "./helper_cuda.h"
#include <fstream>
#include <algorithm>
#include <cusparse.h>

#define dtype float
#define TILE_DIM 8
#define BLOCK_ROWS 2

__global__ void dummyKernel() {}

void printMatrix(dtype *matrix, int rows, int cols, const char* ST){
    int i, j;
    printf("%s:\n", ST);
    for (i = 0; i < rows; i++) {
        printf("\t");
        for (j = 0; j < cols; j++) {
            printf("%6.3f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n\n");
}

int read_mtx(const char* path, dtype*& matrix, int number[3]) {
    std::ifstream fin(path);
    if (!fin.is_open()) {
        fprintf(stderr, "Error opening file %s.\n", path);
        return -1;
    }

    int M, N, L;

    // Ignore headers and comments:
    while (fin.peek() == '%') fin.ignore(2048, '\n');

    // Read defining parameters:
    fin >> M >> N >> L;
    matrix = (dtype*)malloc(M * N * sizeof(dtype));
    if (!matrix) {
        fprintf(stderr, "Error allocating memory for matrix.\n");
        fin.close();
        return -1;
    }
    std::fill(matrix, matrix + M * N, 0.0f);

    // Read the data
    for (int l = 0; l < L; l++) {
        int m, n;
        double data;
        fin >> m >> n >> data;
        matrix[(m - 1) + (n - 1) * M] = static_cast<dtype>(data);
    }

    fin.close();
    number[0] = M;
    number[1] = N;
    number[2] = L;

    return 0;
}

void cusparseTranspose(cusparseHandle_t handle, int m, int n, int nnz, 
                             const float *d_csrVal, const int *d_csrRowPtr, const int *d_csrColInd, 
                             float *d_cscVal, int *d_cscColPtr, int *d_cscRowInd, void*& buffer) {
    size_t bufferSize;
    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                                  d_cscVal, d_cscColPtr, d_cscRowInd, 
                                  CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, 
                                  CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "CUSPARSE Error: %s\n", cusparseGetErrorString(status));
        exit(EXIT_FAILURE);
    }
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));
    
    status = cusparseCsr2cscEx2(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                    d_cscVal, d_cscColPtr, d_cscRowInd, 
                    CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, 
                    CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "CUSPARSE Error: %s\n", cusparseGetErrorString(status));
        checkCudaErrors(cudaFree(buffer));  // Free the buffer before exiting
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(buffer));
}

__global__ void transposeGlobalMatrix(dtype *matrix, dtype *transpose, int rows, int cols) {
    const uint x = blockIdx.x * TILE_DIM + threadIdx.x;
    const uint y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if ((x < cols) && (y + i < rows)) {
            transpose[(x * rows) + (y + i)] = matrix[(y + i) * cols + x];
        }
    }
}

__global__ void transposeSharedMatrix(dtype *matrix, dtype *transpose, int rows, int cols) {
    __shared__ dtype tile[TILE_DIM][TILE_DIM + 1]; // Add padding to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = x + y * cols;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < cols && (y + i) < rows) {
            tile[threadIdx.y + i][threadIdx.x] = matrix[index_in + i * cols];
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = x + y * rows;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (x < rows && (y + i) < cols) {
            transpose[index_out + i * rows] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

void printSparseMatrix(int *cscRowPtr, int *cscColInd, dtype *csrVal, int n, int nnz, const char* ST) {
    printf("%s (CSC Format):\n", ST);
    printf("Row Pointers: ");
    for (int i = 0; i <= n; i++) {
        printf("%d ", cscRowPtr[i]);
    }
    printf("\nCol Indices: ");
    for (int i = 0; i < nnz; i++) {
        printf("%d ", cscColInd[i]);
    }
    printf("\nValues: ");
    for (int i = 0; i < nnz; i++) {
        printf("%6.3f ", csrVal[i]);
    }
    printf("\n\n");
}

__global__ void countNnzPerColumn(int nnz, int *d_csrColInd, int *d_nnzPerCol) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < nnz) {
        atomicAdd(&d_nnzPerCol[d_csrColInd[tid]], 1);
    }
}

__global__ void fillCscColPtr(int n, int *d_nnzPerCol, int *d_cscColPtr) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0) {
        d_cscColPtr[0] = 0;
        for (int i = 0; i < n; i++) {
            d_cscColPtr[i+1] = d_cscColPtr[i] + d_nnzPerCol[i];
        }
    }
}

__global__ void fillCscValAndRowInd(int m, int n, int nnz, dtype *d_csrVal, int *d_csrRowPtr, 
                                    int *d_csrColInd, dtype *d_cscVal, int *d_cscColPtr, int *d_cscRowInd) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < m) {
        for (int i = d_csrRowPtr[tid]; i < d_csrRowPtr[tid]; i++) {
            int col = d_csrColInd[i];
            int dest = atomicAdd(&d_cscColPtr[col], 1);
            d_cscVal[dest] = d_csrVal[i];
            d_cscRowInd[dest] = tid;
        }
    }
}

void sparseMatrixTranspose(int m, int n, int nnz, dtype *d_csrVal, int *d_csrRowPtr, 
                            int *d_csrColInd, dtype *d_cscVal, int *d_cscColPtr, int *d_cscRowInd,
                            int sharedMemory, cudaStream_t stream){
    //We need to go from a csr matrix format to a csc matrix format
    //Allocate temporary device memory
    int *d_nnzPerCol;
    checkCudaErrors(cudaMalloc((void**)&d_nnzPerCol, n * sizeof(int)));
    checkCudaErrors(cudaMemset(d_nnzPerCol, 0, n * sizeof(int)));

    //Launch kernel to count the non-zero elements per column for the transposed matrix
    int blockSize = 256;
    int gridSize = (nnz + blockSize - 1) / blockSize;
    countNnzPerColumn<<<gridSize, blockSize, sharedMemory, stream>>>(nnz, d_csrColInd, d_nnzPerCol);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    //Launch kernel to fill the column pointers for the CSC format
    gridSize = 1;  //Only one block needed to fill the column pointers
    fillCscColPtr<<<gridSize, blockSize, sharedMemory, stream>>>(n, d_nnzPerCol, d_cscColPtr);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //Copy column pointers back and adjust them to be exclusive scan
    checkCudaErrors(cudaMemcpy(d_nnzPerCol, d_cscColPtr, n * sizeof(int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_cscColPtr, d_nnzPerCol, n * sizeof(int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_cscColPtr + 1, d_nnzPerCol, n * sizeof(int), cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemset(d_cscColPtr, 0, sizeof(int)));

    //Launch kernel to fill the CSC values and row indices
    gridSize = (m + blockSize - 1) / blockSize;
    fillCscValAndRowInd<<<gridSize, blockSize, sharedMemory, stream>>>(m, n, nnz, d_csrVal, d_csrRowPtr, 
                                                 d_csrColInd, d_cscVal, d_cscColPtr, d_cscRowInd);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //Free temporary device memory
    checkCudaErrors(cudaFree(d_nnzPerCol));
}

#endif