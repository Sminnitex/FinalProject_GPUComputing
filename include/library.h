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

void cusparseTransposeBuffer(cusparseHandle_t handle, int m, int n, int nnz, 
                             const float *d_csrVal, const int *d_csrRowPtr, const int *d_csrColInd, 
                             float *d_cscVal, int *d_cscColPtr, int *d_cscRowInd, 
                             size_t& bufferSize, void*& buffer) {
    cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                                  d_cscVal, d_cscColPtr, d_cscRowInd, 
                                  CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, 
                                  CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMalloc(&buffer, bufferSize));
}

void cusparseTranspose(cusparseHandle_t handle, int m, int n, int nnz, 
                       const float *d_csrVal, const int *d_csrRowPtr, const int *d_csrColInd, 
                       float *d_cscVal, int *d_cscColPtr, int *d_cscRowInd, 
                       void *buffer) {
    cusparseCsr2cscEx2(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                       d_cscVal, d_cscColPtr, d_cscRowInd, 
                       CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, 
                       CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
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

#endif