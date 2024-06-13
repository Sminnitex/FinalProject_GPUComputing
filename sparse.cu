#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "./include/library.h"
#include "./include/helper_cuda.h"
#include <cuda_runtime.h>
#include <cusparse.h>

#define NDEVICE 2
#define TIMER_DEF     struct timeval temp_1, temp_2
#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)
#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)
#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

int main(int argc, char *argv[]) {
    //Initialize all the stuff we need
    srand(time(NULL));
    dtype *matrix;
    int number[3];
    read_mtx("./dataset/1138_bus/1138_bus.mtx", matrix, number);
    int blocksize = 32;
    int gridsize = (number[0] + blocksize - 1) / 32;

    printf("==============================================================\n");
    printf("STATS OF MY PROBLEM\n");
    printf("block size = %d \n", blocksize);
    printf("grid size = %d \n", gridsize);
    dim3 block_size(blocksize, blocksize, 1);
    dim3 grid_size(gridsize, gridsize, 1);
    printf("%d: block_size = (%d, %d), grid_size = (%d, %d)\n", __LINE__, block_size.x, block_size.y, grid_size.x, grid_size.y);
    int sharedMemSize = sizeof(dtype) * block_size.x * block_size.y * 2;

    //Print device properties for unitn cluster
    FILE *file = fopen("warp.txt", "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }
    
    printf("==============================================================\n");
    printf("DEVICE PROPERTIES\n");
    char ch;
    while ((ch = fgetc(file)) != EOF) {
        printf("%c", ch);
    }
    fclose(file);

    //Initialize kernel and timer
    dummyKernel<<<grid_size, block_size>>>();
    TIMER_DEF;
    float times[NDEVICE] = {0};
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Example sparse matrix in CSR format
    int m = number[0]; // number of rows
    int n = number[1]; // number of columns
    int nnz = number[2];  // number of non-zero elements

    // Allocate and initialize host memory
    int *h_csrRowPtr = (int *)malloc((m + 1) * sizeof(int));
    int *h_csrColInd = (int *)malloc(nnz * sizeof(int));
    float *h_csrVal = (float *)malloc(nnz * sizeof(float));

    // Allocate device memory
    int *d_csrRowPtr, *d_csrColInd, *d_cscRowInd, *d_cscColPtr;
    float *d_csrVal, *d_cscVal;
    cudaMalloc((void **)&d_csrRowPtr, (m + 1) * sizeof(int));
    cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int));
    cudaMalloc((void **)&d_csrVal, nnz * sizeof(float));
    cudaMalloc((void **)&d_cscRowInd, nnz * sizeof(int));
    cudaMalloc((void **)&d_cscColPtr, (n + 1) * sizeof(int));
    cudaMalloc((void **)&d_cscVal, nnz * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice);

    // Perform sparse matrix transpose
    cusparseStatus_t status;
    size_t bufferSize;
    void *buffer;
    status = cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                                           d_cscVal, d_cscColPtr, d_cscRowInd, 
                                           CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, 
                                           CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Buffer size calculation failed\n");
        return 1;
    }
    cudaMalloc(&buffer, bufferSize);

    status = cusparseCsr2cscEx2(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                                d_cscVal, d_cscColPtr, d_cscRowInd, 
                                CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, 
                                CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("Matrix transpose failed\n");
        return 1;
    }

    //Print effective Bandwidth
    printf("==============================================================\n");
    printf("STATS\n");
    printf("My Sparse Matrix Transpose Effective Bandwidth(GB/s): %f\n", (2 * number[0] * number[1] * sizeof(dtype)) / (1e9 * times[1]));
    printf("Cuda Sparse Matrix Transpose Effective Bandwidth(GB/s): %f\n", (2 * number[0] * number[1] * sizeof(dtype)) / (1e9 * times[0]));
    
    //Destroy everything
    cudaFree(buffer);
    cudaFree(d_csrRowPtr);
    cudaFree(d_csrColInd);
    cudaFree(d_csrVal);
    cudaFree(d_cscRowInd);
    cudaFree(d_cscColPtr);
    cudaFree(d_cscVal);

    cusparseDestroy(handle);

    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);

    return 0;
}
