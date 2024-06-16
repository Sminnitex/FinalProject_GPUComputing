#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>
#include "./include/library.h"
#include "./include/helper_cuda.h"
#include <cuda_runtime.h>
#include <cusparse.h>

#define NDEVICE 4
#define TIMER_DEF     struct timeval temp_1, temp_2
#define TIMER_START   gettimeofday(&temp_1, (struct timezone*)0)
#define TIMER_STOP    gettimeofday(&temp_2, (struct timezone*)0)
#define TIMER_ELAPSED ((temp_2.tv_sec-temp_1.tv_sec)+(temp_2.tv_usec-temp_1.tv_usec)/1000000.0)

int main(int argc, char *argv[]) {
    //Initialize all the stuff we need
    srand(time(NULL));
    dtype *matrix = NULL;
    int number[3];
    read_mtx("./dataset/1138_bus/1138_bus.mtx", matrix, number);
    int blocksize = 32;
    int gridsize = 28;

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
    checkCudaErrors(cudaGetLastError());

    //Assign number of rows, columns and non zero elements
    int m = number[0]; 
    int n = number[1]; 
    int nnz = number[2];  

    //Allocate and initialize host memory
    int *h_csrRowPtr = (int *)malloc((m + 1) * sizeof(int));
    int *h_csrColInd = (int *)malloc(nnz * sizeof(int));
    float *h_csrVal = (float *)malloc(nnz * sizeof(float));

    //Allocate device memory
    int *d_csrRowPtr, *d_csrColInd, *d_cscRowInd, *d_cscColPtr;
    float *d_csrVal, *d_cscVal;
    checkCudaErrors(cudaMalloc((void **)&d_csrRowPtr, (m + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_csrVal, nnz * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&d_cscRowInd, nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_cscColPtr, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&d_cscVal, nnz * sizeof(float)));

    //Assign values host memory
    h_csrRowPtr[0] = 0;
    int nnz_counter = 0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (matrix[i + j * m] != 0) {
                h_csrColInd[nnz_counter] = j;
                h_csrVal[nnz_counter] = matrix[i + j * m];
                nnz_counter++;
            }
        }
        h_csrRowPtr[i + 1] = nnz_counter;
    }

    //Copy data to device
    checkCudaErrors(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice));

    //Perform sparse matrix transpose with cusparse
    size_t bufferSize;
    void *buffer;
    TIMER_START;
    cusparseTransposeBuffer(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                           d_cscVal, d_cscColPtr, d_cscRowInd, bufferSize, buffer);
    TIMER_STOP;
    times[0] = TIMER_ELAPSED;

    TIMER_START;
    cusparseTranspose(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                      d_cscVal, d_cscColPtr, d_cscRowInd, buffer);
    TIMER_STOP;
    times[1] = TIMER_ELAPSED;

    //Perform a normal matrix transpose with kernels from homework 2
    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));
    dtype *transpose = NULL, *transposeShared = NULL;
    checkCudaErrors(cudaMallocManaged(&transpose, sizeof(dtype) * m * n));
    checkCudaErrors(cudaMallocManaged(&transposeShared, sizeof(dtype) * m * n));

    TIMER_START;
    transposeGlobalMatrix<<<grid_size, block_size, sharedMemSize, stream>>>(matrix, transpose, m, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    TIMER_STOP;
    times[2] = TIMER_ELAPSED;

    TIMER_START;
    transposeSharedMatrix<<<grid_size, block_size, sharedMemSize, stream>>>(matrix, transposeShared, m, n);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    TIMER_STOP;
    times[3] = TIMER_ELAPSED;

    //Print effective Bandwidth
    printf("==============================================================\n");
    printf("STATS\n");
    printf("Sparse Matrix Transpose With Cusparse Buffer Effective Bandwidth(GB/s): %f\n", (2 * number[0] * number[1] * sizeof(dtype)) / (1e9 * times[0]));
    printf("Sparse Matrix Transpose With Cusparse Effective Bandwidth(GB/s): %f\n", (2 * number[0] * number[1] * sizeof(dtype)) / (1e9 * times[1]));
    printf("Global Matrix Transpose Effective Bandwidth(GB/s): %f\n", (2 * number[0] * number[1] * sizeof(dtype)) / (1e9 * times[2]));
    printf("Shared Matrix Transpose Effective Bandwidth(GB/s): %f\n", (2 * number[0] * number[1] * sizeof(dtype)) / (1e9 * times[3]));

    //Destroy everything
    checkCudaErrors(cudaFree(buffer));
    checkCudaErrors(cudaFree(d_csrRowPtr));
    checkCudaErrors(cudaFree(d_csrColInd));
    checkCudaErrors(cudaFree(d_csrVal));
    checkCudaErrors(cudaFree(d_cscRowInd));
    checkCudaErrors(cudaFree(d_cscColPtr));
    checkCudaErrors(cudaFree(d_cscVal));

    cusparseDestroy(handle);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(transpose));
    checkCudaErrors(cudaFree(transposeShared));

    free(h_csrRowPtr);
    free(h_csrColInd);
    free(h_csrVal);
    free(matrix);

    return 0;
}