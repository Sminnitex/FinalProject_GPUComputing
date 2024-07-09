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
    //The paths of our benchmark matrices
    const char* path[] = {
        "./dataset/1138_bus/1138_bus.mtx",
        "./dataset/Maragal_3/Maragal_3.mtx",
        "./dataset/photogrammetry/photogrammetry.mtx",
        "./dataset/plbuckle/plbuckle.mtx",
        "./dataset/bcsstk17/bcsstk17.mtx",
        "./dataset/filter2D/filter2D.mtx",
        "./dataset/SiH4/SiH4.mtx",
        "./dataset/linverse/linverse.mtx",
        "./dataset/t2dah_a/t2dah_a.mtx",
        "./dataset/barrier2-10/barrier2-10.mtx"
    };

    //Stats of my problem
    srand(time(NULL));
    int array_length = sizeof(path) / sizeof(path[0]);  
    int blocksize = 16;
    int gridsize = 7;
    printf("==============================================================\n");
    printf("STATS OF MY PROBLEM\n");
    printf("block size = %d \n", blocksize);
    printf("grid size = %d \n", gridsize);
    dim3 block_size(blocksize, blocksize, 1);
    dim3 grid_size(gridsize, gridsize, 1);
    printf("%d: block_size = (%d, %d), grid_size = (%d, %d)\n", __LINE__, block_size.x, block_size.y, grid_size.x, grid_size.y);
    int sharedMemSize = sizeof(dtype) * block_size.x * block_size.y * 2;
    int *number, m, n, nnz;
    int nnz_counter;

    //Timer definitions
    TIMER_DEF;
    float times[NDEVICE] = {0};
    char filename[556];
    FILE *csvtime[array_length];

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

    for(int k = 0; k < array_length; k++){
        //Initialize all the stuff we need
        dtype *matrix = NULL;
        number = (int *)malloc(3 * sizeof(int));
        read_mtx(path[k], matrix, number);
        nnz_counter = 0;
        
        //Assign number of rows, columns and non zero elements
        m = number[0]; 
        n = number[1]; 
        nnz = number[2];  

        //Initialize kernel
        dummyKernel<<<grid_size, block_size>>>();
        checkCudaErrors(cudaGetLastError()); 
        checkCudaErrors(cudaDeviceSynchronize());

        //Cusparse handle and stream
        cusparseHandle_t handle;
        cusparseCreate(&handle);
        checkCudaErrors(cudaGetLastError()); 
        cudaStream_t stream;
        checkCudaErrors(cudaStreamCreate(&stream));


        //Allocate and initialize host memory
        int *h_csrRowPtr = (int *)malloc((m + 1) * sizeof(int));
        int *h_csrColInd = (int *)malloc(nnz * sizeof(int));
        dtype *h_csrVal = (dtype *)malloc(nnz * sizeof(dtype));

        if (h_csrRowPtr == NULL || h_csrColInd == NULL || h_csrVal == NULL) {
           fprintf(stderr, "Error allocating host memory\n");
           return 1;
        }

        //Allocate device memory
        int *d_csrRowPtr, *d_csrColInd, *d_cscRowInd, *d_cscColPtr;
        dtype *d_csrVal, *d_cscVal;
        checkCudaErrors(cudaMalloc((void **)&d_csrRowPtr, (m + 1) * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_csrColInd, nnz * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_csrVal, nnz * sizeof(dtype)));
        checkCudaErrors(cudaMalloc((void **)&d_cscRowInd, nnz * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_cscColPtr, (n + 1) * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_cscVal, nnz * sizeof(dtype)));

        //Assign values host memory
        h_csrRowPtr[0] = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (matrix[i + j * m] != 0) {
                    if (nnz_counter >= nnz) {
                        fprintf(stderr, "Error: nnz_counter (%d) exceeded nnz (%d) at i=%d, j=%d\n", nnz_counter, nnz, i, j);
                        exit(EXIT_FAILURE);
                    }
                    h_csrColInd[nnz_counter] = j;
                    h_csrVal[nnz_counter] = matrix[i + j * m]; 
                    nnz_counter++;
                }
            }
        if (i + 1 > m) {
            fprintf(stderr, "Error: i+1 (%d) exceeded m (%d)\n", i + 1, m);
            exit(EXIT_FAILURE);
        }
        h_csrRowPtr[i + 1] = nnz_counter;
    }
        
        //Copy data to device
        checkCudaErrors(cudaMemcpyAsync(d_csrRowPtr, h_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice, stream));
        checkCudaErrors(cudaMemcpyAsync(d_csrVal, h_csrVal, nnz * sizeof(dtype), cudaMemcpyHostToDevice, stream));
        
        //Perform sparse matrix transpose with cusparse
        void *buffer;
        cusparseBuffer(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                            d_cscVal, d_cscColPtr, d_cscRowInd, buffer);
        TIMER_START;
        cusparseTranspose(handle, m, n, nnz, d_csrVal, d_csrRowPtr, d_csrColInd, 
                            d_cscVal, d_cscColPtr, d_cscRowInd, buffer);
        TIMER_STOP;
        times[0] = TIMER_ELAPSED;
        
        //Copy the transposed matrix back to host
        checkCudaErrors(cudaMemcpy(h_csrRowPtr, d_cscColPtr, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_csrColInd, d_cscRowInd, nnz * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_csrVal, d_cscVal, nnz * sizeof(dtype), cudaMemcpyDeviceToHost));

        //Perform a normal matrix transpose with kernels from homework 2
        dtype *transpose = NULL, *transposeShared = NULL, *d_matrix = NULL;
        checkCudaErrors(cudaMallocManaged(&d_matrix, sizeof(dtype) * m * n ));
        checkCudaErrors(cudaMallocManaged(&transpose, sizeof(dtype) * m * n));
        checkCudaErrors(cudaMallocManaged(&transposeShared, sizeof(dtype) * m * n));
        checkCudaErrors(cudaMemcpy(d_matrix, matrix, sizeof(dtype) * m * n , cudaMemcpyHostToDevice));

        //Global matrix transpose
        TIMER_START;
        transposeGlobalMatrix<<<grid_size, block_size, sharedMemSize, stream>>>(d_matrix, transpose, m, n);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaStreamSynchronize(stream));
        TIMER_STOP;
        times[1] = TIMER_ELAPSED;

        //Shared matrix transpose
        TIMER_START;
        transposeSharedMatrix<<<grid_size, block_size, sharedMemSize, stream>>>(d_matrix, transposeShared, m, n);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaStreamSynchronize(stream));
        TIMER_STOP;
        times[2] = TIMER_ELAPSED;

        //Copy back to host for debug purposes
        dtype *h_transpose = NULL, *h_transposeShared = NULL;
        h_transpose = (dtype*)malloc(m * n * sizeof(dtype));
        h_transposeShared = (dtype*)malloc(m * n * sizeof(dtype));
        checkCudaErrors(cudaMemcpy(h_transpose, transpose, m * n * sizeof(dtype), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_transposeShared, transposeShared, m * n * sizeof(dtype), cudaMemcpyDeviceToHost));

        //Perform a transpose with kernel adapted for sparse matrices
        int *d_my_csrRowPtr, *d_my_csrColInd, *d_my_cscRowInd, *d_my_cscColPtr;
        dtype *d_my_csrVal, *d_my_cscVal;
        checkCudaErrors(cudaMalloc((void **)&d_my_csrRowPtr, (m + 1) * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_my_csrColInd, nnz * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_my_csrVal, nnz * sizeof(dtype)));
        checkCudaErrors(cudaMalloc((void **)&d_my_cscRowInd, nnz * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_my_cscColPtr, (n + 1) * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **)&d_my_cscVal, nnz * sizeof(dtype)));

        checkCudaErrors(cudaMemcpy(d_my_csrRowPtr, h_csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_my_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_my_csrVal, h_csrVal, nnz * sizeof(dtype), cudaMemcpyHostToDevice));
         
        TIMER_START;
        sparseMatrixTranspose(m, n, nnz, d_my_csrVal, d_my_csrRowPtr, d_csrColInd, d_my_cscVal, d_my_cscColPtr, d_my_cscRowInd, sharedMemSize, stream);
        TIMER_STOP;
        times[3] = TIMER_ELAPSED;

        //Print effective Bandwidth
        printf("==============================================================\n");
        printf("STATS of %s\n", path[k]);
        printf("Sparse Matrix Transpose With Cusparse Effective Bandwidth(GB/s): %f\n", (2 * m * n * sizeof(dtype)) / (1e9 * times[0]));
        printf("Global Matrix Transpose Effective Bandwidth(GB/s): %f\n", (2 * m * n * sizeof(dtype)) / (1e9 * times[1]));
        printf("Shared Matrix Transpose Effective Bandwidth(GB/s): %f\n", (2 * m * n * sizeof(dtype)) / (1e9 * times[2]));
        printf("My Sparse Matrix Transpose Effective Bandwidth(GB/s): %f\n", (2 * m * n * sizeof(dtype)) / (1e9 * times[3]));

        //Produce output files
        sprintf(filename, "output/Matrix%d.csv", k);
        csvtime[k] = fopen(filename, "w");
        if (csvtime[k] == NULL) {
            printf("Error opening file!\n");
            return 1;
        }
        printf("%c", filename[k]);
        fprintf(csvtime[k], "TheRowsAre,Legend,Cusparse,Global,Shared,MySparse\n");
        fprintf(csvtime[k], "Bandwidth,Rows,Columns,NonZeros\n");

        for (int i = 0; i < 4; i++) {
            fprintf(csvtime[k], "%f,%d,%d,%d\n", (2 * m * n * sizeof(dtype)) / (1e9 * times[i]), m, n, nnz);
        }

        fclose(csvtime[k]);

        //Lines for debug purposes
        //printMatrix(matrix, m, n, "Matrix");
        //printSparseMatrix(h_csrRowPtr, h_csrColInd, h_csrVal, n, nnz, "Cusparse Transposed Matrix");
        //printMatrix(h_transpose, m, n, "Transpose");
        //printMatrix(h_transposeShared, m, n, "Transpose Shared");
        //printDeviceData<<<1, 1, 0, stream>>>(d_cscColPtr, d_cscRowInd, d_csrVal, m, nnz);

        //Destroy everything
        checkCudaErrors(cudaFree(d_csrRowPtr));
        checkCudaErrors(cudaFree(d_csrColInd));
        checkCudaErrors(cudaFree(d_csrVal));
        checkCudaErrors(cudaFree(d_cscRowInd));
        checkCudaErrors(cudaFree(d_cscColPtr));
        checkCudaErrors(cudaFree(d_cscVal));
        checkCudaErrors(cudaFree(d_matrix));
        cusparseDestroy(handle);
        checkCudaErrors(cudaFree(transpose));
        checkCudaErrors(cudaFree(transposeShared));
        checkCudaErrors(cudaStreamDestroy(stream));

        checkCudaErrors(cudaFree(d_my_csrRowPtr));
        checkCudaErrors(cudaFree(d_my_csrColInd));
        checkCudaErrors(cudaFree(d_my_csrVal));
        checkCudaErrors(cudaFree(d_my_cscRowInd));
        checkCudaErrors(cudaFree(d_my_cscColPtr));
        checkCudaErrors(cudaFree(d_my_cscVal));

        free(h_csrRowPtr);
        free(h_csrColInd);
        free(h_csrVal);
        free(matrix);
        free(h_transpose);
        free(h_transposeShared);
        free(number);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceReset());
        
        //Reset times array
        for (int i = 0; i < NDEVICE; i++) {
            times[i] = 0.0;
        }
    }

    return 0;
}
