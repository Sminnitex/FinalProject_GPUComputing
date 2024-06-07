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
    int power = strtol(argv[1], NULL, 10);
    long number = pow(2, power);
    int blocksize = 32;
    int gridsize = (number + blocksize - 1) / 32;

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


    //Lines for debug purposes
    //printMatrix(matrix, number, number, "Matrix");
    //printMatrix(transpose, number, number, "My Sparse Matrix Transpose");
    //printMatrix(transposeCuda, number, number, "Cuda Sparse Matrix Transpose");

    //Print effective Bandwidth
    printf("==============================================================\n");
    printf("STATS\n");
    printf("My Sparse Matrix Transpose Effective Bandwidth(GB/s): %f\n", (2 * number * number * sizeof(dtype)) / (1e9 * times[1]));
    printf("Cuda Sparse Matrix Transpose Effective Bandwidth(GB/s): %f\n", (2 * number * number * sizeof(dtype)) / (1e9 * times[0]));
    
    //Destroy everything
    cusparseDestroy(handle);

    return 0;
}
