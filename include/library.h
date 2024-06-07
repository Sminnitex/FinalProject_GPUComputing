#ifndef LIBRARY_H
#define LIBRARY_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "./helper_cuda.h"

#define dtype float
#define TILE_DIM 8
#define BLOCK_ROWS 2

__global__ void dummyKernel() {}

void printMatrix(dtype *matrix, int rows, int cols, char ST[56]){
    int i, j;  \
      printf("%s:\n", ( ST ));  \
      for (i=0; i< ( rows ); i++) {  \
        printf("\t");  \
        for (j=0; j< ( cols ); j++)  \
          printf("%6.3f ", matrix[i*( cols ) + j]);  \
        printf("\n");  \
      }  \
      printf("\n\n");  \
}


#endif  