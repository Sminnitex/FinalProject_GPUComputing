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

void read_mtx(char path[100], dtype *matrix, int number[2]){
  std::ifstream fin(path);
  int M, N, L;

  //Ignore headers and comments:
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  //Read defining parameters:
  fin >> M >> N >> L;
  matrix = new dtype[M*N];	     
  std::fill(matrix, matrix + M*N, 0.); 

  //Read the data
  for (int l = 0; l < L; l++){
    int m, n;
    double data;
    fin >> m >> n >> data;
    matrix[(m-1) + (n-1)*M] = data;
  }

  fin.close();
  number[0] = M;
  number[1] = N;
  number[2] = L;

  //Line for debug purposes
  //printMatrix(matrix, M, N, "Matrix");
}


#endif  