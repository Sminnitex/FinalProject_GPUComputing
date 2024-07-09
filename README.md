# 2.1.2 Sparse Matrix Transposition
This project requires to design an efficient algorithm to transpose a sparse matrix. Specifically the
matrix should be highly sparse, namely the number of zero element is more that 75% of the whole
(n × n) elements. The implementation should emphasize:
• storage format for storing sparse matrices (for example, compressed sparse row);
• the implementation to perform the transposition;
• a comparison against vendors’ library (e.g., cuSPARSE);
• dataset for the benchmark (compare all the implementation presented by selecting at least 10
matrices from suite sparse matrix collection https://sparse.tamu.edu/);
As usual, the metric to consider is the effective bandwidth.

#   To run the code
Is sufficient to run the following commands on the repo folder
``` 
make
./bin/final
``` 
And to check the plots on python
``` 
python plot.py
``` 

This project is the direct evolution of https://github.com/Sminnitex/Homework2_GPUComputing
Therefore the prerequisites are the same as the one described in that repository, and are:

>   python3 equipped with matplotlib, pandas and numpy
>   cuda and the cuda toolkit
>   gcc and cmake

# Attention
In the Dataset folder, the Barrier2 Matrix is available both unzipped and zipped. Since the unzipped version is very large and GitHub might not have uploaded it correctly, please unzip the matrix from the tar file again if necessary