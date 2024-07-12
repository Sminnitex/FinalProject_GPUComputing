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
If you run it on the Unitrento cluster in the sparse.cu file on lines from 20 to 29, and on line 55 is necessary to change the relative path to the corrispondend absolute path

This project is the direct evolution of https://github.com/Sminnitex/Homework2_GPUComputing
Therefore the prerequisites are the same as the one described in that repository, and are:

>   python3 equipped with matplotlib, pandas and numpy
>   cuda and the cuda toolkit
>   gcc and cmake

# Abstract
A Sparse Matrix is a Matrix where most elements
are zeros, in my previous work I analyzed some basic techniques
of matrix transposition using the CPU and the GPU; but this
scenario opens to new possible horizons of efficiency. Knowing
our matrices will be composed almost everywhere by zeros we can
change our way of thinking the transposition trying to implement
a method that takes advantage of this information. To perform a
transposition everything we need to do is to change every column
value to the row value and vice-versa, however, in this case, the
resultant matrix won’t be sorted as we require. In fact we don’t
need to go through every value in memory, but we can just
change the coordinates of that values that aren’t zero, and by
knowing the dimension of the matrix we implicitly know that
every other value is indeed zero. In this paper we are going to
explore some state of the art methods found in other researches,
then we are going to discuss my method and make a comparison
with the cuSPARSE library by Nvidia and with other simpler
method to perform a normal matrix transpose with the GPU
developed during the homeworks of this semester; then we are
going to plot the results obtained and make some conclusions
about it.
# INTRODUCTION
In this project, we are going to analyze the problem of
sparse matrix transposition. A sparse matrix is a type of matrix
where most elements are zeros, and only some of them are non
zero values.
Considering the work developed in, where I explored
the efficiency of a classic matrix transposition algorithm,
the problem of sparse matrix transposition introduces unique
optimization opportunities. In a standard matrix, every element
must be processed and stored, leading to significant compu-
tational and memory overhead. However, in a sparse matrix,
the majority of elements do not need to be considered at all,
as they are zeros. So we can build specialized algorithms for
handling the non zero elements.
In fact traditional matrix transposition involves swapping
the rows and columns of a matrix. For a normal matrix, this
operation requires iterating over all elements, which can be
computationally expensive for large matrices.
This project aims to compare the performance of different
transposition algorithms, including the classic matrix transpo-
sition algorithm developed on and the optimized library
designed for sparse matrices by NVIDIA, cuSPARSE.
# STATE OF THE ART
One of the most efficient ways of overcoming this task
is to store our sparse matrix in a Compressed Row Storage
(CRS), and then perform a transposition passing to a Com-
pressed Column Storage (CCS); or as they are called in, Compressed Sparse Row (CSR) and Compressed Sparse
Column (CSC). CSR and CSC are two storage techniques to
represent sparse matrices, used also by the cuSPARSE library
to perform the transposition: basically it consist in dividing
our matrix in three vectors;
• Value Vector: of type dtype (int or float considering the
case of study), where we contain only the non zero values
of our sparse matrix;
• Pointer Vector: This array stores the pointer in the value
vector where each row or column, considering the format
chosen, starts. The length of this array is the number of
rows or columns in the matrix plus one;
• Index Vector: This array stores the index of the other
dimension, column or row, in order to triangulate the
position of the non zero value contained in the value
vector.
As we can understand we know that every element is zero
but the non zero values, for which we keep only a pointer
and an index that can be exchanged in order to perform the
transpose. This is how the cuSPARSE library perform the
transposes; therefore our work, considering the state of
the art, will be to start from a CSR compression format, and
pass to a CSC compression format to handle the transpose in
a comparable way to cuSPARSE, but also in what calls
one of the most efficient ways possible.

# EXPERIMENTS

Now let’s see some results considering the bandwidth as
our main performance measure. To calculate the bandwidth
I considered as the only data moved the Non zero elements,
therefore I didn’t multiplicate the numerator for the whole
rows and columns dimension as I did in, since the zero
elements in the process of transposition are ignored both in My
Kernel for Sparse Transposition and in the one of the library
cuSPARSE. On top of that, consider the zero elements
in the bandwidth calculation could lead to some misleading
results since in optimized algorithms as the one produced
by Nvidia it inflates the amount of data considered to be
moving, even thought the zeros aren’t really considered by
the technique proposed. Even in normal matrix transpose could
lead to some strange results given the possible errors in the
use of cache, but even thought I decided to consider only the
non zero elements to give a more fair comparison between the
techniques and compare the actual workload of all kernels.
The first plot at fig. 1 is divided in two parts, above there
is the bandwidth value calculated in respect to the Non Zero
values of each sparse matrix, and below there is the plot of Non
zero values for each benchmark matrix. For the plot above I
decided to put a confrontation between the cuSPARSE library
of NVIDIA, the global and shared memory kernels used on in order to confront simple transposition techniques with
kernel ad hoc for sparse matrices, and with the label ”MyS-
parse” my function to manage sparse matrix transposition.

![alt text](https://github.com/Sminnitex/FinalProject_GPUComputing/blob/master/figures/NonZeros.png?raw=true)

As we could expect, even considering the way I calculated
the bandwidth, the Bandwidth calculated in GB/s grows almost
linearly with the grows of non zero values. We can also see
on the plot below how ”bcsstk35”, the last benchmark matrix,
has way more non zero values than every other matrix.
Now in the second plot fig. 2 we will see the bandwidth
value in respect with the dimension of the matrices, so
considering also the zero values in the transposition, and below
another plot to confront the dimension of each matrix.
As we can see here, considering the bandwidth value in
GB/s as before, and the dimensions that has to be multiplied by
1e8, the bandwidth grows for each technique with dimensions,
but not quite as before. In fact we have a significant drop in
performance for matrices of dimensions between 1 ∗ 1e8 and
2 ∗ 1e8 such as ”bcsstk17”, ”linverse” and ”t2dah a” as can
be seen in the plot of reference below. That happens because
those are the sparse matrices that in respect with dimensions

![alt text](https://github.com/Sminnitex/FinalProject_GPUComputing/blob/master/figures/Dimensions.png?raw=true)

have an higher concentration of non zero values, as we can
see in fig. 1. In the more dense matrices we have a significant
improve in performance in the classic transpose techniques,
and a less important improve also for the sparse techniques.
Then we step to the last plot, where we analyze the
bandwidth in respect to each Matrix.

![alt text](https://github.com/Sminnitex/FinalProject_GPUComputing/blob/master/figures/perfile.png?raw=true)

Here we can see clearly how all four techniques follow a
general trend in performance in respect to the amount of data
they can move, with the two normal transposition techniques
that are comparable in performance in the same way the sparse
kernels are comparable. One thing that should also be noted,
is that if we try, for example, to consider in the calculation of
the bandwidth also the zero values we would have a significant
improvement in performances. In fact the effective bandwidth
value would easily overcame the theoretical maximum of
bandwidth of the A30, but it wouldn’t be a trustful measure;
since of course the sparse kernels aren’t even actually dealing
with those values but only moving the pointers and indexes.
# CONCLUSION
All considered we can say that bandwidth grows almost
linearly with the number of non-zero elements in the matrix.
Including zero elements in bandwidth calculations shows less
consistent performance growth, highlighting the efficiency
of sparse-specific techniques. Specialized sparse transposition
techniques (cuSPARSE and ”MySparse”) are more efficient
for sparse matrices, focusing on non-zero elements and out-
performing general transposition techniques in these cases.
The plots showed how the classic techniques have higher
bandwidth value but the reason for this difference is about the
complexity of the two functions, where the classic only per-
form a transpose the two methods perform multiple operations,
and in case of my function opening multiple kernels with dif-
ferent grid and block size. In fact performing the timer measure
only on the third kernel the one which actually perform the
transpose the results in bandwidth would be higher even of
the cuSPARSE library, but since even the function perform
multiple operations it wouldn’t make a fair comparison to
count only the last step of my function for the plot results.
However the two functions are perfectly comparable in each
step therefore I found the results acceptable.
