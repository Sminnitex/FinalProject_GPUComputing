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