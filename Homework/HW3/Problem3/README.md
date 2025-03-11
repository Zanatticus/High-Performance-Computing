# Problem 3 (20 Points)

In this problem, you will modify the `matmul.c` program provided, optimizing the execution of the matrix multiplication with first a dense matrix, and second with a sparse matrix. You are welcome to use **pthreads**, **OpenMP**, or any of the optimizations that were presented in class to accelerate this code. Do not change the sparsity of the matrices in `matmul.c`. Do not use a **GPU** and do not use **OpenBLAS** in your solution.  

There will be prizes awarded for the **fastest dense** and the **fastest sparse** implementations.

## Answers

### Initial Observations

The code to generate a sparse matrix shows that nonzero elements only appear in the upper right corner of the matrix (above a diagonal) with only odd-numbered rows containing the nonzero elements. All even numbered rows are entirely zero. This means that CSR might be a good format to store the sparse matrix in. At first assumption, the DIA format might have been used, but the diagonal concept is actually a red herring since the nonzero elements are not on the diagonal (they are just being contained by the diagonal). The sparse matrix multiplication duration does not include the time to convert to and form CSR format matrices.

### Dense Matrix Optimization Results
Compiling with no optimization flags:
```
Dense Matrix Multiplication Duration: 8395.593915 ms
Sparse Matrix Multiplication Duration: 8312.916006 ms
Sparse Matrix Multiplication Sparsity: 0.750977 
```

Compiling with `-O3` optimization flag:
```
Dense Matrix Multiplication Duration: 1598.310833 ms
Sparse Matrix Multiplication Duration: 1558.179026 ms
Sparse Matrix Multiplication Sparsity: 0.750977 
```

Compiling with `-O3` and `-fopenmp` optimization flags with pragma directives for dense multiplication:

```
Dense Matrix Multiplication Duration: 422.341221 ms
Sparse Matrix Multiplication Duration: 1548.046255 ms
Sparse Matrix Multiplication Sparsity: 0.750977 
```

Compiling with `-O3` and `-fopenmp` optimization flags with pragma directives for dense multiplication and CSR reformat for sparse matrices:
```
Dense Matrix Multiplication Duration: 380.621120 ms
Sparse Matrix Multiplication Duration: 4.249631 ms
Sparse Matrix Multiplication Sparsity: 0.750977 
```

### Fastest Dense Matrix Multiplication
***380.621120 ms***

### Fastest Sparse Matrix Multiplication
***4.249631 ms***

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```
- This program was compiled and run on Rho on the Vector system.
    - Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
    - 80 Cores
    - CPU MHz: 3600.00