# Problem 3 (20 Points)

In this problem, you will modify the `matmul.c` program provided, optimizing the execution of the matrix multiplication with first a dense matrix, and second with a sparse matrix. You are welcome to use **pthreads**, **OpenMP**, or any of the optimizations that were presented in class to accelerate this code. Do not change the sparsity of the matrices in `matmul.c`. Do not use a **GPU** and do not use **OpenBLAS** in your solution.  

There will be prizes awarded for the **fastest dense** and the **fastest sparse** implementations.

## Answers

### Optimization Observations
Compiling with no optimization flags:
```
Starting dense matrix multiply... 
Dense Matrix Multiplication Result: 4.44488e+07 
Dense Matrix Multiplication Duration: 8395.593915 ms
Starting sparse matrix multiply... 
Sparse Matrix Multiplication Result: 0 
Sparse Matrix Multiplication Duration: 8312.916006 ms
Sparse Matrix Multiplication Sparsity: 0.750977 
```

Compiling with `-O3` optimization flag:
```
Starting dense matrix multiply... 
Dense Matrix Multiplication Result: 4.44488e+07 
Dense Matrix Multiplication Duration: 1598.310833 ms
Starting sparse matrix multiply... 
Sparse Matrix Multiplication Result: 0 
Sparse Matrix Multiplication Duration: 1558.179026 ms
Sparse Matrix Multiplication Sparsity: 0.750977 
```

Compiling with `-O3` and `-fopenmp` optimization flags with pragma directives:
```
Starting dense matrix multiply... 
Dense Matrix Multiplication Result: 4.44488e+07 
Dense Matrix Multiplication Duration: 422.341221 ms
Starting sparse matrix multiply... 
Sparse Matrix Multiplication Result: 0 
Sparse Matrix Multiplication Duration: 1548.046255 ms
Sparse Matrix Multiplication Sparsity: 0.750977 
```

### Fastest Dense Matrix Multiplication

### Fastest Sparse Matrix Multiplication


## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```