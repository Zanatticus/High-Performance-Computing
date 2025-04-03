## Homework 1 Extra Credit Assignment

Explain the resulting performance while combining loop tiling and OpenMP in the matrix multiplication code `mat-mat-mul.openMP.c`. Submit a set of experiments that support your conclusions. Conclusions without supporting evidence will not receive any credit. You can earn up to 20 points of extra credit.

## Results

#### Block Size = 64

```
Check c[511][511] = 1024
Check c[511][511] = 1024
Check c[511][511] = 1024
Time for first loop (not parallelized) = 72.495593
Time for second loop (parallelized) = 17.022816
Time for third loop (parallelized and tiled) = 19.326230
```
