## Homework 1 Extra Credit Assignment

Explain the resulting performance while combining loop tiling and OpenMP in the matrix multiplication code `mat-mat-mul.openMP.c`. Submit a set of experiments that support your conclusions. Conclusions without supporting evidence will not receive any credit. You can earn up to 20 points of extra credit.

## Results

#### Matrix Size = 512, Block Size = 64

```
Check c[511][511] = 1024
Check c[511][511] = 1024
Check c[511][511] = 1024
Time for first loop (not parallelized) = 74.428008
Time for second loop (parallelized) = 18.023810
Time for third loop (parallelized and tiled) = 22.615076
```

#### Matrix Size = 1024, Block Size = 64

```
Check c[1023][1023] = 2048
Check c[1023][1023] = 2048
Check c[1023][1023] = 2048
Time for first loop (not parallelized) = 579.254628
Time for second loop (parallelized) = 293.887828
Time for third loop (parallelized and tiled) = 102.262245
```

#### Matrix Size = 2048, Block Size = 64

```
Check c[2047][2047] = 4096
Check c[2047][2047] = 4096
Check c[2047][2047] = 4096
Time for first loop (not parallelized) = 4771.836319
Time for second loop (parallelized) = 2363.713324
Time for third loop (parallelized and tiled) = 1352.423509
```