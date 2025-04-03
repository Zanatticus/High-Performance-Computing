## Homework 1 Extra Credit Assignment

Explain the resulting performance while combining loop tiling and OpenMP in the matrix multiplication code `mat-mat-mul.openMP.c`. Submit a set of experiments that support your conclusions. Conclusions without supporting evidence will not receive any credit. You can earn up to 20 points of extra credit.

## Results

#### Matrix Size = 256, Block Size = 32
```
Check c[255][255] = 512
Check c[255][255] = 512
Check c[255][255] = 512
Time for first loop (not parallelized) = 7.372049
Time for second loop (parallelized) = 2.903589
Time for third loop (parallelized and tiled) = 1.976034
```

#### Matrix Size = 512, Block Size = 32
```
Check c[511][511] = 1024
Check c[511][511] = 1024
Check c[511][511] = 1024
Time for first loop (not parallelized) = 72.999603
Time for second loop (parallelized) = 17.916428
Time for third loop (parallelized and tiled) = 10.893311
```

#### Matrix Size = 1024, Block Size = 32
```
Check c[1023][1023] = 2048
Check c[1023][1023] = 2048
Check c[1023][1023] = 2048
Time for first loop (not parallelized) = 581.402366
Time for second loop (parallelized) = 293.306907
Time for third loop (parallelized and tiled) = 108.819711
```

#### Matrix Size = 2048, Block Size = 32
```
Check c[2047][2047] = 4096
Check c[2047][2047] = 4096
Check c[2047][2047] = 4096
Time for first loop (not parallelized) = 4655.247017
Time for second loop (parallelized) = 2362.602352
Time for third loop (parallelized and tiled) = 605.065219
```

#### Matrix Size = 4096, Block Size = 32
```
Check c[4095][4095] = 8192
Check c[4095][4095] = 8192
Check c[4095][4095] = 8192
Time for first loop (not parallelized) = 170970.085622
Time for second loop (parallelized) = 38735.821467
Time for third loop (parallelized and tiled) = 7705.725870
```

#### Matrix Size = 256, Block Size = 64
```
Check c[255][255] = 512
Check c[255][255] = 512
Check c[255][255] = 512
Time for first loop (not parallelized) = 7.316591
Time for second loop (parallelized) = 3.714261
Time for third loop (parallelized and tiled) = 4.077266
```

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

#### Matrix Size = 4096, Block Size = 64

```
Check c[4095][4095] = 8192
Check c[4095][4095] = 8192
Check c[4095][4095] = 8192
Time for first loop (not parallelized) = 171936.734603
Time for second loop (parallelized) = 40081.673256
Time for third loop (parallelized and tiled) = 8360.462945
```

### Discussion

These results were gathered on the Explorer Cluster, Node c2195. The hardware configuration for this platform consists of the following key elements:
- L1d cache: 32 KB per core × 28 cores = 896 KB total
- L2 cache: 256 KB per core × 28 = 7 MB total
- L3 cache: 35 MB per socket × 2 = 70 MB total (shared per socket)
- Total cores: 28 physical (14 per socket)

Optimize Tiling:
- Integer matrix multiplication, 4 bytes per int
    - Tile Size = 32: 
        - 32 x 32 x 4 = 8 KB per matrix tile
        - 3 Matrices (A, B, C)
            - 3 x 8 KB = 24 KB per tile computation
    - Tile Size = 64:
        - 64 x 64 x 4 = 16 KB per matrix tile
        - 3 Matrices (A, B, C)
            - 3 x 16 KB = 48 KB per tile computation

    - Since this is parallelized, the size of 32 KB per core matches the Tile Size of 32 best, which matches the performance results gathered above.
