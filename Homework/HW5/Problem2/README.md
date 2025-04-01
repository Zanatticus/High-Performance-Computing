# Problem 1 (40 Points)

The code below carries out a “nearest neighbor” or “stencil” computation. This class of algorithm appears frequently in image processing and visualization applications. The memory reference pattern for matrix b exhibits reuse in 3 dimensions. Your task is to develop a C/CUDA version of this code that initializes b on the host and then uses tiling on the GPU to exploit locality in GPU shared memory across the 3 dimensions of b:

```
#define n 32
float a[n][n][n], b[n][n][n];
for (i = 1; i < n - 1; i++)
    for (j = 1; j < n - 1; j++)
        for (k = 1; k < n - 1; k++) {
            a[i][j][k] = 0.75 * (b[i - 1][j][k] + b[i + 1][j][k] + b[i][j - 1][k] +
                                    b[i][j + 1][k] + b[i][j][k - 1] + b[i][j][k + 1]);
        }
```

a. Evaluate the performance of computing a tiled versus non-tiled implementation in your GPU application. Explore what happens when you change the value of n. Consider the performance for at least two additional values for n.
b. Explore and report on other optimizations to accelerate your code on the GPU further.

## Part (a)

The maximum tiling size I could implement was 10. The following results were obtained for the different values of N while tile size was set to 10. The results are divided into three sections: C/C++ default computation, CUDA non-tiled stencil computation, and CUDA tiled stencil computation, with a plot at the end showcasing the execution time of the implementations.

#### N = 8
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 512
Execution Time: 0.0000008480 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 512
Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (1, 1, 1) = 1
Execution Time (including device sync & copy): 0.0001449230 seconds
Execution Time (excluding device sync & copy): 0.0001191850 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 512
Tile Size: 10 | Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (1, 1, 1) = 1
Execution Time (including device sync & copy): 0.0000578860 seconds
Execution Time (excluding device sync & copy): 0.0000385520 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### N = 16
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 4096
Execution Time: 0.0000115220 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 4096
Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (2, 2, 2) = 8
Execution Time (including device sync & copy): 0.0001544940 seconds
Execution Time (excluding device sync & copy): 0.0001146360 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 4096
Tile Size: 10 | Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (2, 2, 2) = 8
Execution Time (including device sync & copy): 0.0000813660 seconds
Execution Time (excluding device sync & copy): 0.0000424970 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### N = 32
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 32768
Execution Time: 0.0000635430 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 32768
Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (4, 4, 4) = 64
Execution Time (including device sync & copy): 0.0002637590 seconds
Execution Time (excluding device sync & copy): 0.0001294710 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 32768
Tile Size: 10 | Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (4, 4, 4) = 64
Execution Time (including device sync & copy): 0.0001004860 seconds
Execution Time (excluding device sync & copy): 0.0000456810 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### N = 64
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 262144
Execution Time: 0.0006465060 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 262144
Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (7, 7, 7) = 343
Execution Time (including device sync & copy): 0.0010352730 seconds
Execution Time (excluding device sync & copy): 0.0001290720 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 262144
Tile Size: 10 | Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (7, 7, 7) = 343
Execution Time (including device sync & copy): 0.0269766680 seconds
Execution Time (excluding device sync & copy): 0.0000500480 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### N = 128
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 2097152
Execution Time: 0.0036693990 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 2097152
Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (13, 13, 13) = 2197
Execution Time (including device sync & copy): 0.0037689610 seconds
Execution Time (excluding device sync & copy): 0.0001254600 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 2097152
Tile Size: 10 | Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (13, 13, 13) = 2197
Execution Time (including device sync & copy): 0.0274593330 seconds
Execution Time (excluding device sync & copy): 0.0000506910 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### N = 256
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 16777216
Execution Time: 0.0280731530 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 16777216
Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (26, 26, 26) = 17576
Execution Time (including device sync & copy): 0.0296211910 seconds
Execution Time (excluding device sync & copy): 0.0001773530 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 16777216
Tile Size: 10 | Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (26, 26, 26) = 17576
Execution Time (including device sync & copy): 0.0122606390 seconds
Execution Time (excluding device sync & copy): 0.0001329600 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### N = 512
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2281501430 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (52, 52, 52) = 140608
Execution Time (including device sync & copy): 0.2365533530 seconds
Execution Time (excluding device sync & copy): 0.0001840300 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 10 | Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (52, 52, 52) = 140608
Execution Time (including device sync & copy): 0.0944161650 seconds
Execution Time (excluding device sync & copy): 0.0001522010 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

### Performance Plots

![C_C++ Execution Time (s), Non-Tiled CUDA Execution Time (s) and Tiled CUDA Execution Time (s)](plots/C_C++%20Execution%20Time%20(s),%20Non-Tiled%20CUDA%20Execution%20Time%20(s)%20and%20Tiled%20CUDA%20Execution%20Time%20(s).png)

![C_C++ Execution Time (s), Non-Tiled CUDA Execution Time (s) and Tiled CUDA Execution Time (s)(1)](plots/C_C++%20Execution%20Time%20(s),%20Non-Tiled%20CUDA%20Execution%20Time%20(s)%20and%20Tiled%20CUDA%20Execution%20Time%20(s)(1).png)

![Non-Tiled CUDA Execution Time (s) and Tiled CUDA Execution Time (s)](plots/Non-Tiled%20CUDA%20Execution%20Time%20(s)%20and%20Tiled%20CUDA%20Execution%20Time%20(s).png)

## Part (b)

### Cubed Tile Size Exploration
Varying the tile size from 1-10 with a set N value of 512, and setting the block size to `TILE_SIZE * TILE_SIZE * TILE_SIZE` and the grid size to `(N + TILE_SIZE - 1) / TILE_SIZE` for each dimension (x,y,z), the following results were obtained:

#### Tile Size = 1
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2197300830 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (1, 1, 1) = 1 | Grid Size (Number of Blocks): (512, 512, 512) = 134217728
Execution Time (including device sync & copy): 0.5257719020 seconds
Execution Time (excluding device sync & copy): 0.0001762510 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 1 | Block Size (Threads Per Block): (1, 1, 1) = 1 | Grid Size (Number of Blocks): (512, 512, 512) = 134217728
Execution Time (including device sync & copy): 0.3840996130 seconds
Execution Time (excluding device sync & copy): 0.0001101310 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```


#### Tile Size = 2
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2190685120 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (2, 2, 2) = 8 | Grid Size (Number of Blocks): (256, 256, 256) = 16777216
Execution Time (including device sync & copy): 0.2521871010 seconds
Execution Time (excluding device sync & copy): 0.0001754270 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 2 | Block Size (Threads Per Block): (2, 2, 2) = 8 | Grid Size (Number of Blocks): (256, 256, 256) = 16777216
Execution Time (including device sync & copy): 0.1131959800 seconds
Execution Time (excluding device sync & copy): 0.0001171920 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### Tile Size = 3
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2193801180 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (3, 3, 3) = 27 | Grid Size (Number of Blocks): (171, 171, 171) = 5000211
Execution Time (including device sync & copy): 0.2287246260 seconds
Execution Time (excluding device sync & copy): 0.0001747670 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 3 | Block Size (Threads Per Block): (3, 3, 3) = 27 | Grid Size (Number of Blocks): (171, 171, 171) = 5000211
Execution Time (including device sync & copy): 0.0907904680 seconds
Execution Time (excluding device sync & copy): 0.0001196800 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### Tile Size = 4
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2195382820 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (4, 4, 4) = 64 | Grid Size (Number of Blocks): (128, 128, 128) = 2097152
Execution Time (including device sync & copy): 0.2315951590 seconds
Execution Time (excluding device sync & copy): 0.0001752310 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 4 | Block Size (Threads Per Block): (4, 4, 4) = 64 | Grid Size (Number of Blocks): (128, 128, 128) = 2097152
Execution Time (including device sync & copy): 0.0887722790 seconds
Execution Time (excluding device sync & copy): 0.0001171080 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### Tile Size = 5
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2192068990 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (5, 5, 5) = 125 | Grid Size (Number of Blocks): (103, 103, 103) = 1092727
Execution Time (including device sync & copy): 0.2237763130 seconds
Execution Time (excluding device sync & copy): 0.0001764920 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 5 | Block Size (Threads Per Block): (5, 5, 5) = 125 | Grid Size (Number of Blocks): (103, 103, 103) = 1092727
Execution Time (including device sync & copy): 0.0802186180 seconds
Execution Time (excluding device sync & copy): 0.0001146050 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### Tile Size = 6
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2192910640 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (6, 6, 6) = 216 | Grid Size (Number of Blocks): (86, 86, 86) = 636056
Execution Time (including device sync & copy): 0.2134836340 seconds
Execution Time (excluding device sync & copy): 0.0001696160 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 6 | Block Size (Threads Per Block): (6, 6, 6) = 216 | Grid Size (Number of Blocks): (86, 86, 86) = 636056
Execution Time (including device sync & copy): 0.0739495880 seconds
Execution Time (excluding device sync & copy): 0.0001214500 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### Tile Size = 7
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2179097950 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (7, 7, 7) = 343 | Grid Size (Number of Blocks): (74, 74, 74) = 405224
Execution Time (including device sync & copy): 0.2091049390 seconds
Execution Time (excluding device sync & copy): 0.0001952820 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 7 | Block Size (Threads Per Block): (7, 7, 7) = 343 | Grid Size (Number of Blocks): (74, 74, 74) = 405224
Execution Time (including device sync & copy): 0.0704871500 seconds
Execution Time (excluding device sync & copy): 0.0001480180 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```


#### Tile Size = 8
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2197923940 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (8, 8, 8) = 512 | Grid Size (Number of Blocks): (64, 64, 64) = 262144
Execution Time (including device sync & copy): 0.2070560910 seconds
Execution Time (excluding device sync & copy): 0.0001699440 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 8 | Block Size (Threads Per Block): (8, 8, 8) = 512 | Grid Size (Number of Blocks): (64, 64, 64) = 262144
Execution Time (including device sync & copy): 0.0656135710 seconds
Execution Time (excluding device sync & copy): 0.0001223830 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```


#### Tile Size = 9
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2202654180 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (9, 9, 9) = 729 | Grid Size (Number of Blocks): (57, 57, 57) = 185193
Execution Time (including device sync & copy): 0.2067529490 seconds
Execution Time (excluding device sync & copy): 0.0001834070 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 9 | Block Size (Threads Per Block): (9, 9, 9) = 729 | Grid Size (Number of Blocks): (57, 57, 57) = 185193
Execution Time (including device sync & copy): 0.0658499980 seconds
Execution Time (excluding device sync & copy): 0.0001423220 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```


#### Tile Size = 10
```
==========================================
C/C++ Default Computation Results
==========================================

Total elements: 134217728
Execution Time: 0.2179391140 seconds

==========================================
CUDA Non-Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (52, 52, 52) = 140608
Execution Time (including device sync & copy): 0.2098333060 seconds
Execution Time (excluding device sync & copy): 0.0001929040 seconds

Verifying Non-Tiled Kernel Results...
Results Match Ground-Truth!

==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 10 | Block Size (Threads Per Block): (10, 10, 10) = 1000 | Grid Size (Number of Blocks): (52, 52, 52) = 140608
Execution Time (including device sync & copy): 0.0669604910 seconds
Execution Time (excluding device sync & copy): 0.0001409430 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

### Performance Plots For Cubed Tile Size Exploration

![Non-Tiled CUDA Execution Time (s) and Tiled CUDA Execution Time (s) Including Synchronization and Memory Copy](plots/Non-Tiled%20CUDA%20Execution%20Time%20(s)%20and%20Tiled%20CUDA%20Execution%20Time%20(s)%20Including%20Synchronization%20and%20Memory%20Copy.png)

![Non-Tiled CUDA Execution Time (s) and Tiled CUDA Execution Time (s) Excluding Synchronization and Memory Copy](plots/Non-Tiled%20CUDA%20Execution%20Time%20(s)%20and%20Tiled%20CUDA%20Execution%20Time%20(s)%20Excluding%20Synchronization%20and%20Memory%20Copy.png)

From these plots, we can see that tiling does not really impact the performance of raw program execution in a noticeable way, but instead impacts the performance of execution when including memory synchronization and transfers. There is a large improvement when jumping from 1 tile to 2 tiles, but that improvement immediately tapers off and has a severely diminishing return with an increasing number of tiles.

### Optimizing Full Utilization

NVIDIA Tesla P100 (Pascal GP100) Architecture Key Specs (PCIE-12GB variant):
1. Streaming Multiprocessors (SMs): 56 SMs
2. CUDA Cores per SM: 64 (organized as 2 units of 32)
3. Warp Size: 32 threads (standard)
4. Max Threads per Block: 1024

From the above specifications, we want to make sure that the max number of threads is a multiple of the warp size (32) so as to fully utilize the smallest block of execution in a GPU. As such, to maximize the number of threads to fully utilize the hardware resources, I had to configure the program to use an asymmetrical, non-cubed shaped block and grid size to achieve the maximum 1024 threads. I compared just the implementation for a tile size of 8, a block size of 8x8x16 (1024 threads), and a corresponding grid size of 64x64x32 (131072 blocks). The results were as follows:

#### Tile Size = 8, Block Size = (8, 8, 16) = 1024, Grid Size = (64, 64, 32) = 131072

```
==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 8 | Block Size (Threads Per Block): (8, 8, 16) = 1024 | Grid Size (Number of Blocks): (64, 64, 32) = 131072
Execution Time (including device sync & copy): 0.0527355390 seconds
Execution Time (excluding device sync & copy): 0.0001530490 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

#### Tile Size = 8, Cubed Dimensions

```
==========================================
CUDA Tiled Stencil Computation Results
==========================================

Total elements: 134217728
Tile Size: 8 | Block Size (Threads Per Block): (8, 8, 8) = 512 | Grid Size (Number of Blocks): (64, 64, 64) = 262144
Execution Time (including device sync & copy): 0.0656135710 seconds
Execution Time (excluding device sync & copy): 0.0001223830 seconds

Verifying Tiled Kernel Results...
Results Match Ground-Truth!
```

In optimizing the block size we can see a slight improvement in the execution time of the tiled kernel (including device sync and copy) compared to when the block size was not fully optimized (0.0527... seconds vs. 0.0656... seconds).

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```
- The CUDA program needed a node with access to an NVIDIA GPU:
```srun --partition=courses-gpu --nodes=1 --pty --gres=gpu:1 --time=01:00:00 /bin/bash```