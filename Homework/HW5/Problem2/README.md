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

// EXPLORE DIFFERENT TILE SIZES INSTEAD OF PRESET 10

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```
- The CUDA program needed a node with access to an NVIDIA GPU:
```srun --partition=courses-gpu --nodes=1 --pty --gres=gpu:1 --time=01:00:00 /bin/bash```