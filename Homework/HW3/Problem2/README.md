# Problem 2 (30 Points)

In this problem, explore the benefits of compiling on the Explorer cluster with floating point vector extensions (e.g., AVX). To allocate a node on Explorer with AVX512 support, you will need to specify `--constraint=cascadelake`. To utilize AVX512 instructions, make sure to compile using the `-mavx512f` flag.

a.) Using the `dotproduct.c` example provided, develop an AVX512-accelerated version of matrix-vector multiplication. The example should give you a good start on the AVX intrinsics that you need to use in the program. Report on the speedup that you obtain as compared to a matrix-vector multiplication that does not use vectorization.

b.) Using the same code, generate an assembly listing (using the `-S` flag) and identify 3 different vector instructions that the compiler generated, explaining their operation.

## Part (a)


## Part (b)


## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```