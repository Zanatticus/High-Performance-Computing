# Problem 2 (30 Points)

In this problem, explore the benefits of compiling on the Explorer cluster with floating point vector extensions (e.g., AVX). To allocate a node on Explorer with AVX512 support, you will need to specify `--constraint=cascadelake`. To utilize AVX512 instructions, make sure to compile using the `-mavx512f` flag.

a.) Using the `dotproduct.c` example provided, develop an AVX512-accelerated version of matrix-vector multiplication. The example should give you a good start on the AVX intrinsics that you need to use in the program. Report on the speedup that you obtain as compared to a matrix-vector multiplication that does not use vectorization.

b.) Using the same code, generate an assembly listing (using the `-S` flag) and identify 3 different vector instructions that the compiler generated, explaining their operation.

## Part (a)

### 10x10 Matrix-Vector Multiplication
```
Matrix-Vector Multiplication Duration Without AVX: 0.000477 ms
Matrix-Vector Multiplication Duration With AVX: 0.000324 ms
AVX Speedup: 1.470588
Results Match
```

### 40x40 Matrix-Vector Multiplication
```
Matrix-Vector Multiplication Duration Without AVX: 0.003571 ms
Matrix-Vector Multiplication Duration With AVX: 0.001232 ms
AVX Speedup: 2.897833
Results Match
```

### 79x79 Matrix-Vector Multiplication (AVX Rounding Cutoff)
```
Matrix-Vector Multiplication Duration Without AVX: 0.019398 ms
Matrix-Vector Multiplication Duration With AVX: 0.005028 ms
AVX Speedup: 3.858118
Results Match
```

### 1024x1024 Matrix-Vector Multiplication
```
Matrix-Vector Multiplication Duration Without AVX: 2.900288 ms
Matrix-Vector Multiplication Duration With AVX: 0.294106 ms
AVX Speedup: 9.861384
Results Do Not Match:   result[0] = 357389440.000000, result_avx[0] = 357389824.000000
```

### 2048x2048 Matrix-Vector Multiplication
```
Matrix-Vector Multiplication Duration Without AVX: 8.709888 ms
Matrix-Vector Multiplication Duration With AVX: 0.865028 ms
AVX Speedup: 10.068905
Results Do Not Match:   result[0] = 2861216768.000000, result_avx[0] = 2861213952.000000
```

### 4096x4096 Matrix-Vector Multiplication
```
Matrix-Vector Multiplication Duration Without AVX: 17.864567 ms
Matrix-Vector Multiplication Duration With AVX: 3.988850 ms
AVX Speedup: 4.478626
Results Do Not Match:   result[0] = 22898118656.000000, result_avx[0] = 22898106368.000000
```

### Observations

The AVX512-accelerated version of matrix-vector multiplication provides a significant speedup over the non-vectorized version. The speedup is most pronounced for larger matrix sizes, with the speedup factor increasing as the matrix size grows. For smaller matrix sizes, the overhead of setting up the AVX512 instructions may outweigh the benefits of vectorization, leading to a smaller speedup or potentially even a slowdown in some special cases. Interestingly, the speedup which seems to ramp from `~1.5X` from `2x2` matrices spikes to `~30X` for a `1x1` matrix. Also, to matrices that are sufficiently large (since the values of the matrices are initialized to their indices), a rounding error in the AVX accelerated resultant vector is observed. This is likely because AVX512 chances the associativity of floating point data, leading to a different order of operations and thus a different result. Eventually with large enough numbers, the accumulation errors become significant enough to cause a difference in the results.

If we compare the two versions with matrices and vectors all initialized to `1.0`, the speedup results maintain roughly the same trend as well.

## Part (b)

### MOVUPS (Move Unaligned Packed Single-Precision Floating-Point Values)
- `vmovups	(%rdx,%rax), %zmm6`
- Moves 512 bits of packed single-precision floating-point values from the source operand (second operand) to the destination operand (first operand). This instruction can be used to load a ZMM register from a 512-bit float32 memory location, to store the contents of a ZMM register into memory. The destination operand is updated according to the writemask.

### VFMADD231PS (Vector Fused Multiply-Add of Packed Single Precision Floating-Point Values)
- `vfmadd231ps	(%rsi,%rax), %zmm6, %zmm0`
- This instruction multiplies the four, eight or sixteen packed single precision floating-point values from the second source operand to the four, eight or sixteen packed single precision floating-point values in the third source operand, adds the infinite precision intermediate result to the four, eight or sixteen packed single precision floating-point values in the first source operand, performs rounding and stores the resulting four, eight or sixteen packed single precision floating-point values to the destination operand (first source operand).

### VADDSS (Vector Add Scalar Single-Precision Floating-Point Values)
- `vaddss	%xmm4, %xmm0, %xmm3`
- Adds the low single-precision floating-point values from the second source operand and the first source operand, and stores the double-precision floating-point result in the destination operand. The second source operand can be an XMM register or a 64-bit memory location. The first source and destination operands are XMM registers.

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make``` or ```make assembly``` to generate an assembly listing.

- Node Allocation was made with `srun -p courses -N 1 --constraint=cascadelake --pty --time=03:00:00 /bin/bash`

- Data was collected on the `d0010` node of the Explorer cluster.
    - `Intel(R) Xeon(R) Platinum 8276 CPU @ 2.20GHz`