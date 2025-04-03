# EECE 5640 Homework

This directory contains homework assignments for the High-Performance Computing course. Each subfolder corresponds to a specific homework assignment, containing the source code, documentation, and any required input/output files.

## Contents

- [**Homework 1**](./HW1): Benchmarking and pthreads introduction
  - *Problem 1*: Benchmark 3 workloads on two systems and analyze compiler optimizations
  - *Problem 2*: Parallel sorting of 10,000 integers using pthreads
  - *Problem 3*: System architecture report on a selected Explorer node
  - *Problem 4*: Analyze Top500 systems and design your own HPC architecture
  - *Problem 5*: Compare Green500 and Top500 trends (extra credit)
  - *Extra Credit Assignment*: Loop tiling and OpenMP in matrix multiplication

- [**Homework 2**](./HW2): Concurrency and OpenMP in real-world problems
  - *Problem 1*: Monte Carlo and Leibniz Pi computation using pthreads/OpenMP
  - *Problem 2*: Dining Philosophers with pthreads, synchronization strategies
  - *Problem 3*: OpenMP-based graph coloring and scaling analysis
  - *Problem 4*: Sparse matrix optimizations using reordering techniques

- [**Homework 3**](./HW3): Vectorization, BLAS, and matrix computation optimizations
  - *Problem 1*: Evaluate float vs double precision using Taylor series
  - *Problem 2*: AVX512-accelerated matrix-vector multiplication with intrinsics
  - *Problem 3*: Optimized matmul with dense and sparse matrices (no GPU/BLAS)
  - *Problem 4*: OpenBLAS matrix-matrix multiplication and hardware comparison
  - *Problem 5*: Analyze a novel sparse matrix format from literature

- [**Homework 4**](./HW4): MPI programming and distributed histogramming
  - *Problem 1*: Message passing and process/node ID relay across 64 MPI processes
  - *Problem 2*: Parallel histogramming with MPI and performance analysis on 2–8 nodes
  - *Problem 3*: Comparative study of two MPI performance analysis tools
  - *Problem 4*: Exascale computing barriers past and present (extra credit)

- [**Homework 5**](./HW5): CUDA, tiling, and GPU vs CPU performance
  - *Problem 1*: Histogramming with CUDA and performance comparison with OpenMP
  - *Problem 2*: 3D stencil computation with tiled shared memory GPU kernels
  - *Problem 3*: Ampere vs Hopper GPU architecture comparison
  - *Problem 4*: Compute π on the GPU using the Leibniz series (extra credit)

## Code Formatting

All source code in this directory follows the [Clang-Format](https://clang.llvm.org/docs/ClangFormat.html) style guidelines. To ensure consistency, use the `clang-format` tool to format your code before submission. Note that some Clang formatting options are not supported for some code within the homework assignments (see `#pragma` directives for example, clang-format does not support them).)

### Usage

To format your code, run the following command in the terminal for a specific homework subdirectory (HW1 subdirectory for example):

```bash
./format-all-files.sh --name=HW1
```

OR format all files at once with the following command:

```bash
./format-all-files.sh --all
```