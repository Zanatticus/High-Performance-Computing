# Problem 4 (20 Points)

In this problem, you will utilize the OpenBLAS library available on Explorer. To use OpenBLAS, you will need to issue `load openblas/0.3.29`. You can refer to the `blas_simple.c` code provided for an example code.

a.) Develop a matrix-matrix multiplication, multiplying two single-precision floating point matrices that are **256×256** elements. Compare your implementation to your dense implementation in problem 3. Discuss which result is faster and why.

b.) Run your **OpenBLAS-accelerated** program on two different CPU platforms in Explorer. Discuss the **CPUs** you are using, and the **performance differences** you obtain.

## Part (a)
The fastest dense matrix multiplication duration I managed to get from problem 3 was `46.639208 ms` and `380.621120 ms` for `N=256` and `N=512`, respectively.
Using OpenBLAS, I managed to get the following results for `N=256` and `N=512`:

```
Matrix Dimensions: N=256, M=256, K=256
OpenBLAS Matrix Multiplication Duration: 0.716843 ms
```

```
Matrix Dimensions: N=512, M=512, K=512
OpenBLAS Matrix Multiplication Duration: 4.348857 ms
```

Overall, for `N=256`, OpenBLAS had a speedup of `65.062x` and for `N=512`, OpenBLAS had a speedup of `87.522x`. This is likely because OpenBLAS is optimized for matrix multiplication and is able to take advantage of parallelism and other optimizations that I must not have implemented in my dense matrix multiplication code. For example, OpenBLAS utilizes SIMD instructions (e.g. AVX, AVX512, etc.), multithreading (for parallelism and throughput), optimized cache and memory access patterns, and more, to perform matrix multiplication faster.

The above results were gathered on node `c2192` on the Explorer cluster.

## Part (b)

### Node c2192
- **N=256:** 0.716843 ms, 65.062x speedup
- **N=512:** 4.348857 ms, 87.522x speedup
- **CPU:** Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
    - **Cores:** 28
    - **Max MHz:** 3300 MHz
    - **Min MHz:** 1200 MHz
- **Cache Hierarchy:**
    - **L1d:** 896 KiB (28 instances)
    - **L1i:** 896 KiB (28 instances)
    - **L2:** 7 MiB (28 instances)
    - **L3:** 70 MiB (2 instances)

### Node d0011 (cascade lake) 
- **N=256:** 0.705761 ms, 66.084x speedup
- **N=512:** 4.204937 ms, 90.518x speedup
- **CPU:** Intel(R) Xeon(R) Platinum 8276 CPU @ 2.20GHz
    - **Cores:** 56
    - **Max MHz:** 4000 MHz
    - **Min MHz:** 1000 MHz
- **Cache Hierarchy:**
    - **L1d:** 1.8 MiB (56 instances)
    - **L1i:** 1.8 MiB (56 instances)
    - **L2:** 56 MiB (56 instances)
    - **L3:** 77 MiB (2 instances)

The cascade lake node (d0011) slightly outperformed node c2192. This could be attributed to the fact that the cascade lake node has a higher clock speed and more cores to utilize during the matrix multiplication program. Also, the cascade lake node has larger cache sizes which could have helped in reducing cache misses (since more of the matrix data could be stored in the cache) as compared to node c2192.

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```