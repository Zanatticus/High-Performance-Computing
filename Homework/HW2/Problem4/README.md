# Problem 4 (10 Points)

Read the paper provided on Sparse Matrix-Vector (SpMV) reordering optimizations performed on multicore processors. (only parts a and b are required)

a.) Select one of the reordering schemes described and provide pseudocode (i.e., detailed steps and describe the transformations) for 2 of the schemes described in the paper.
b.) For 1 of the hardware platforms discussed in the paper, provide the details of the associated memory hierarchy on the system.
c.) Completing this part is optional, but can earn extra credit on your quiz average. 

Develop your own SpMV (sparse matrix-vector multiplication) implementation that uses one of the memory reordering algorithms described in the paper (6 are provided). You will need to generate your own sparse matrices or use those available from MatrixMarket (https://math.nist.gov/MatrixMarket/). Report on the speedup achieved as compared to using the standard dense SpMV kernel approach. 

Students can earn an extra 10 points of extra credit for implementing and evaluating each memory reordering algorithm (60 maximum points).

*Written answers to the questions should be included in your homework 2 write-up in pdf format. You should include your C/C++ program and the README file in the zip file submitted.

## Part (a)

### Scheme 1 Pseudocode (Reverse Cuthill-McKee (CM))

Assuming a symmetric `n x n` matrix is given:
1. Initialize a queue `Q` and an empty list for permutation order of objects `R`.
2. Find the object with the smallest degree whose index has not yet been added to `R` and add it's index to `R`.
3. When an index is added to `R`. add all of the neighbors of the object at that index to the queue `Q` in increasing order of degree.
    - The neighbors must be nodes with non-zero values amongst the non-diagonal elements in the row of the object.
4. Take the first node in the queue `Q` and add it to `R` if it has not already been added. Then add the neighbors of that node to the queue `Q` in increasing order of degree.
5. Repeat step 4 until the queue `Q` is empty.
6. If the queue `Q` is empty, but objects in the matrix have not been added to `R`, repeat steps 2-5 until all objects have been added to `R`.
7. Once all objects are included in `R`, the matrix is reordered according to the order of the indices in `R`.
8. Reverse the indices in `R` to get the finalized reverse Cuthill-McKee ordering.

### Scheme 2 Pseudocode (Gray Ordering)



## Part (b)

Associated Memory Hierarchy for the **Milan B** hardware platform:
- CPU: **[AMD Epyc 7763](https://en.wikichip.org/wiki/amd/epyc/7763), x86-64, 128 cores**
- Cache Levels:
    - L1 Instruction Cache (per core): **32 KiB**
        - Associativity: **8-way**
        - Total Size: 64 x 32 KiB = 2048 KiB = **2 MiB**
    - L1 Data Cache (per core): **32 KiB**
        - Associativity: **8-way**
        - Total Size: 64 x 32 KiB = 2048 KiB = **2 MiB**
    - L2 Cache (per core): **512 KiB**
        - Associativity: **8-way**
        - Total Size: 64 x 512 KiB = 32768 KiB = **32 MiB**
    - L3 Cache (per socket): **256 MiB**
        - Associativity: **16-way**
        - Total Size: 8 x 32 MiB = **256 MiB**
- Integrated Memory Controller:
    - Memory Type: **DDR4-3200**
    - Memory Channels: **8**
    - Max Memory Capacity: **4 TiB**
    - Max Bandwidth: **190.73 GiB/s**

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```