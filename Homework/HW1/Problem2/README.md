# Problem 2 (30 Points)

In this problem, you will start by selecting a sorting algorithm to sort 10,000 random integers with values 1-10,000. We will use pthreads in the problem to generate a parallel version of your sorting algorithm. If you select code from the web (and you are encouraged to do so), make sure to cite the source of where you obtained the code. You are also welcome to write your own code from scratch. You will also need to generate the 10,000 random integers (do not time this part of your program). Make sure to print out the final results of the sorted integers to demonstrate that your code works (do not include a printout of the 10,000 numbers). Provide your source code in your submission on Canvas.

&nbsp;&nbsp;&nbsp;&nbsp;a. Run your program with 1, 2, 4, 8 and 32 threads on any system of your choosing. The system should have at least 4 cores for you to run your program on. Report the performance of your program. 
&nbsp;&nbsp;&nbsp;&nbsp;b. Describe some of the challenges faced when performing sorting with multiple threads. 
&nbsp;&nbsp;&nbsp;&nbsp;c. Evaluate both the weak scaling and strong scaling properties of your sorting implementation. 

*You should include your C/C++ program for this problem. Also include your written answers to questions 2 (a-c) in your homework 1 write-up.



*Sources Cited: https://www.geeksforgeeks.org/merge-sort-using-multi-threading/*

*Code compiled and run on vector.coe.neu.edu (nproc=80)*

## Part (a)
Running the program with 1, 2, 4, 8, 32 threads:
|    **Threads**   	|   1   	| 2     	| 4     	| 8     	| 32     	|
|:------------:	|:-----:	|-------	|-------	|-------	|--------	|
| **Latency (ms)** 	| 6.643 	| 6.851 	| 8.509 	| 9.357 	| 14.121 	|

Surprisingly, the latency increases as the number of threads increases. 

## Part (b)
Describe some of the challenges faced when performing sorting with multiple threads:

1. **Thread Creation Overhead**: Creating threads is an expensive operation. The overhead of creating threads is likely to be more than the time saved by parallelizing the merge sort algorithm.
2. **Thread Synchronization**: When multiple threads are working on the same data, it is essential to synchronize the threads. This means that access to shared data structures and resources (like merging sorted subarrays) needs to be synchronized which leads to delays.
3. **Memory Contention**: Multiple threads accessing the same memory locations can lead to memory contention. This can slow down the program due to cache misses and other memory access delays.


## Part (c)

#### Weak Scaling of the Merge Sort Algorithm:

Starting Number of Elements: 10000

| **Number of Elements** 	| 10000 	| 20000  	| 40000  	| 80000  	| 160000  	| 320000  	| 640000  	| 1280000 	| 2560000 	|
|------------------------	|-------	|--------	|--------	|--------	|---------	|---------	|---------	|---------	|---------	|
|       **Threads**      	|   1   	| 2      	| 4      	| 8      	| 16      	| 32      	| 64      	| 128     	| 256     	|
|    **Latency (ms)**    	| 6.506 	| 13.655 	| 28.144 	| 76.731 	| 227.545 	| 472.222 	| 1219.09 	| 2087.72 	| 4186.92 	|


#### Strong Scaling of the Merge Sort Algorithm:

Number of Elements: 10000

|    **Threads**   	|   1   	| 2     	| 4     	| 8     	| 16     	| 32     	| 64    	| 128    	| 256    	|
|:----------------:	|:-----:	|-------	|-------	|-------	|--------	|--------	|-------	|--------	|--------	|
| **Latency (ms)** 	| 6.643 	| 6.851 	| 8.509 	| 9.357 	| 11.575 	| 14.121 	| 19.89 	| 30.428 	| 44.603 	|

#### Evaluation of Scaling:
**Weak Scaling:** The program shows an increase in latency as the number of threads and problem size increase.
**Strong Scaling:** The program does not exhibit good strong scaling, as the latency increases with the number of threads.


## Miscellaneous
- The program was compiled and run using the following command:
```g++ -pthread merge_sort.cpp -o merge_sort``` or simply ```make```
