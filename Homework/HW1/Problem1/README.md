# Problem 1 (30 Points)

In this problem, you are to select a set of 3 single-threaded benchmark programs. Try to provide some diversity in the set of workloads you have chosen (e.g., floating point, integer, memory intensive, sparse). Then complete the following experiments using these benchmarks. Make sure to provide as many details as possible about the systems you are using, and where you obtained the source code for these benchmarks. 
&nbsp;&nbsp;&nbsp;&nbsp;a. Compile and run these 3 benchmarks on two different Linux-based system of your choosing (you can also use either the COE systems or the Explorer systems). Provide detailed information about the platforms you chose, including the model and frequency of the CPU, number of cores, the memory size, and the operating system version. You should record the execution time, averaged over 10 runs of the program. Is any run of the program faster than another? If so, comment on any difference you observe in your  rite-up, providing some justification for the differences. 

&nbsp;&nbsp;&nbsp;&nbsp;b. Comment on the differences observed in the timings on the two systems and try to explain what is responsible for these differences. 

&nbsp;&nbsp;&nbsp;&nbsp;c. Next, explore the compiler optimizations available with the compiler on one of your systems (e.g., gcc or g++), and report on the performance improvements found for the 3 workloads. Describe the optimization you applied and provide insight why each of your benchmarks benefitted from the specific compiler optimization applied. 

&nbsp;&nbsp;&nbsp;&nbsp;d. Summarizing benchmark performance with a single metric can be understood by a wider audience. Performance metrics such as FLOPS and MIPS have been used to report the performance of different systems. For one of your workloads, devise your own metric, possibly following how SPEC reports on performance. Generate a plot using this metric, while running the workload on 2 different CPUs platforms. 

&nbsp;&nbsp;&nbsp;&nbsp;e. Assume that you were going to rewrite these applications using pthreads. Describe how you would use pthreads to obtain additional speedup by running your benchmark on multiple cores. 

*Answers to this question should be included in your homework 1 write-up in pdf format.

## Part (a)

**Explorer Cluster, Node c0744:**
- CPU Model: Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz
- CPU Frequency: Min=1200.0000 MHz, Max=3300.0000 MHz, Current=3300.0000 MHz
- CPU Cores: 28
- Memory Size: 251 GiB (263358376)
- Cache Hierarchy: 
    - L1d: 32K, 8-way set associative
    - L1i: 32K, 8-way set associative
    - L2: 256K, 8-way set associative
    - L3: 35M, 20-way set associative
- Operating System: Rocky Linux 9.3 (Blue Onyx)

**Vector:**
- CPU Model: Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
- CPU Frequency: Min=1200.0000 MHz, Max=3600.0000 MHz, Current=3600.0000 MHz
- CPU Cores: 80
- Memory Size: 755 GiB (792237220)
- Cache Hierarchy:
    - L1d: 32K
    - L1i: 32K
    - L2: 256K
    - L3: 51M
- Operating System: Rocky Linux 8.7 (Green Obsidian)

### Floating Point Benchmark

|               **Operation**               	| **MulBigDouble** 	| **DivBigDouble** 	| **MulSmallDouble** 	| **DivSmallDouble** 	| **AddBigDouble** 	| **SubBigDouble** 	| **AddSmallDouble** 	| **SubSmallDouble** 	|
|:-----------------------------------------:	|:----------------:	|:----------------:	|:------------------:	|:------------------:	|:----------------:	|:----------------:	|:------------------:	|:------------------:	|
| **Explorer Cluster Average Latency (ms)** 	|        45        	|        143       	|         69         	|         158        	|        45        	|         69        	|         69         	|         75         	|
|      **Vector Average Latency (ms)**      	|        167        	|        350       	|         184         	|         260        	|        111        	|        111        	|         106         	|         100         	|

### Integer Benchmark

|               **Operation**               	| **MulBigInt** 	| **DivBigInt** 	| **MulSmallInt** 	| **DivSmallInt** 	| **AddBigInt** 	| **SubBigInt** 	| **AddSmallInt** 	| **SubSmallInt** 	|
|:-----------------------------------------:	|:-------------:	|:-------------:	|:---------------:	|:---------------:	|:-------------:	|:-------------:	|:---------------:	|:---------------:	|
| **Explorer Cluster Average Latency (ms)** 	|       46      	|      843      	|        41       	|       935       	|       50      	|       38      	|        41       	|        41       	|
|      **Vector Average Latency (ms)**      	|       112      	|      1309     	|        64       	|       1110      	|       52      	|       48      	|        56       	|        57       	|

### Memory Intensive Benchmark

|               **Operation**              	| Forward Sum 	| Reverse Sum 	| Random Sum 	|
|:----------------------------------------:	|:-----------:	|:-----------:	|:----------:	|
| **Explorer Cluster Average Latency (s)** 	|  21.444922  	|   5.386331  	| 174.070636 	|
|      **Vector Average Latency (s)**      	|  18.711402  	|   4.402960  	| 105.419321 	|


Overall, there were no outlier runs of each benchmark (i.e. every benchmark run had consistently the same latency values with very little variance).


## Part (b)

The Explorer Cluster had better performance compared to the Vector system for 2/3 benchmarks. The Explorer Cluster had a lower average latency for the floating-point and integer benchmarks, while the Vector system had a lower average latency for the memory-intensive benchmark.

In the cases where Explorer outperformed Vector, the most likely cause is because 

In the case where Vector outperformed Explorer, the most likely cause is because 
 

## Part (c)



## Part (d)



## Part (e)



