# Problem 1 (30 Points)

In this problem you will develop two different implementations of the computation of pi (p) numerically using pthreads and OpenMP. Then you will compare them in terms of scalability and accuracy. Undergraduates/PLUS-One students only need to complete parts b and c of this problem for full credit, though can complete part a (i.e., the pthreads implementation) to receive 20 points of quiz extra credit. In this problem, you will develop a program that computes the value of pi. You can refer to the following video that suggests a way to compute this using Monte Carlo simulation: https://www.youtube.com/watch?v=M34TO71SKGk&ab_channel=PhysicsGirl 

This is not the most efficient way to compute pi, though will provide you with a baseline. There are better ways to compute the value. Select a more efficient method (e.g., Leibniz’s formula) and compare it to the Monte Carlo method in terms of convergence rate, assessing the accuracy of the value of pi as a function of runtime. 

a. Evaluate the speedup that you achieve by using pthreads and multiple cores. You are free to use as many threads as you like. The program should take two input parameters, the  number of threads and the number of “darts” thrown. Your program should print out the time required to compute pi and the final value of pi. Make sure to document the system you are running on and the number of hardware threads available. 

b. Now develop the same program using OpenMP. Repeat all of the steps requested in part a. 

c. Now compare the two implementations in terms of strong and weak scaling, where the number of Monte Carlo simulations (i.e., “darts” thrown) is used to assess weak scaling. Make sure you plot your results.

*Written answers to the questions should be included in your homework 2 write-up in pdf format. You should include your C/C++ program and the README file in the zip file submitted.

## Part (a)

**Data was collected on the following system:**
Vector Node Xi, Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz, 80 cores, running at 3600 MHz

### Monte Carlo Method for Approximating PI using Pthreads

| Number of "Darts" 	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|
|:-----------------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|
|      Threads      	|      1      	|      2      	|      4      	|      8      	|      16     	|      32     	|      64     	|     128     	|
|    Latency (s)    	| 0.161186879 	| 0.085395628 	| 0.042522278 	| 0.024704975 	| 0.012632099 	| 0.008237528 	| 0.006388601 	| 0.007657536 	|
|    Value of Pi    	|   3.139588  	|   3.141412  	|   3.13996   	|   3.142236  	|   3.141536  	|   3.143572  	|   3.139644  	|   3.141084  	|
|      Speedup*     	|      1      	| 1.887530811 	| 3.790645435 	| 6.524470436 	| 12.76010258 	| 19.56738466 	| 25.23038753 	| 21.04944449 	|

<p align="center">Table 1: Weak Scaling for Monte Carlo Method to Approximate PI using Pthreads. Speedup is compared against single-threaded performance.</p>

| Number of "Darts" 	|   1000000   	|    2000000   	|    4000000   	|    8000000   	|   16000000   	|   32000000   	|   64000000   	|  128000000  	|
|:-----------------:	|:-----------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:-----------:	|
|      Threads      	|      1      	|       2      	|       4      	|       8      	|      16      	|      32      	|      64      	|     128     	|
|    Latency (s)    	| 0.150774269 	|  0.164094394 	|  0.174795918 	|  0.180297205 	|  0.186066276 	|  0.188868483 	|  0.268806894 	| 0.484494829 	|
|    Value of Pi    	|   3.142096  	|   3.140892   	|   3.142606   	|   3.141271   	|  3.14151775  	|  3.141394125 	|  3.141726375 	| 3.141656469 	|
|      Speedup*     	|      1      	| 0.9188264469 	| 0.8625731695 	| 0.8362540562 	| 0.8103256121 	| 0.7983029598 	| 0.5609017937 	| 0.311198923 	|

<p align="center">Table 2: Strong Scaling for Monte Carlo Method to Approximate PI using Pthreads. Speedup is compared against single-threaded performance.</p>

### Leibniz's Formula for Approximating PI using Pthreads

| Number of "Darts" 	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|
|:-----------------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|
|      Threads      	|      1      	|      2      	|      4      	|      8      	|      16     	|      32     	|      64     	|     128     	|
|    Latency (s)    	| 0.004975266 	| 0.002922062 	| 0.003218681 	| 0.003444885 	| 0.003529307 	| 0.003012068 	| 0.003074669 	| 0.004674836 	|
|    Value of Pi    	| 3.141595125 	| 3.141594648 	| 3.141594648 	| 3.141595602 	| 3.141595364 	| 3.141595125 	| 3.141597986 	| 3.141596317 	|
|      Speedup*     	|      1      	| 1.702655864 	| 1.545746845 	| 1.444247341 	| 1.409700545 	| 1.618146864 	| 1.618146864 	| 1.064265356 	|

<p align="center">Table 3: Weak Scaling for Leibniz's Formula to Approximate PI using Pthreads. Speedup is compared against single-threaded performance.</p>

| Number of "Darts" 	|   1000000   	|    2000000   	|    4000000   	|    8000000   	|    16000000   	|    32000000   	|   64000000   	|   128000000   	|
|:-----------------:	|:-----------:	|:------------:	|:------------:	|:------------:	|:-------------:	|:-------------:	|:------------:	|:-------------:	|
|      Threads      	|      1      	|       2      	|       4      	|       8      	|       16      	|       32      	|      64      	|      128      	|
|    Latency (s)    	| 0.005019043 	|  0.006625896 	|  0.013584704 	|  0.027963448 	|  0.062699994  	|  0.088700306  	|  0.141299254 	|  0.178024105  	|
|    Value of Pi    	| 3.141595125 	|  3.141595602 	|  3.14159584  	|  3.14159584  	|   3.14159584  	|   3.14159584  	|  3.14159584  	|   3.14159584  	|
|      Speedup*     	|      1      	| 0.7574889494 	| 0.3694628164 	| 0.1794858417 	| 0.08004854036 	| 0.05658428055 	| 0.0355206617 	| 0.02819305285 	|

<p align="center">Table 4: Strong Scaling for Leibniz's Formula to Approximate PI using Pthreads. Speedup is compared against single-threaded performance.</p>

### Observations
Both the Monte Carlo and Leibniz's Formula methods for approximating PI using Pthreads show a decrease in speedup as the number of threads increases while the number of "darts" also increases. On the other hand, when the number of "darts" is fixed and the number of threads increases, the speedup for both methods increases, with the Monte Carlo method exhibiting greater speedups compared to the Leibniz's Formula method when using their respective single-threaded performance as a baseline.

### Miscellaneous Observations

When running Leibniz's Formula with a depth of 5 and on one thread:

> SUM: 1  
> SUM: 0.6666666**865**348815918  
> SUM: 0.8666666**746**1395263672  
> SUM: 0.7238095**402**717590332  
> SUM: 0.8349206**447**6013183594  
> Approximated value of PI: 3.3396825**790405273438**

The actual values (according to a calculator) should be:

> SUM: 1  
> SUM: 0.6666666**667**  
> SUM: 0.8666666**667**  
> SUM: 0.7238095**238**  
> SUM: 0.8349206**349**  
> Approximated value of PI: 3.3396825**397**  

This clearly showcases the inaccuracies and lack of precision with floating point data types on computers. There is a limit to precision!

## Part (b)

### Monte Carlo Method for Approximating PI using OpenMP

| Number of "Darts" 	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|
|:-----------------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|
|      Threads      	|      1      	|      2      	|      4      	|      8      	|      16     	|      32     	|      64     	|     128     	|
|    Latency (s)    	| 0.148587449 	|  0.07603915 	| 0.039739611 	| 0.021159209 	|  0.01315859 	| 0.007108923 	| 0.006688395 	| 0.013466174 	|
|    Value of Pi    	|   3.141076  	|   3.141168  	|   3.14048   	|   3.140328  	|   3.140652  	|   3.142184  	|   3.142592  	|   3.141308  	|
|      Speedup*     	|      1      	| 1.954091399 	| 3.739026258 	| 7.022353671 	| 11.29204945 	| 20.90154149 	| 22.21571079 	| 11.03412513 	|

<p align="center">Table 5: Weak Scaling for Monte Carlo Method to Approximate PI using OpenMP. Speedup is compared against single-threaded performance.</p>

| Number of "Darts" 	|   1000000   	|    2000000   	|    4000000   	|    8000000   	|   16000000   	|   32000000   	|   64000000   	|   128000000  	|
|:-----------------:	|:-----------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|
|      Threads      	|      1      	|       2      	|       4      	|       8      	|      16      	|      32      	|      64      	|      128     	|
|    Latency (s)    	| 0.140900122 	|  0.176763486 	|  0.17684653  	|  0.184477723 	|  0.191892187 	|  0.18603232  	|  0.268400492 	|  0.480545591 	|
|    Value of Pi    	|   3.141396  	|   3.141222   	|   3.141653   	|   3.1421595  	|   3.141157   	|  3.14160825  	|  3.14173675  	|  3.141699094 	|
|      Speedup*     	|      1      	| 0.7971110165 	| 0.7967367072 	| 0.7637785187 	| 0.7342671122 	| 0.7573959299 	| 0.5249622344 	| 0.2932086458 	|

<p align="center">Table 6: Strong Scaling for Monte Carlo Method to Approximate PI using OpenMP. Speedup is compared against single-threaded performance.</p>

### Leibniz's Formula for Approximating PI using OpenMP

| Number of "Darts" 	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|   1000000   	|
|:-----------------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|:-----------:	|
|      Threads      	|      1      	|      2      	|      4      	|      8      	|      16     	|      32     	|      64     	|     128     	|
|    Latency (s)    	| 0.004514544 	| 0.002829124 	| 0.001672591 	| 0.001137956 	| 0.001004632 	| 0.001421967 	| 0.001421967 	| 0.001421967 	|
|    Value of Pi    	| 3.141595125 	| 3.141594648 	| 3.141594648 	| 3.141595364 	| 3.141595125 	| 3.141595602 	| 3.141595602 	| 3.141595602 	|
|      Speedup*     	|      1      	| 1.595739176 	| 2.699132065 	| 3.967239507 	| 4.493729047 	| 3.174858488 	| 3.174858488 	| 3.174858488 	|

<p align="center">Table 7: Weak Scaling for Leibniz's Formula to Approximate PI using OpenMP. Speedup is compared against single-threaded performance.</p>

| Number of "Darts" 	|   1000000   	|   2000000   	|    4000000   	|    8000000   	|   16000000   	|   32000000   	|   64000000   	|   128000000  	|
|:-----------------:	|:-----------:	|:-----------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|:------------:	|
|      Threads      	|      1      	|      2      	|       4      	|       8      	|      16      	|      32      	|      64      	|      128     	|
|    Latency (s)    	| 0.004886531 	| 0.005431097 	|  0.005450916 	|  0.005441585 	|  0.00673029  	|  0.007049581 	|  0.008907675 	|  0.026552492 	|
|    Value of Pi    	| 3.141595125 	| 3.141595602 	|  3.14159584  	|  3.14159584  	|  3.14159584  	|  3.141596079 	|  3.14159584  	|  3.141596079 	|
|      Speedup*     	|      1      	| 0.899731859 	| 0.8964605215 	| 0.8979977341 	| 0.7260505862 	| 0.6931661612 	| 0.5485753578 	| 0.1840328584 	|

<p align="center">Table 8: Strong Scaling for Leibniz's Formula to Approximate PI using OpenMP. Speedup is compared against single-threaded performance.</p>

### Observations
Both the Monte Carlo and Leibniz's Formula methods for approximating PI using OpenMP show a decrease in speedup as the number of threads increases while the number of "darts" also increases. On the other hand, when the number of "darts" is fixed and the number of threads increases, the speedup for both methods increases, with the Monte Carlo method exhibiting greater speedups compared to the Leibniz's Formula method when using their respective single-threaded performance as a baseline.

## Part (c)

![Monte Carlo Weak Scaling Speedup vs. Leibniz Weak Scaling Speedup Using Pthreads](Monte%20Carlo%20Weak%20Scaling%20Speedup%20vs.%20Leibniz%20Weak%20Scaling%20Speedup%20Using%20Pthreads.png)

![Monte Carlo Strong Scaling Speedup vs. Leibniz Strong Scaling Speedup Using Pthreads](Monte%20Carlo%20Strong%20Scaling%20Speedup%20vs.%20Leibniz%20Strong%20Scaling%20Speedup%20Using%20Pthreads.png)

![Monte Carlo Weak Scaling Speedup vs. Leibniz Weak Scaling Speedup Using OpenMP](Monte%20Carlo%20Weak%20Scaling%20Speedup%20vs.%20Leibniz%20Weak%20Scaling%20Speedup%20Using%20OpenMP.png)

![Monte Carlo Strong Scaling Speedup vs. Leibniz Strong Scaling Speedup Using OpenMP](Monte%20Carlo%20Strong%20Scaling%20Speedup%20vs.%20Leibniz%20Strong%20Scaling%20Speedup%20Using%20OpenMP.png)


## Miscellaneous
- The program was compiled and run using the following command within the respective makefile directory for each implementation:
```make```