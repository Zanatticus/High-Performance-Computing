# Problem 1 (30 Points)

In this problem you will develop two different implementations of the computation of pi (p) numerically using pthreads and OpenMP. Then you will compare them in terms of scalability and accuracy. Undergraduates/PLUS-One students only need to complete parts b and c of this problem for full credit, though can complete part a (i.e., the pthreads implementation) to receive 20 points of quiz extra credit. In this problem, you will develop a program that computes the value of pi. You can refer to the following video that suggests a way to compute this using Monte Carlo simulation: https://www.youtube.com/watch?v=M34TO71SKGk&ab_channel=PhysicsGirl 

This is not the most efficient way to compute pi, though will provide you with a baseline. There are better ways to compute the value. Select a more efficient method (e.g., Leibniz’s formula) and compare it to the Monte Carlo method in terms of convergence rate, assessing the accuracy of the value of pi as a function of runtime. 

a. Evaluate the speedup that you achieve by using pthreads and multiple cores. You are free to use as many threads as you like. The program should take two input parameters, the  number of threads and the number of “darts” thrown. Your program should print out the time required to compute pi and the final value of pi. Make sure to document the system you are running on and the number of hardware threads available. 

b. Now develop the same program using OpenMP. Repeat all of the steps requested in part a. 

c. Now compare the two implementations in terms of strong and weak scaling, where the number of Monte Carlo simulations (i.e., “darts” thrown) is used to assess weak scaling. Make sure you plot your results.

* Written answers to the questions should be included in your homework 2 write-up in pdf format. You should include your C/C++ program and the README file in the zip file submitted.

## Part (a)


## Part (b)


## Part (c)


## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```