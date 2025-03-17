# Problem 1 (30 Points)

Develop a simple MPI program using C/C++ and OpenMPI that uses at least 4 nodes on Explorer and utilizes at least 16 processes on each node (a minimum of 64 processes in total). You are suggested to use the sample batch script provided on Canvas for specifying your OpenMPI configuration and running your program.

a.) Start with an integer variable that you will pass to each process, where process 1 prints the value, increments the value by 1, and sends it to process 2. Process 2 prints the value, then increments the value by 1 and sends it to process 3. Repeat this for all 64 processes. When performing printing, print both the integer value, as well as identify which process is printing, and on which node this process is running on.

b.) Next, extend your program such that once the value gets to 64, decrement the value by 2 in each step, and continue to print out the value until the decremented value is zero.

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```