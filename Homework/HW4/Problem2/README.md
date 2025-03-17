# Problem 2 (30 Points)

Develop a parallel histogramming program using C/C++ and OpenMPI. A histogram is used to summarize the distribution of data values in a data set. The most common form of histogramming splits the data range into equal-sized bins. For each bin, the number of data values in the data set that falls into that class are totaled. Your input to this program will be integers in the range 1-100,000 (use a random number generator that first generates the numbers). Your input data set should contain 8 million integers. You will vary the number of bins. You need to figure out how to assign bins to OpenMPI processes. You are suggested to use the sample batch script provided on Canvas for specifying your OpenMPI configuration and running your program (you will need to change some of the job parameters).

a.) Assume there are 128 bins. Perform binning across nodes and processes using OpenMPI, and then perform a reduction on the lead node, combining your partial results. Run this on 2, 4 and 8 nodes on Explorer. Your program should print out the number of values that fall into each bin. Compare the performance between running this on 2, 4 and 8 nodes. Comment on the differences.

b.) For this part, assume you have 32 bins. Perform binning on each process using OpenMPI, and then perform a reduction on the lead node, combining your partial results. Run this on 2 and 4 nodes on Explorer. Your program should print out the number of values that fall into each bin. Compare the performance between running this on 2 and 4 nodes. Comment on the differences.

c.) Compare the performance measured in parts a.) and b.). Try to explain why one is faster than the other and run additional experiments to support your claims.

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```