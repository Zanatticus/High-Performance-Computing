# Problem 3 (30 Points)

Graph coloring involves coloring a graph G (V, E) such that no two adjacent vertices have the same color. This algorithm is used in several important applications, including register allocation in compilers, sparse matrix ordering and VLSI routing. The goal is to use the least number of colors to color the graph. An exact solution to graph coloring is NP-hard. You don’t need to obtain the optimal solution (i.e., you can use a greedy approach). Make sure to test your solution with different graphs that you will generate. Describe how you tested your implementation.

a.) There are a few approaches you can take to solve this problem. Develop your solution using OpenMP.
b.) Evaluate the strong scaling and weak scaling performance of your implementation. Compare the runtime taken to the algorithmic complexity (big-O) of the algorithm you have chosen for your implementation. 

* Written answers to the questions should be included in your homework 2 write-up in pdf format. You should include your C/C++ program and the README file in the zip file submitted.

## Part (a)

There were two algorithms that I implemented for this problem. The first one was a sequential, non-parallel program to color the graph. The second one was a parallel program that used OpenMP to color the graph. The parallel program was implemented using the Gebremedhin-Manne algorithm. I considered using the Jones-Plassman algorithm, but I ran into some issues with the implementation. I used the following sources as reference to implement the Gebremedhin-Manne algorithm:
- https://www.osti.gov/biblio/1246285
- https://www.osti.gov/servlets/purl/1246285

The program was run using the following resources:
- Rho on Vector
- 80 cores
- Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
- Actual CPU speed: 3015.278 MHz

## Part (b)

### Weak Scaling
#### *Weak Scaling Results (Low Neighbor Density, Low Vertex Count)*

|   Number of Vertices   	|         50        	|         50        	|         50        	|         50        	|         50        	|         50        	|         50        	|         50        	|
|:----------------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|
|    Number of Threads   	|         1         	|         2         	|         4         	|         8         	|         16        	|         32        	|         64        	|        128        	|
| Sequential Latency (s) 	| 0.000147475000000 	| 0.000123431000000 	| 0.000219751000000 	| 0.000135177000000 	| 0.000131705000000 	| 0.000161019000000 	| 0.000193656000000 	| 0.000216418000000 	|
| Parallel Latency (s)   	| 0.000247144000000 	| 0.000408772000000 	| 0.000553714000000 	| 0.000990607000000 	| 0.001505155000000 	| 0.002321689000000 	| 0.004029204000000 	| 0.009183830000000 	|

#### *Weak Scaling Results (Low Neighbor Density, High Vertex Count)*

|   Number of Vertices   	|        5000       	|        5000       	|        5000       	|        5000       	|        5000       	|        5000       	|        5000       	|        5000       	|
|:----------------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|
|    Number of Threads   	|         1         	|         2         	|         4         	|         8         	|         16        	|         32        	|         64        	|        128        	|
| Sequential Latency (s) 	| 0.019141204000000 	| 0.020462251000000 	| 0.018466747000000 	| 0.017000681000000 	| 0.021097227000000 	| 0.021336284000000 	| 0.021267299000000 	| 0.017734126000000 	|
| Parallel Latency (s)   	| 0.027271996000000 	| 0.023330019000000 	| 0.019376828000000 	| 0.023762920000000 	| 0.029977729000000 	| 0.030018244000000 	| 0.036650816000000 	| 0.046117249000000 	|

#### *Weak Scaling Results (High Neighbor Density, Low Vertex Count)*

|   Number of Vertices   	|         50        	|         50        	|         50        	|         50        	|         50        	|         50        	|         50        	|         50        	|
|:----------------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|
|    Number of Threads   	|         1         	|         2         	|         4         	|         8         	|         16        	|         32        	|         64        	|        128        	|
| Sequential Latency (s) 	| 0.001143292000000 	| 0.001101315000000 	| 0.001098681000000 	| 0.001091475000000 	| 0.001067719000000 	| 0.001267272000000 	| 0.001109860000000 	| 0.001129597000000 	|
| Parallel Latency (s)   	| 0.002787945000000 	| 0.002365725000000 	| 0.002382072000000 	| 0.002818613000000 	| 0.003771817000000 	| 0.005091095000000 	| 0.008057402000000 	| 0.019890172000000 	|

#### *Weak Scaling Results (High Neighbor Density, High Vertex Count)*

|   Number of Vertices   	|        5000        	|        5000        	|        5000        	|        5000        	|        5000        	|        5000        	|        5000        	|        5000        	|
|:----------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|:------------------:	|
|    Number of Threads   	|          1         	|          2         	|          4         	|          8         	|         16         	|         32         	|         64         	|         128        	|
| Sequential Latency (s) 	| 19.514392294000000 	| 19.514392294000000 	| 19.514392294000000 	| 19.514392294000000 	| 19.514392294000000 	| 19.514392294000000 	| 19.514392294000000 	| 19.514392294000000 	|
| Parallel Latency (s)   	| 62.028325002999999 	| 44.920666076000003 	| 34.433435312000000 	| 30.521134642000000 	| 29.908935541000002 	| 28.246148859000002 	| 29.340540300000001 	| 30.125050165000001 	|


### Strong Scaling
#### *Strong Scaling Results (Low Neighbor Density)*

|   Number of Vertices   	|         50        	|        100        	|        200        	|        400        	|        800        	|        1600       	|        3200       	|        6400       	|
|:----------------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|
|    Number of Threads   	|         1         	|         2         	|         4         	|         8         	|         16        	|         32        	|         64        	|        128        	|
| Sequential Latency (s) 	| 0.000119960000000 	| 0.000250363000000 	| 0.000573653000000 	| 0.001131919000000 	| 0.002555596000000 	| 0.004861588000000 	| 0.010552855000000 	| 0.022072515000000 	|
| Parallel Latency (s)   	| 0.000282503000000 	| 0.000616967000000 	| 0.000988593000000 	| 0.002338956000000 	| 0.005553444000000 	| 0.012509572000000 	| 0.024726601000000 	| 0.053921053000000 	|

#### *Strong Scaling Results (High Neighbor Density)*

|   Number of Vertices   	|         50        	|        100        	|        200        	|        400        	|        800        	|        1600       	|        3200        	|        6400        	|
|:----------------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:-----------------:	|:------------------:	|:------------------:	|
|    Number of Threads   	|         1         	|         2         	|         4         	|         8         	|         16        	|         32        	|         64         	|         128        	|
| Sequential Latency (s) 	| 0.001143021000000 	| 0.004446017000000 	| 0.018232367000000 	| 0.080539635000000 	| 0.356109796000000 	| 1.616247782000000 	|  7.312984351000000 	| 32.849850363000002 	|
| Parallel Latency (s)   	| 0.002776892000000 	| 0.010003139000000 	| 0.033789319000000 	| 0.144835970000000 	| 0.613463176000000 	| 2.670794211000000 	| 11.785003091000000 	| 49.053064049000000 	|

### Algorithmic Complexity and Observations

#### *Algorithmic Complexity*
Let `n` be the number of vertices and `m` be the number of edges in the graph. The Gebremedhin-Manne algorithm has the following complexity:
1. The loop that initializes priorities run in **`O(n)`**
2. The outer while loop `while (!vertex_list.empty())` iterates over the graph until all vertices are colored, with the worst case scenario being **`O(n)`** if only one node is colored per iteration.
3. The independent set selection which has each vertex chechking its neighbors to determine if it has the highest priority runs in **`O(m)`**
4. Assigning colors to the vertices runs in **`O(n)`**
5. Updating the vertex list runs in **`O(n)`**

Since the outer while loop runs in **`O(n)`**, and the independent set selection & coloring step runs in **`O(m)`** and **`O(n)`**, respectively, the overall complexity becomes **`O(nm + n^2)`**.

Considering that I collected data for sparse and dense graphs, it's important to note that the complexity can simplify as follows (applying **`m = O(n)`** for sparse graphs and **`m = O(n^2)`** for dense graphs):

- Sparse Graphs: **`O(n^2)`**
- Dense Graphs: **`O(n^3)`**

#### *Observations*
- The sequential algorithm always performed better than the parallel algorithm. Even though the sequential algorithm had a algorithmic complexity of **`O(n^2)`**, it was able to color the graph faster than the parallel algorithm since there was overall less overhead and computations required.
- The parallel algorithm performed much better on sparse graphs than dense graphs. This is because the parallel algorithm had to perform more computations on dense graphs, which led to more overhead and longer runtimes. This also aligns with the algorithmic complexity of the parallel algorithm on dense graphs, which is **`O(n^3)`** as opposed to **`O(n^2)`** on sparse graphs.
- Looking at weak scaling, the parallel algorithm did not scale well for low neighbor densities, and exhibited some slight positive scaling for high neghbor densities with low vertex counts. The most notable scaling, however, was for high neighbor densities with high vertex counts, where the parallel algorithm scaled well up until it reached a barrier at 8 threads.
- Looking at strong scaling, the parallel algorithm did not scale well for both low and high neighbor densities.

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```