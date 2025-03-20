# Problem 1 (30 Points)

Develop a simple MPI program using C/C++ and OpenMPI that uses at least 4 nodes on Explorer and utilizes at least 16 processes on each node (a minimum of 64 processes in total). You are suggested to use the sample batch script provided on Canvas for specifying your OpenMPI configuration and running your program.

a.) Start with an integer variable that you will pass to each process, where process 1 prints the value, increments the value by 1, and sends it to process 2. Process 2 prints the value, then increments the value by 1 and sends it to process 3. Repeat this for all 64 processes. When performing printing, print both the integer value, as well as identify which process is printing, and on which node this process is running on.

b.) Next, extend your program such that once the value gets to 64, decrement the value by 2 in each step, and continue to print out the value until the decremented value is zero.

## Part (a)

```
Process 0 on node c0441 started with counter: 1
Process 1 on node c0441 received counter: 2
Process 2 on node c0441 received counter: 3
Process 3 on node c0441 received counter: 4
Process 4 on node c0441 received counter: 5
Process 5 on node c0441 received counter: 6
Process 6 on node c0441 received counter: 7
Process 7 on node c0441 received counter: 8
Process 8 on node c0441 received counter: 9
Process 9 on node c0441 received counter: 10
Process 10 on node c0441 received counter: 11
Process 11 on node c0441 received counter: 12
Process 12 on node c0441 received counter: 13
Process 13 on node c0441 received counter: 14
Process 14 on node c0441 received counter: 15
Process 15 on node c0441 received counter: 16
Process 16 on node c0442 received counter: 17
Process 17 on node c0442 received counter: 18
Process 18 on node c0442 received counter: 19
Process 19 on node c0442 received counter: 20
Process 20 on node c0442 received counter: 21
Process 21 on node c0442 received counter: 22
Process 22 on node c0442 received counter: 23
Process 23 on node c0442 received counter: 24
Process 24 on node c0442 received counter: 25
Process 25 on node c0442 received counter: 26
Process 26 on node c0442 received counter: 27
Process 27 on node c0442 received counter: 28
Process 28 on node c0442 received counter: 29
Process 29 on node c0442 received counter: 30
Process 30 on node c0442 received counter: 31
Process 31 on node c0442 received counter: 32
Process 32 on node c0443 received counter: 33
Process 33 on node c0443 received counter: 34
Process 34 on node c0443 received counter: 35
Process 35 on node c0443 received counter: 36
Process 36 on node c0443 received counter: 37
Process 37 on node c0443 received counter: 38
Process 38 on node c0443 received counter: 39
Process 39 on node c0443 received counter: 40
Process 40 on node c0443 received counter: 41
Process 41 on node c0443 received counter: 42
Process 42 on node c0443 received counter: 43
Process 43 on node c0443 received counter: 44
Process 44 on node c0443 received counter: 45
Process 45 on node c0443 received counter: 46
Process 46 on node c0443 received counter: 47
Process 47 on node c0443 received counter: 48
Process 48 on node c0444 received counter: 49
Process 49 on node c0444 received counter: 50
Process 50 on node c0444 received counter: 51
Process 51 on node c0444 received counter: 52
Process 52 on node c0444 received counter: 53
Process 53 on node c0444 received counter: 54
Process 54 on node c0444 received counter: 55
Process 55 on node c0444 received counter: 56
Process 56 on node c0444 received counter: 57
Process 57 on node c0444 received counter: 58
Process 58 on node c0444 received counter: 59
Process 59 on node c0444 received counter: 60
Process 60 on node c0444 received counter: 61
Process 61 on node c0444 received counter: 62
Process 62 on node c0444 received counter: 63
Process 63 on node c0444 received counter: 64
```

## Part (b)

```
```

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```