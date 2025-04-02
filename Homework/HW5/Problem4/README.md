# Problem 4 (50 Points Extra Credit)
Let’s revisit computing the value of pi, but this time we will use a series. For instance, we provide you with code for the Leibniz’s series, developed by Jose Cintra. Implement this series on the GPU, allowing the user to enter the number of iterations. Make sure to develop an efficient computation of this kernel that utilizes the parallelism provided on the GPU. Then modify this code to use single precision math. Show results for at least 10 different number of iterations of the series and discuss how precision plays a role in the rate of convergence.

## Answers

#### Iterations = \( 2^{19} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 524288
Execution Time (CPU): 0.0007383510 seconds
Approximation of PI: 3.141593933105469

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 524288
Execution Time (CPU): 0.0017269360 seconds
Approximation of PI: 3.141590746241052

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 524288
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 512
Execution Time (including device sync & copy): 0.0002171550 seconds
Approximation of PI: 3.141583204269409

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 524288
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 512
Execution Time (including device sync & copy): 0.0001700230 seconds
Approximation of PI: 3.141590746241157
```

#### Iterations = \( 2^{20} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 1048576
Execution Time (CPU): 0.0012225380 seconds
Approximation of PI: 3.141595363616943

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 1048576
Execution Time (CPU): 0.0016188070 seconds
Approximation of PI: 3.141591699915466

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 1048576
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 1024
Execution Time (including device sync & copy): 0.0002847950 seconds
Approximation of PI: 3.141583681106567

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 1048576
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 1024
Execution Time (including device sync & copy): 0.0002340070 seconds
Approximation of PI: 3.141591699915472
```

#### Iterations = \( 2^{21} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 2097152
Execution Time (CPU): 0.0022777510 seconds
Approximation of PI: 3.141596078872681

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 2097152
Execution Time (CPU): 0.0030535050 seconds
Approximation of PI: 3.141592176752559

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 2097152
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 2048
Execution Time (including device sync & copy): 0.0003313670 seconds
Approximation of PI: 3.141583204269409

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 2097152
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 2048
Execution Time (including device sync & copy): 0.0002757990 seconds
Approximation of PI: 3.141592176752630
```

#### Iterations = \( 2^{22} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 4194304
Execution Time (CPU): 0.0042974750 seconds
Approximation of PI: 3.141596555709839

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 4194304
Execution Time (CPU): 0.0060143200 seconds
Approximation of PI: 3.141592415171123

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 4194304
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 4096
Execution Time (including device sync & copy): 0.0004342760 seconds
Approximation of PI: 3.141583204269409

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 4194304
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 4096
Execution Time (including device sync & copy): 0.0004016280 seconds
Approximation of PI: 3.141592415171216
```

#### Iterations = \( 2^{23} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 8388608
Execution Time (CPU): 0.0081314380 seconds
Approximation of PI: 3.141596794128418

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 8388608
Execution Time (CPU): 0.0119504260 seconds
Approximation of PI: 3.141592534380551

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 8388608
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 8192
Execution Time (including device sync & copy): 0.0006222660 seconds
Approximation of PI: 3.141583442687988

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 8388608
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 8192
Execution Time (including device sync & copy): 0.0006328600 seconds
Approximation of PI: 3.141592534380528
```

#### Iterations = \( 2^{24} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 16777216
Execution Time (CPU): 0.0162674930 seconds
Approximation of PI: 3.141596794128418

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 16777216
Execution Time (CPU): 0.0236003180 seconds
Approximation of PI: 3.141592593985150

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 16777216
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 16384
Execution Time (including device sync & copy): 0.0010693350 seconds
Approximation of PI: 3.141583204269409

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 16777216
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 16384
Execution Time (including device sync & copy): 0.0011193310 seconds
Approximation of PI: 3.141592593985187
```

#### Iterations = \( 2^{25} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 33554432
Execution Time (CPU): 0.0310713740 seconds
Approximation of PI: 3.141596794128418

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 33554432
Execution Time (CPU): 0.0472041090 seconds
Approximation of PI: 3.141592623788183

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 33554432
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 32768
Execution Time (including device sync & copy): 0.0019356070 seconds
Approximation of PI: 3.141583442687988

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 33554432
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 32768
Execution Time (including device sync & copy): 0.0020704320 seconds
Approximation of PI: 3.141592623787523
```

#### Iterations = \( 2^{26} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 67108864
Execution Time (CPU): 0.0754622350 seconds
Approximation of PI: 3.141596794128418

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 67108864
Execution Time (CPU): 0.1159866040 seconds
Approximation of PI: 3.141592638688858

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 67108864
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 65536
Execution Time (including device sync & copy): 0.0037290120 seconds
Approximation of PI: 3.141583204269409

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 67108864
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 65536
Execution Time (including device sync & copy): 0.0040331290 seconds
Approximation of PI: 3.141592638688680
```

#### Iterations = \( 2^{27} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 134217728
Execution Time (CPU): 0.1259569390 seconds
Approximation of PI: 3.141596794128418

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 134217728
Execution Time (CPU): 0.1903001710 seconds
Approximation of PI: 3.141592646138478

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 134217728
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 131072
Execution Time (including device sync & copy): 0.0071862980 seconds
Approximation of PI: 3.141582965850830

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 134217728
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 131072
Execution Time (including device sync & copy): 0.0078439080 seconds
Approximation of PI: 3.141592646139258
```

#### Iterations = \( 2^{28} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 268435456
Execution Time (CPU): 0.2505057560 seconds
Approximation of PI: 3.141596794128418

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 268435456
Execution Time (CPU): 0.3835540900 seconds
Approximation of PI: 3.141592649864012

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 268435456
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 262144
Execution Time (including device sync & copy): 0.0141801060 seconds
Approximation of PI: 3.141582965850830

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 268435456
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 262144
Execution Time (including device sync & copy): 0.0154727100 seconds
Approximation of PI: 3.141592649864482
```

#### Iterations = \( 2^{29} \)
```
==========================================
 CPU Leibniz Float Pi Computation Results
==========================================

Total iterations: 536870912
Execution Time (CPU): 0.6765264140 seconds
Approximation of PI: 3.141596794128418

===========================================
 CPU Leibniz Double Pi Computation Results
===========================================

Total iterations: 536870912
Execution Time (CPU): 0.9324062030 seconds
Approximation of PI: 3.141592651726695

===========================================
 CUDA Leibniz Float Pi Computation Results
===========================================

Total iterations: 536870912
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 524288
Execution Time (including device sync & copy): 0.0281990680 seconds
Approximation of PI: 3.141583442687988

============================================
 CUDA Leibniz Double Pi Computation Results
============================================

Total iterations: 536870912
Block Size (Threads Per Block): 1024 | Grid Size (Number of Blocks): 524288
Execution Time (including device sync & copy): 0.0309165270 seconds
Approximation of PI: 3.141592651726598
```

### Discussion

Comparing the usage of floats versus doubles, we can clearly see that as the total number of iterations increase, the accuracy of the approximation of pi increases. However, the float implementation plateaus earlier than the double implementation, as the float implementation is simply less precise. Before this plateau point, both implementations converge at the same rate, but because the double precision can represent more decimal places, it is able to converge to a more accurate approximation of pi with more iterations. Performance wise, both implementations take roughly the same amount of time on a GPU, but the speedup of using a CUDA optimized program as opposed to running it on a CPU becomes evident when using a substantial amount of iterations.


## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```