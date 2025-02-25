# Problem 1 (20 Points)

In this problem, you will utilize the IEEE 754 format and evaluate the performance implications of using floats versus doubles in a computation.

a.) Compute f(x) = sin(x) using a Taylor series expansion. To refresh your memory:

\[
\sin(x) = \sum_{0}^{\infty} \frac{(-1)^n}{(2n+1)!} x^{2n+1}
\]

\[
\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \frac{x^9}{9!} \dots
\]

You should select the number of terms you want to compute (but at least **10 terms**). Compute \( \sin(x) \) for **4 different values**, though be careful not to use too large a value. Generate two versions of your code, first defining `x` and `sin(x)` to use **floats (SP)**, and second, defining them as **doubles (DP)**. Discuss any differences you find in your results for \( f(x) \). You should provide an in-depth discussion on the results you get and the reasons for any differences.

b.) Provide both IEEE 754 single and double precision representations for the following numbers:
- **2.1**
- **6300**
- **-1.044**

## Part (a)


## Part (b)

<span style="color:red">RED</span> = Sign Bit
<span style="color:green">GREEN</span> = Exponent Bits
<span style="color:blue">BLUE</span> = Mantissa Bits

### *IEEE 754 Single Precision Representations*
- 2.1<sub>10</sub>
    - <span style="color:red">0</span><span style="color:green">100 0000 0</span><span style="color:blue">000 0110 0110 0110 0110 0110</span>
- 6300<sub>10</sub>
    - <span style="color:red">0</span><span style="color:green">100 0101 1</span><span style="color:blue">100 0100 1110 0000 0000 0000</span>
- -1.044<sub>10</sub>
    - <span style="color:red">1</span><span style="color:green">011 1111 1</span><span style="color:blue">000 0101 1010 0001 1100 1011</span>

### *IEEE 754 Double Precision Representations*
- 2.1<sub>10</sub>
    - <span style="color:red">0</span><span style="color:green">100 0000 0000</span><span style="color:blue"> 0000 1100 1100 1100 1100 1100 1100 1100 1100 1100 1100 1100 1101</span>
- 6300<sub>10</sub>
    - <span style="color:red">0</span><span style="color:green">100 0000 1011</span><span style="color:blue"> 1000 1001 1100 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000</span>
- -1.044<sub>10</sub>
    - <span style="color:red">1</span><span style="color:green">011 1111 1111</span><span style="color:blue"> 0000 1011 0100 0011 1001 0101 1000 0001 0000 0110 0010 0100 1110</span>

## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```