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

### Results for 10 Terms
```
Using 10 terms in the Taylor series expansion.

Results are given in radians:
-------------------------------------------------------
sin(0.5) using float:  0.479425519704818725585937500000
sin(0.5) using double: 0.479425538604203005377257795772
Actual value:          0.479425538604203000273287935215
-------------------------------------------------------
sin(1.0) using float:  0.841470956802368164062500000000
sin(1.0) using double: 0.841470984807896504875657228695
Actual value:          0.841470984807896506652502321630
-------------------------------------------------------
sin(2.0) using float:  0.909297466278076171875000000000
sin(2.0) using double: 0.909297426825640964231922680483
Actual value:          0.909297426825681695396019865911
-------------------------------------------------------
sin(3.0) using float:  0.141119971871376037597656250000
sin(3.0) using double: 0.141120007858714924253717981628
Actual value:          0.141120008059867222100744802808
-------------------------------------------------------
```

### Results for 20 Terms
```
Using 20 terms in the Taylor series expansion.

Results are given in radians:
-------------------------------------------------------
sin(0.5) using float:  0.479425519704818725585937500000
sin(0.5) using double: 0.479425538604203005377257795772
Actual value:          0.479425538604203000273287935215
-------------------------------------------------------
sin(1.0) using float:  0.841470956802368164062500000000
sin(1.0) using double: 0.841470984807896504875657228695
Actual value:          0.841470984807896506652502321630
-------------------------------------------------------
sin(2.0) using float:  0.909297347068786621093750000000
sin(2.0) using double: 0.909297310288980642489775618742
Actual value:          0.909297426825681695396019865911
-------------------------------------------------------
sin(3.0) using float:  -1.21354091167449951171875000000
sin(3.0) using double: -1.21354092079542041915374284144
Actual value:          0.141120008059867222100744802808
-------------------------------------------------------
```

### Results for 30 Terms
```
Using 30 terms in the Taylor series expansion.

Results are given in radians:
-------------------------------------------------------
sin(0.5) using float:  0.479425519704818725585937500000
sin(0.5) using double: 0.479425538604203005377257795772
Actual value:          0.479425538604203000273287935215
-------------------------------------------------------
sin(1.0) using float:  0.841470956802368164062500000000
sin(1.0) using double: 0.841470984807896504875657228695
Actual value:          0.841470984807896506652502321630
-------------------------------------------------------
sin(2.0) using float:  -2.62850379943847656250000000000
sin(2.0) using double: -2.62850368957573143902095580415
Actual value:          0.909297426825681695396019865911
-------------------------------------------------------
sin(3.0) using float:  -86934339584.0000000000000000000
sin(3.0) using double: -86934340718.0200042724609375000
Actual value:          0.141120008059867222100744802808
-------------------------------------------------------
```

### Results for 40 Terms
```
Using 40 terms in the Taylor series expansion.

Results are given in radians:
-------------------------------------------------------
sin(0.5) using float:  -nan
sin(0.5) using double: -nan
Actual value:          0.479425538604203000273287935215
-------------------------------------------------------
sin(1.0) using float:  -nan
sin(1.0) using double: -nan
Actual value:          0.841470984807896506652502321630
-------------------------------------------------------
sin(2.0) using float:  -nan
sin(2.0) using double: -nan
Actual value:          0.909297426825681695396019865911
-------------------------------------------------------
sin(3.0) using float:  -nan
sin(3.0) using double: -nan
Actual value:          0.141120008059867222100744802808
-------------------------------------------------------
```

### Discussion

Taking a look at the results for using 10 terms in the Taylor series expansion approximation, we can clearly see that the results for the floating point (single-precision) data format are not as precise as the double-precision data format. At around the 8th decimal place, the single-precision data format begins to lose precision. This pattern is prevalent for some of the other values of `x`, with a few rounding exceptions. On the other hand, for the double-precision format, it is precise up until the 18th decimal place until we begin to see it skew from the actual value. This is expected as the double-precision format has a higher precision than the single-precision format.

An interesting observation is that at larger number of terms, the results begin to breakdown exponentially. I largely speculate that this is because at a larger number of terms, the computer begins to fail to calculate the factorials properly as the numbers are such large values. This is evident in the results for 30 and 40 terms, where the results are not even close to the actual value and quite nonsensical. For these input values, it seems the most number of terms that can be used before any form of breakdown occurs is ~17 terms.

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