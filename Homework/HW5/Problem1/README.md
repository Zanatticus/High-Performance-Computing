# Problem 1 (40 Points)

Let us revisit the histogramming problem assigned in Homework 4. As in Homework 4, your input to this program will be integers in the range 1-100,000 this time (use a random number generator that generates the numbers on the host). Your host-based data set should contain N integers, where N can be varied in the program.

a. This time you will implement a histogramming kernel in CUDA and run on a single GPU (P100 or V100). You can choose how to compute each class of the data set on the GPU. Attempt to adjust the grid size to get the best performance as you vary N. Experiment with \( N = 2^{12} \) to \( 2^{23} \). When the GPU finishes, print one element from each class in class ascending order on the host (do not include the printing time in the timing measurements, though do include the device-to-host communication in the timing measurement). Plot your results in a graph.

b. Compare your GPU performance with running this same code on a CPU using OpenMP.

## Part (a)

This program was run on the `NVIDIA Tesla P100 GPU` on `Node c2187` of the Explorer system.

#### Results for 128 bins, N = \( 2^{12} \), block size = 64, grid size = 64
<details>
<summary>Click to view full histogram results</summary>

```
CUDA Histogram Results
============================

Total elements: 4096 | Range: 1-100000 | Number of Bins: 128
Block Size (Threads Per Block): 64 | Grid Size (Number of Blocks): 64
Execution Time (including device sync & copy): 0.0001670010 seconds

Bin 0: [1 - 781]
  ├── Count: 27
  └── Example Value: 132
Bin 1: [782 - 1562]
  ├── Count: 43
  └── Example Value: 1184
Bin 2: [1563 - 2343]
  ├── Count: 29
  └── Example Value: 1878
Bin 3: [2344 - 3125]
  ├── Count: 33
  └── Example Value: 3070
Bin 4: [3126 - 3906]
  ├── Count: 37
  └── Example Value: 3158
Bin 5: [3907 - 4687]
  ├── Count: 29
  └── Example Value: 4314
Bin 6: [4688 - 5468]
  ├── Count: 32
  └── Example Value: 5341
Bin 7: [5469 - 6250]
  ├── Count: 31
  └── Example Value: 5546
Bin 8: [6251 - 7031]
  ├── Count: 38
  └── Example Value: 6578
Bin 9: [7032 - 7812]
  ├── Count: 35
  └── Example Value: 7281
Bin 10: [7813 - 8593]
  ├── Count: 33
  └── Example Value: 8583
Bin 11: [8594 - 9375]
  ├── Count: 22
  └── Example Value: 8922
Bin 12: [9376 - 10156]
  ├── Count: 32
  └── Example Value: 9434
Bin 13: [10157 - 10937]
  ├── Count: 34
  └── Example Value: 10295
Bin 14: [10938 - 11718]
  ├── Count: 29
  └── Example Value: 11054
Bin 15: [11719 - 12500]
  ├── Count: 35
  └── Example Value: 11797
Bin 16: [12501 - 13281]
  ├── Count: 38
  └── Example Value: 13070
Bin 17: [13282 - 14062]
  ├── Count: 30
  └── Example Value: 14058
Bin 18: [14063 - 14843]
  ├── Count: 28
  └── Example Value: 14108
Bin 19: [14844 - 15625]
  ├── Count: 28
  └── Example Value: 15465
Bin 20: [15626 - 16406]
  ├── Count: 34
  └── Example Value: 16036
Bin 21: [16407 - 17187]
  ├── Count: 35
  └── Example Value: 16555
Bin 22: [17188 - 17968]
  ├── Count: 33
  └── Example Value: 17639
Bin 23: [17969 - 18750]
  ├── Count: 35
  └── Example Value: 18065
Bin 24: [18751 - 19531]
  ├── Count: 41
  └── Example Value: 18860
Bin 25: [19532 - 20312]
  ├── Count: 31
  └── Example Value: 20173
Bin 26: [20313 - 21093]
  ├── Count: 38
  └── Example Value: 20660
Bin 27: [21094 - 21875]
  ├── Count: 25
  └── Example Value: 21336
Bin 28: [21876 - 22656]
  ├── Count: 44
  └── Example Value: 22574
Bin 29: [22657 - 23437]
  ├── Count: 38
  └── Example Value: 22903
Bin 30: [23438 - 24218]
  ├── Count: 27
  └── Example Value: 24047
Bin 31: [24219 - 25000]
  ├── Count: 23
  └── Example Value: 24791
Bin 32: [25001 - 25781]
  ├── Count: 27
  └── Example Value: 25353
Bin 33: [25782 - 26562]
  ├── Count: 36
  └── Example Value: 26313
Bin 34: [26563 - 27343]
  ├── Count: 32
  └── Example Value: 27017
Bin 35: [27344 - 28125]
  ├── Count: 28
  └── Example Value: 27352
Bin 36: [28126 - 28906]
  ├── Count: 42
  └── Example Value: 28829
Bin 37: [28907 - 29687]
  ├── Count: 24
  └── Example Value: 29295
Bin 38: [29688 - 30468]
  ├── Count: 29
  └── Example Value: 30243
Bin 39: [30469 - 31250]
  ├── Count: 27
  └── Example Value: 31110
Bin 40: [31251 - 32031]
  ├── Count: 26
  └── Example Value: 31810
Bin 41: [32032 - 32812]
  ├── Count: 31
  └── Example Value: 32174
Bin 42: [32813 - 33593]
  ├── Count: 39
  └── Example Value: 33024
Bin 43: [33594 - 34375]
  ├── Count: 37
  └── Example Value: 33765
Bin 44: [34376 - 35156]
  ├── Count: 35
  └── Example Value: 34464
Bin 45: [35157 - 35937]
  ├── Count: 35
  └── Example Value: 35380
Bin 46: [35938 - 36718]
  ├── Count: 30
  └── Example Value: 36637
Bin 47: [36719 - 37500]
  ├── Count: 27
  └── Example Value: 36998
Bin 48: [37501 - 38281]
  ├── Count: 30
  └── Example Value: 37954
Bin 49: [38282 - 39062]
  ├── Count: 35
  └── Example Value: 39022
Bin 50: [39063 - 39843]
  ├── Count: 30
  └── Example Value: 39468
Bin 51: [39844 - 40625]
  ├── Count: 35
  └── Example Value: 39852
Bin 52: [40626 - 41406]
  ├── Count: 31
  └── Example Value: 41124
Bin 53: [41407 - 42187]
  ├── Count: 29
  └── Example Value: 41485
Bin 54: [42188 - 42968]
  ├── Count: 30
  └── Example Value: 42492
Bin 55: [42969 - 43750]
  ├── Count: 30
  └── Example Value: 43659
Bin 56: [43751 - 44531]
  ├── Count: 27
  └── Example Value: 44071
Bin 57: [44532 - 45312]
  ├── Count: 34
  └── Example Value: 44875
Bin 58: [45313 - 46093]
  ├── Count: 29
  └── Example Value: 45488
Bin 59: [46094 - 46875]
  ├── Count: 30
  └── Example Value: 46268
Bin 60: [46876 - 47656]
  ├── Count: 35
  └── Example Value: 47228
Bin 61: [47657 - 48437]
  ├── Count: 32
  └── Example Value: 48347
Bin 62: [48438 - 49218]
  ├── Count: 38
  └── Example Value: 48978
Bin 63: [49219 - 50000]
  ├── Count: 33
  └── Example Value: 49340
Bin 64: [50001 - 50781]
  ├── Count: 29
  └── Example Value: 50574
Bin 65: [50782 - 51562]
  ├── Count: 30
  └── Example Value: 51321
Bin 66: [51563 - 52343]
  ├── Count: 31
  └── Example Value: 52168
Bin 67: [52344 - 53125]
  ├── Count: 32
  └── Example Value: 52618
Bin 68: [53126 - 53906]
  ├── Count: 42
  └── Example Value: 53826
Bin 69: [53907 - 54687]
  ├── Count: 40
  └── Example Value: 54422
Bin 70: [54688 - 55468]
  ├── Count: 27
  └── Example Value: 55352
Bin 71: [55469 - 56250]
  ├── Count: 28
  └── Example Value: 55857
Bin 72: [56251 - 57031]
  ├── Count: 26
  └── Example Value: 56299
Bin 73: [57032 - 57812]
  ├── Count: 31
  └── Example Value: 57670
Bin 74: [57813 - 58593]
  ├── Count: 20
  └── Example Value: 57915
Bin 75: [58594 - 59375]
  ├── Count: 39
  └── Example Value: 58926
Bin 76: [59376 - 60156]
  ├── Count: 37
  └── Example Value: 60030
Bin 77: [60157 - 60937]
  ├── Count: 31
  └── Example Value: 60872
Bin 78: [60938 - 61718]
  ├── Count: 29
  └── Example Value: 61452
Bin 79: [61719 - 62500]
  ├── Count: 41
  └── Example Value: 62298
Bin 80: [62501 - 63281]
  ├── Count: 40
  └── Example Value: 63073
Bin 81: [63282 - 64062]
  ├── Count: 31
  └── Example Value: 63971
Bin 82: [64063 - 64843]
  ├── Count: 34
  └── Example Value: 64525
Bin 83: [64844 - 65625]
  ├── Count: 37
  └── Example Value: 65015
Bin 84: [65626 - 66406]
  ├── Count: 39
  └── Example Value: 65872
Bin 85: [66407 - 67187]
  ├── Count: 29
  └── Example Value: 66769
Bin 86: [67188 - 67968]
  ├── Count: 31
  └── Example Value: 67655
Bin 87: [67969 - 68750]
  ├── Count: 26
  └── Example Value: 68448
Bin 88: [68751 - 69531]
  ├── Count: 30
  └── Example Value: 69411
Bin 89: [69532 - 70312]
  ├── Count: 33
  └── Example Value: 70038
Bin 90: [70313 - 71093]
  ├── Count: 29
  └── Example Value: 70642
Bin 91: [71094 - 71875]
  ├── Count: 27
  └── Example Value: 71126
Bin 92: [71876 - 72656]
  ├── Count: 31
  └── Example Value: 72059
Bin 93: [72657 - 73437]
  ├── Count: 37
  └── Example Value: 72785
Bin 94: [73438 - 74218]
  ├── Count: 34
  └── Example Value: 73538
Bin 95: [74219 - 75000]
  ├── Count: 40
  └── Example Value: 74581
Bin 96: [75001 - 75781]
  ├── Count: 34
  └── Example Value: 75110
Bin 97: [75782 - 76562]
  ├── Count: 32
  └── Example Value: 76365
Bin 98: [76563 - 77343]
  ├── Count: 33
  └── Example Value: 76781
Bin 99: [77344 - 78125]
  ├── Count: 33
  └── Example Value: 77617
Bin 100: [78126 - 78906]
  ├── Count: 24
  └── Example Value: 78342
Bin 101: [78907 - 79687]
  ├── Count: 28
  └── Example Value: 79208
Bin 102: [79688 - 80468]
  ├── Count: 23
  └── Example Value: 79928
Bin 103: [80469 - 81250]
  ├── Count: 29
  └── Example Value: 80710
Bin 104: [81251 - 82031]
  ├── Count: 25
  └── Example Value: 82015
Bin 105: [82032 - 82812]
  ├── Count: 36
  └── Example Value: 82339
Bin 106: [82813 - 83593]
  ├── Count: 23
  └── Example Value: 82934
Bin 107: [83594 - 84375]
  ├── Count: 24
  └── Example Value: 84265
Bin 108: [84376 - 85156]
  ├── Count: 23
  └── Example Value: 84813
Bin 109: [85157 - 85937]
  ├── Count: 34
  └── Example Value: 85753
Bin 110: [85938 - 86718]
  ├── Count: 35
  └── Example Value: 86253
Bin 111: [86719 - 87500]
  ├── Count: 36
  └── Example Value: 86947
Bin 112: [87501 - 88281]
  ├── Count: 37
  └── Example Value: 88264
Bin 113: [88282 - 89062]
  ├── Count: 34
  └── Example Value: 88412
Bin 114: [89063 - 89843]
  ├── Count: 34
  └── Example Value: 89745
Bin 115: [89844 - 90625]
  ├── Count: 38
  └── Example Value: 90359
Bin 116: [90626 - 91406]
  ├── Count: 31
  └── Example Value: 90959
Bin 117: [91407 - 92187]
  ├── Count: 32
  └── Example Value: 91615
Bin 118: [92188 - 92968]
  ├── Count: 38
  └── Example Value: 92686
Bin 119: [92969 - 93750]
  ├── Count: 34
  └── Example Value: 93500
Bin 120: [93751 - 94531]
  ├── Count: 22
  └── Example Value: 94081
Bin 121: [94532 - 95312]
  ├── Count: 33
  └── Example Value: 94978
Bin 122: [95313 - 96093]
  ├── Count: 39
  └── Example Value: 95408
Bin 123: [96094 - 96875]
  ├── Count: 22
  └── Example Value: 96415
Bin 124: [96876 - 97656]
  ├── Count: 29
  └── Example Value: 97112
Bin 125: [97657 - 98437]
  ├── Count: 34
  └── Example Value: 97757
Bin 126: [98438 - 99218]
  ├── Count: 34
  └── Example Value: 98879
Bin 127: [99219 - 100000]
  ├── Count: 32
  └── Example Value: 99905
```
</details>

##### CUDA Program Performance
- Kernel Execution Time: 0.0001670010 seconds

#### Results for 128 bins, N = \( 2^{12} \), block size = 128, grid size = 32
<details>
<summary>Click to view full histogram results</summary>

```
CUDA Histogram Results
============================

Total elements: 4096 | Range: 1-100000 | Number of Bins: 128
Block Size (Threads Per Block): 128 | Grid Size (Number of Blocks): 32
Execution Time (including device sync & copy): 0.0001797160 seconds

Bin 0: [1 - 781]
  ├── Count: 27
  └── Example Value: 770
Bin 1: [782 - 1562]
  ├── Count: 43
  └── Example Value: 822
Bin 2: [1563 - 2343]
  ├── Count: 29
  └── Example Value: 2309
Bin 3: [2344 - 3125]
  ├── Count: 33
  └── Example Value: 2558
Bin 4: [3126 - 3906]
  ├── Count: 37
  └── Example Value: 3606
Bin 5: [3907 - 4687]
  ├── Count: 29
  └── Example Value: 4235
Bin 6: [4688 - 5468]
  ├── Count: 32
  └── Example Value: 5193
Bin 7: [5469 - 6250]
  ├── Count: 31
  └── Example Value: 5947
Bin 8: [6251 - 7031]
  ├── Count: 38
  └── Example Value: 6688
Bin 9: [7032 - 7812]
  ├── Count: 35
  └── Example Value: 7125
Bin 10: [7813 - 8593]
  ├── Count: 33
  └── Example Value: 8343
Bin 11: [8594 - 9375]
  ├── Count: 22
  └── Example Value: 8765
Bin 12: [9376 - 10156]
  ├── Count: 32
  └── Example Value: 10063
Bin 13: [10157 - 10937]
  ├── Count: 34
  └── Example Value: 10513
Bin 14: [10938 - 11718]
  ├── Count: 29
  └── Example Value: 11626
Bin 15: [11719 - 12500]
  ├── Count: 35
  └── Example Value: 12299
Bin 16: [12501 - 13281]
  ├── Count: 38
  └── Example Value: 13070
Bin 17: [13282 - 14062]
  ├── Count: 30
  └── Example Value: 13971
Bin 18: [14063 - 14843]
  ├── Count: 28
  └── Example Value: 14678
Bin 19: [14844 - 15625]
  ├── Count: 28
  └── Example Value: 14947
Bin 20: [15626 - 16406]
  ├── Count: 34
  └── Example Value: 16348
Bin 21: [16407 - 17187]
  ├── Count: 35
  └── Example Value: 17070
Bin 22: [17188 - 17968]
  ├── Count: 33
  └── Example Value: 17554
Bin 23: [17969 - 18750]
  ├── Count: 35
  └── Example Value: 18558
Bin 24: [18751 - 19531]
  ├── Count: 41
  └── Example Value: 19219
Bin 25: [19532 - 20312]
  ├── Count: 31
  └── Example Value: 19678
Bin 26: [20313 - 21093]
  ├── Count: 38
  └── Example Value: 20499
Bin 27: [21094 - 21875]
  ├── Count: 25
  └── Example Value: 21387
Bin 28: [21876 - 22656]
  ├── Count: 44
  └── Example Value: 21893
Bin 29: [22657 - 23437]
  ├── Count: 38
  └── Example Value: 22903
Bin 30: [23438 - 24218]
  ├── Count: 27
  └── Example Value: 23788
Bin 31: [24219 - 25000]
  ├── Count: 23
  └── Example Value: 24791
Bin 32: [25001 - 25781]
  ├── Count: 27
  └── Example Value: 25362
Bin 33: [25782 - 26562]
  ├── Count: 36
  └── Example Value: 26313
Bin 34: [26563 - 27343]
  ├── Count: 32
  └── Example Value: 26772
Bin 35: [27344 - 28125]
  ├── Count: 28
  └── Example Value: 27352
Bin 36: [28126 - 28906]
  ├── Count: 42
  └── Example Value: 28280
Bin 37: [28907 - 29687]
  ├── Count: 24
  └── Example Value: 29295
Bin 38: [29688 - 30468]
  ├── Count: 29
  └── Example Value: 30208
Bin 39: [30469 - 31250]
  ├── Count: 27
  └── Example Value: 30691
Bin 40: [31251 - 32031]
  ├── Count: 26
  └── Example Value: 31627
Bin 41: [32032 - 32812]
  ├── Count: 31
  └── Example Value: 32701
Bin 42: [32813 - 33593]
  ├── Count: 39
  └── Example Value: 33131
Bin 43: [33594 - 34375]
  ├── Count: 37
  └── Example Value: 34226
Bin 44: [34376 - 35156]
  ├── Count: 35
  └── Example Value: 34448
Bin 45: [35157 - 35937]
  ├── Count: 35
  └── Example Value: 35669
Bin 46: [35938 - 36718]
  ├── Count: 30
  └── Example Value: 36272
Bin 47: [36719 - 37500]
  ├── Count: 27
  └── Example Value: 37364
Bin 48: [37501 - 38281]
  ├── Count: 30
  └── Example Value: 37737
Bin 49: [38282 - 39062]
  ├── Count: 35
  └── Example Value: 39022
Bin 50: [39063 - 39843]
  ├── Count: 30
  └── Example Value: 39260
Bin 51: [39844 - 40625]
  ├── Count: 35
  └── Example Value: 39863
Bin 52: [40626 - 41406]
  ├── Count: 31
  └── Example Value: 41321
Bin 53: [41407 - 42187]
  ├── Count: 29
  └── Example Value: 42014
Bin 54: [42188 - 42968]
  ├── Count: 30
  └── Example Value: 42781
Bin 55: [42969 - 43750]
  ├── Count: 30
  └── Example Value: 43429
Bin 56: [43751 - 44531]
  ├── Count: 27
  └── Example Value: 44050
Bin 57: [44532 - 45312]
  ├── Count: 34
  └── Example Value: 45230
Bin 58: [45313 - 46093]
  ├── Count: 29
  └── Example Value: 45765
Bin 59: [46094 - 46875]
  ├── Count: 30
  └── Example Value: 46580
Bin 60: [46876 - 47656]
  ├── Count: 35
  └── Example Value: 47387
Bin 61: [47657 - 48437]
  ├── Count: 32
  └── Example Value: 48402
Bin 62: [48438 - 49218]
  ├── Count: 38
  └── Example Value: 48610
Bin 63: [49219 - 50000]
  ├── Count: 33
  └── Example Value: 49615
Bin 64: [50001 - 50781]
  ├── Count: 29
  └── Example Value: 50053
Bin 65: [50782 - 51562]
  ├── Count: 30
  └── Example Value: 51319
Bin 66: [51563 - 52343]
  ├── Count: 31
  └── Example Value: 52275
Bin 67: [52344 - 53125]
  ├── Count: 32
  └── Example Value: 52482
Bin 68: [53126 - 53906]
  ├── Count: 42
  └── Example Value: 53715
Bin 69: [53907 - 54687]
  ├── Count: 40
  └── Example Value: 54422
Bin 70: [54688 - 55468]
  ├── Count: 27
  └── Example Value: 55352
Bin 71: [55469 - 56250]
  ├── Count: 28
  └── Example Value: 55867
Bin 72: [56251 - 57031]
  ├── Count: 26
  └── Example Value: 56553
Bin 73: [57032 - 57812]
  ├── Count: 31
  └── Example Value: 57437
Bin 74: [57813 - 58593]
  ├── Count: 20
  └── Example Value: 57915
Bin 75: [58594 - 59375]
  ├── Count: 39
  └── Example Value: 58727
Bin 76: [59376 - 60156]
  ├── Count: 37
  └── Example Value: 59480
Bin 77: [60157 - 60937]
  ├── Count: 31
  └── Example Value: 60597
Bin 78: [60938 - 61718]
  ├── Count: 29
  └── Example Value: 61529
Bin 79: [61719 - 62500]
  ├── Count: 41
  └── Example Value: 62298
Bin 80: [62501 - 63281]
  ├── Count: 40
  └── Example Value: 63182
Bin 81: [63282 - 64062]
  ├── Count: 31
  └── Example Value: 64046
Bin 82: [64063 - 64843]
  ├── Count: 34
  └── Example Value: 64525
Bin 83: [64844 - 65625]
  ├── Count: 37
  └── Example Value: 65101
Bin 84: [65626 - 66406]
  ├── Count: 39
  └── Example Value: 65872
Bin 85: [66407 - 67187]
  ├── Count: 29
  └── Example Value: 66705
Bin 86: [67188 - 67968]
  ├── Count: 31
  └── Example Value: 67819
Bin 87: [67969 - 68750]
  ├── Count: 26
  └── Example Value: 68429
Bin 88: [68751 - 69531]
  ├── Count: 30
  └── Example Value: 68785
Bin 89: [69532 - 70312]
  ├── Count: 33
  └── Example Value: 69830
Bin 90: [70313 - 71093]
  ├── Count: 29
  └── Example Value: 70642
Bin 91: [71094 - 71875]
  ├── Count: 27
  └── Example Value: 71523
Bin 92: [71876 - 72656]
  ├── Count: 31
  └── Example Value: 72073
Bin 93: [72657 - 73437]
  ├── Count: 37
  └── Example Value: 73169
Bin 94: [73438 - 74218]
  ├── Count: 34
  └── Example Value: 73514
Bin 95: [74219 - 75000]
  ├── Count: 40
  └── Example Value: 74581
Bin 96: [75001 - 75781]
  ├── Count: 34
  └── Example Value: 75705
Bin 97: [75782 - 76562]
  ├── Count: 32
  └── Example Value: 76181
Bin 98: [76563 - 77343]
  ├── Count: 33
  └── Example Value: 76791
Bin 99: [77344 - 78125]
  ├── Count: 33
  └── Example Value: 77951
Bin 100: [78126 - 78906]
  ├── Count: 24
  └── Example Value: 78441
Bin 101: [78907 - 79687]
  ├── Count: 28
  └── Example Value: 79208
Bin 102: [79688 - 80468]
  ├── Count: 23
  └── Example Value: 80130
Bin 103: [80469 - 81250]
  ├── Count: 29
  └── Example Value: 80520
Bin 104: [81251 - 82031]
  ├── Count: 25
  └── Example Value: 81877
Bin 105: [82032 - 82812]
  ├── Count: 36
  └── Example Value: 82339
Bin 106: [82813 - 83593]
  ├── Count: 23
  └── Example Value: 83399
Bin 107: [83594 - 84375]
  ├── Count: 24
  └── Example Value: 84149
Bin 108: [84376 - 85156]
  ├── Count: 23
  └── Example Value: 84813
Bin 109: [85157 - 85937]
  ├── Count: 34
  └── Example Value: 85901
Bin 110: [85938 - 86718]
  ├── Count: 35
  └── Example Value: 86253
Bin 111: [86719 - 87500]
  ├── Count: 36
  └── Example Value: 86975
Bin 112: [87501 - 88281]
  ├── Count: 37
  └── Example Value: 88264
Bin 113: [88282 - 89062]
  ├── Count: 34
  └── Example Value: 88614
Bin 114: [89063 - 89843]
  ├── Count: 34
  └── Example Value: 89745
Bin 115: [89844 - 90625]
  ├── Count: 38
  └── Example Value: 90359
Bin 116: [90626 - 91406]
  ├── Count: 31
  └── Example Value: 91314
Bin 117: [91407 - 92187]
  ├── Count: 32
  └── Example Value: 91880
Bin 118: [92188 - 92968]
  ├── Count: 38
  └── Example Value: 92686
Bin 119: [92969 - 93750]
  ├── Count: 34
  └── Example Value: 93530
Bin 120: [93751 - 94531]
  ├── Count: 22
  └── Example Value: 94115
Bin 121: [94532 - 95312]
  ├── Count: 33
  └── Example Value: 94661
Bin 122: [95313 - 96093]
  ├── Count: 39
  └── Example Value: 96009
Bin 123: [96094 - 96875]
  ├── Count: 22
  └── Example Value: 96415
Bin 124: [96876 - 97656]
  ├── Count: 29
  └── Example Value: 96912
Bin 125: [97657 - 98437]
  ├── Count: 34
  └── Example Value: 98338
Bin 126: [98438 - 99218]
  ├── Count: 34
  └── Example Value: 99052
Bin 127: [99219 - 100000]
  ├── Count: 32
  └── Example Value: 99632
```
</details>

##### CUDA Program Performance
- Kernel Execution Time: 0.0001797160 seconds

#### Results for 128 bins, N = \( 2^{12} \), block size = 256, grid size = 16
<details>
<summary>Click to view full histogram results</summary>

```
CUDA Histogram Results
============================

Total elements: 4096 | Range: 1-100000 | Number of Bins: 128
Block Size (Threads Per Block): 256 | Grid Size (Number of Blocks): 16
Execution Time (including device sync & copy): 0.0001759200 seconds

Bin 0: [1 - 781]
  ├── Count: 27
  └── Example Value: 708
Bin 1: [782 - 1562]
  ├── Count: 43
  └── Example Value: 1172
Bin 2: [1563 - 2343]
  ├── Count: 29
  └── Example Value: 1746
Bin 3: [2344 - 3125]
  ├── Count: 33
  └── Example Value: 2558
Bin 4: [3126 - 3906]
  ├── Count: 37
  └── Example Value: 3869
Bin 5: [3907 - 4687]
  ├── Count: 29
  └── Example Value: 4140
Bin 6: [4688 - 5468]
  ├── Count: 32
  └── Example Value: 4900
Bin 7: [5469 - 6250]
  ├── Count: 31
  └── Example Value: 5995
Bin 8: [6251 - 7031]
  ├── Count: 38
  └── Example Value: 6507
Bin 9: [7032 - 7812]
  ├── Count: 35
  └── Example Value: 7281
Bin 10: [7813 - 8593]
  ├── Count: 33
  └── Example Value: 8343
Bin 11: [8594 - 9375]
  ├── Count: 22
  └── Example Value: 9365
Bin 12: [9376 - 10156]
  ├── Count: 32
  └── Example Value: 9440
Bin 13: [10157 - 10937]
  ├── Count: 34
  └── Example Value: 10828
Bin 14: [10938 - 11718]
  ├── Count: 29
  └── Example Value: 11155
Bin 15: [11719 - 12500]
  ├── Count: 35
  └── Example Value: 11957
Bin 16: [12501 - 13281]
  ├── Count: 38
  └── Example Value: 13084
Bin 17: [13282 - 14062]
  ├── Count: 30
  └── Example Value: 13991
Bin 18: [14063 - 14843]
  ├── Count: 28
  └── Example Value: 14338
Bin 19: [14844 - 15625]
  ├── Count: 28
  └── Example Value: 15464
Bin 20: [15626 - 16406]
  ├── Count: 34
  └── Example Value: 15922
Bin 21: [16407 - 17187]
  ├── Count: 35
  └── Example Value: 17070
Bin 22: [17188 - 17968]
  ├── Count: 33
  └── Example Value: 17554
Bin 23: [17969 - 18750]
  ├── Count: 35
  └── Example Value: 18558
Bin 24: [18751 - 19531]
  ├── Count: 41
  └── Example Value: 18860
Bin 25: [19532 - 20312]
  ├── Count: 31
  └── Example Value: 20161
Bin 26: [20313 - 21093]
  ├── Count: 38
  └── Example Value: 20361
Bin 27: [21094 - 21875]
  ├── Count: 25
  └── Example Value: 21826
Bin 28: [21876 - 22656]
  ├── Count: 44
  └── Example Value: 22216
Bin 29: [22657 - 23437]
  ├── Count: 38
  └── Example Value: 23173
Bin 30: [23438 - 24218]
  ├── Count: 27
  └── Example Value: 24150
Bin 31: [24219 - 25000]
  ├── Count: 23
  └── Example Value: 24791
Bin 32: [25001 - 25781]
  ├── Count: 27
  └── Example Value: 25353
Bin 33: [25782 - 26562]
  ├── Count: 36
  └── Example Value: 26273
Bin 34: [26563 - 27343]
  ├── Count: 32
  └── Example Value: 26653
Bin 35: [27344 - 28125]
  ├── Count: 28
  └── Example Value: 27352
Bin 36: [28126 - 28906]
  ├── Count: 42
  └── Example Value: 28829
Bin 37: [28907 - 29687]
  ├── Count: 24
  └── Example Value: 29321
Bin 38: [29688 - 30468]
  ├── Count: 29
  └── Example Value: 29849
Bin 39: [30469 - 31250]
  ├── Count: 27
  └── Example Value: 30773
Bin 40: [31251 - 32031]
  ├── Count: 26
  └── Example Value: 31490
Bin 41: [32032 - 32812]
  ├── Count: 31
  └── Example Value: 32174
Bin 42: [32813 - 33593]
  ├── Count: 39
  └── Example Value: 32934
Bin 43: [33594 - 34375]
  ├── Count: 37
  └── Example Value: 34226
Bin 44: [34376 - 35156]
  ├── Count: 35
  └── Example Value: 34970
Bin 45: [35157 - 35937]
  ├── Count: 35
  └── Example Value: 35822
Bin 46: [35938 - 36718]
  ├── Count: 30
  └── Example Value: 36011
Bin 47: [36719 - 37500]
  ├── Count: 27
  └── Example Value: 36916
Bin 48: [37501 - 38281]
  ├── Count: 30
  └── Example Value: 37576
Bin 49: [38282 - 39062]
  ├── Count: 35
  └── Example Value: 38369
Bin 50: [39063 - 39843]
  ├── Count: 30
  └── Example Value: 39681
Bin 51: [39844 - 40625]
  ├── Count: 35
  └── Example Value: 39852
Bin 52: [40626 - 41406]
  ├── Count: 31
  └── Example Value: 41352
Bin 53: [41407 - 42187]
  ├── Count: 29
  └── Example Value: 42044
Bin 54: [42188 - 42968]
  ├── Count: 30
  └── Example Value: 42492
Bin 55: [42969 - 43750]
  ├── Count: 30
  └── Example Value: 43364
Bin 56: [43751 - 44531]
  ├── Count: 27
  └── Example Value: 44071
Bin 57: [44532 - 45312]
  ├── Count: 34
  └── Example Value: 44697
Bin 58: [45313 - 46093]
  ├── Count: 29
  └── Example Value: 45765
Bin 59: [46094 - 46875]
  ├── Count: 30
  └── Example Value: 46591
Bin 60: [46876 - 47656]
  ├── Count: 35
  └── Example Value: 47320
Bin 61: [47657 - 48437]
  ├── Count: 32
  └── Example Value: 47794
Bin 62: [48438 - 49218]
  ├── Count: 38
  └── Example Value: 48570
Bin 63: [49219 - 50000]
  ├── Count: 33
  └── Example Value: 49518
Bin 64: [50001 - 50781]
  ├── Count: 29
  └── Example Value: 50110
Bin 65: [50782 - 51562]
  ├── Count: 30
  └── Example Value: 51143
Bin 66: [51563 - 52343]
  ├── Count: 31
  └── Example Value: 52278
Bin 67: [52344 - 53125]
  ├── Count: 32
  └── Example Value: 53047
Bin 68: [53126 - 53906]
  ├── Count: 42
  └── Example Value: 53878
Bin 69: [53907 - 54687]
  ├── Count: 40
  └── Example Value: 54596
Bin 70: [54688 - 55468]
  ├── Count: 27
  └── Example Value: 55149
Bin 71: [55469 - 56250]
  ├── Count: 28
  └── Example Value: 55857
Bin 72: [56251 - 57031]
  ├── Count: 26
  └── Example Value: 56710
Bin 73: [57032 - 57812]
  ├── Count: 31
  └── Example Value: 57794
Bin 74: [57813 - 58593]
  ├── Count: 20
  └── Example Value: 57915
Bin 75: [58594 - 59375]
  ├── Count: 39
  └── Example Value: 59157
Bin 76: [59376 - 60156]
  ├── Count: 37
  └── Example Value: 59632
Bin 77: [60157 - 60937]
  ├── Count: 31
  └── Example Value: 60872
Bin 78: [60938 - 61718]
  ├── Count: 29
  └── Example Value: 61049
Bin 79: [61719 - 62500]
  ├── Count: 41
  └── Example Value: 62030
Bin 80: [62501 - 63281]
  ├── Count: 40
  └── Example Value: 62845
Bin 81: [63282 - 64062]
  ├── Count: 31
  └── Example Value: 63971
Bin 82: [64063 - 64843]
  ├── Count: 34
  └── Example Value: 64525
Bin 83: [64844 - 65625]
  ├── Count: 37
  └── Example Value: 65418
Bin 84: [65626 - 66406]
  ├── Count: 39
  └── Example Value: 65872
Bin 85: [66407 - 67187]
  ├── Count: 29
  └── Example Value: 66995
Bin 86: [67188 - 67968]
  ├── Count: 31
  └── Example Value: 67745
Bin 87: [67969 - 68750]
  ├── Count: 26
  └── Example Value: 68233
Bin 88: [68751 - 69531]
  ├── Count: 30
  └── Example Value: 69529
Bin 89: [69532 - 70312]
  ├── Count: 33
  └── Example Value: 70038
Bin 90: [70313 - 71093]
  ├── Count: 29
  └── Example Value: 70781
Bin 91: [71094 - 71875]
  ├── Count: 27
  └── Example Value: 71755
Bin 92: [71876 - 72656]
  ├── Count: 31
  └── Example Value: 72327
Bin 93: [72657 - 73437]
  ├── Count: 37
  └── Example Value: 72785
Bin 94: [73438 - 74218]
  ├── Count: 34
  └── Example Value: 73538
Bin 95: [74219 - 75000]
  ├── Count: 40
  └── Example Value: 74447
Bin 96: [75001 - 75781]
  ├── Count: 34
  └── Example Value: 75359
Bin 97: [75782 - 76562]
  ├── Count: 32
  └── Example Value: 76365
Bin 98: [76563 - 77343]
  ├── Count: 33
  └── Example Value: 76988
Bin 99: [77344 - 78125]
  ├── Count: 33
  └── Example Value: 77806
Bin 100: [78126 - 78906]
  ├── Count: 24
  └── Example Value: 78342
Bin 101: [78907 - 79687]
  ├── Count: 28
  └── Example Value: 79193
Bin 102: [79688 - 80468]
  ├── Count: 23
  └── Example Value: 79760
Bin 103: [80469 - 81250]
  ├── Count: 29
  └── Example Value: 81033
Bin 104: [81251 - 82031]
  ├── Count: 25
  └── Example Value: 81273
Bin 105: [82032 - 82812]
  ├── Count: 36
  └── Example Value: 82585
Bin 106: [82813 - 83593]
  ├── Count: 23
  └── Example Value: 82934
Bin 107: [83594 - 84375]
  ├── Count: 24
  └── Example Value: 84149
Bin 108: [84376 - 85156]
  ├── Count: 23
  └── Example Value: 84417
Bin 109: [85157 - 85937]
  ├── Count: 34
  └── Example Value: 85186
Bin 110: [85938 - 86718]
  ├── Count: 35
  └── Example Value: 86253
Bin 111: [86719 - 87500]
  ├── Count: 36
  └── Example Value: 86947
Bin 112: [87501 - 88281]
  ├── Count: 37
  └── Example Value: 88264
Bin 113: [88282 - 89062]
  ├── Count: 34
  └── Example Value: 88821
Bin 114: [89063 - 89843]
  ├── Count: 34
  └── Example Value: 89301
Bin 115: [89844 - 90625]
  ├── Count: 38
  └── Example Value: 90359
Bin 116: [90626 - 91406]
  ├── Count: 31
  └── Example Value: 90637
Bin 117: [91407 - 92187]
  ├── Count: 32
  └── Example Value: 91639
Bin 118: [92188 - 92968]
  ├── Count: 38
  └── Example Value: 92332
Bin 119: [92969 - 93750]
  ├── Count: 34
  └── Example Value: 93555
Bin 120: [93751 - 94531]
  ├── Count: 22
  └── Example Value: 94315
Bin 121: [94532 - 95312]
  ├── Count: 33
  └── Example Value: 94950
Bin 122: [95313 - 96093]
  ├── Count: 39
  └── Example Value: 95512
Bin 123: [96094 - 96875]
  ├── Count: 22
  └── Example Value: 96620
Bin 124: [96876 - 97656]
  ├── Count: 29
  └── Example Value: 97112
Bin 125: [97657 - 98437]
  ├── Count: 34
  └── Example Value: 98338
Bin 126: [98438 - 99218]
  ├── Count: 34
  └── Example Value: 98481
Bin 127: [99219 - 100000]
  ├── Count: 32
  └── Example Value: 99485
```
</details>

##### CUDA Program Performance
- Kernel Execution Time: 0.0001759200 seconds

#### Results for 128 bins, N = \( 2^{23} \), block size = 64, grid size = 131072
<details>
<summary>Click to view full histogram results</summary>

```
CUDA Histogram Results
============================

Total elements: 8388608 | Range: 1-100000 | Number of Bins: 128
Block Size (Threads Per Block): 64 | Grid Size (Number of Blocks): 131072
Execution Time (including device sync & copy): 0.0135063770 seconds

Bin 0: [1 - 781]
  └── Count: 65657
  └── Example Value: 322
Bin 1: [782 - 1562]
  └── Count: 65500
  └── Example Value: 1287
Bin 2: [1563 - 2343]
  └── Count: 65822
  └── Example Value: 1761
Bin 3: [2344 - 3125]
  └── Count: 65479
  └── Example Value: 2907
Bin 4: [3126 - 3906]
  └── Count: 65381
  └── Example Value: 3707
Bin 5: [3907 - 4687]
  └── Count: 65570
  └── Example Value: 4487
Bin 6: [4688 - 5468]
  └── Count: 65741
  └── Example Value: 4979
Bin 7: [5469 - 6250]
  └── Count: 65360
  └── Example Value: 5661
Bin 8: [6251 - 7031]
  └── Count: 65671
  └── Example Value: 6932
Bin 9: [7032 - 7812]
  └── Count: 65842
  └── Example Value: 7371
Bin 10: [7813 - 8593]
  └── Count: 65372
  └── Example Value: 8108
Bin 11: [8594 - 9375]
  └── Count: 65626
  └── Example Value: 9091
Bin 12: [9376 - 10156]
  └── Count: 65314
  └── Example Value: 9913
Bin 13: [10157 - 10937]
  └── Count: 65591
  └── Example Value: 10428
Bin 14: [10938 - 11718]
  └── Count: 65905
  └── Example Value: 10993
Bin 15: [11719 - 12500]
  └── Count: 66003
  └── Example Value: 11827
Bin 16: [12501 - 13281]
  └── Count: 65433
  └── Example Value: 13031
Bin 17: [13282 - 14062]
  └── Count: 65846
  └── Example Value: 13949
Bin 18: [14063 - 14843]
  └── Count: 65653
  └── Example Value: 14801
Bin 19: [14844 - 15625]
  └── Count: 65577
  └── Example Value: 15126
Bin 20: [15626 - 16406]
  └── Count: 65661
  └── Example Value: 16154
Bin 21: [16407 - 17187]
  └── Count: 65120
  └── Example Value: 16966
Bin 22: [17188 - 17968]
  └── Count: 65548
  └── Example Value: 17544
Bin 23: [17969 - 18750]
  └── Count: 65523
  └── Example Value: 18108
Bin 24: [18751 - 19531]
  └── Count: 65544
  └── Example Value: 19412
Bin 25: [19532 - 20312]
  └── Count: 65467
  └── Example Value: 19910
Bin 26: [20313 - 21093]
  └── Count: 65639
  └── Example Value: 20532
Bin 27: [21094 - 21875]
  └── Count: 65279
  └── Example Value: 21118
Bin 28: [21876 - 22656]
  └── Count: 65321
  └── Example Value: 22273
Bin 29: [22657 - 23437]
  └── Count: 65543
  └── Example Value: 22930
Bin 30: [23438 - 24218]
  └── Count: 65849
  └── Example Value: 23769
Bin 31: [24219 - 25000]
  └── Count: 65476
  └── Example Value: 24465
Bin 32: [25001 - 25781]
  └── Count: 65582
  └── Example Value: 25125
Bin 33: [25782 - 26562]
  └── Count: 65930
  └── Example Value: 26265
Bin 34: [26563 - 27343]
  └── Count: 65487
  └── Example Value: 27011
Bin 35: [27344 - 28125]
  └── Count: 66058
  └── Example Value: 27400
Bin 36: [28126 - 28906]
  └── Count: 65442
  └── Example Value: 28294
Bin 37: [28907 - 29687]
  └── Count: 65429
  └── Example Value: 29437
Bin 38: [29688 - 30468]
  └── Count: 65292
  └── Example Value: 30150
Bin 39: [30469 - 31250]
  └── Count: 65624
  └── Example Value: 30531
Bin 40: [31251 - 32031]
  └── Count: 65438
  └── Example Value: 31601
Bin 41: [32032 - 32812]
  └── Count: 65164
  └── Example Value: 32240
Bin 42: [32813 - 33593]
  └── Count: 65378
  └── Example Value: 33194
Bin 43: [33594 - 34375]
  └── Count: 65940
  └── Example Value: 34273
Bin 44: [34376 - 35156]
  └── Count: 65776
  └── Example Value: 34445
Bin 45: [35157 - 35937]
  └── Count: 64887
  └── Example Value: 35326
Bin 46: [35938 - 36718]
  └── Count: 65923
  └── Example Value: 36694
Bin 47: [36719 - 37500]
  └── Count: 65692
  └── Example Value: 37464
Bin 48: [37501 - 38281]
  └── Count: 65470
  └── Example Value: 37582
Bin 49: [38282 - 39062]
  └── Count: 65447
  └── Example Value: 38510
Bin 50: [39063 - 39843]
  └── Count: 65309
  └── Example Value: 39256
Bin 51: [39844 - 40625]
  └── Count: 65604
  └── Example Value: 40188
Bin 52: [40626 - 41406]
  └── Count: 65633
  └── Example Value: 41124
Bin 53: [41407 - 42187]
  └── Count: 65341
  └── Example Value: 41608
Bin 54: [42188 - 42968]
  └── Count: 65168
  └── Example Value: 42293
Bin 55: [42969 - 43750]
  └── Count: 65576
  └── Example Value: 43004
Bin 56: [43751 - 44531]
  └── Count: 65178
  └── Example Value: 43830
Bin 57: [44532 - 45312]
  └── Count: 64987
  └── Example Value: 45021
Bin 58: [45313 - 46093]
  └── Count: 65444
  └── Example Value: 46018
Bin 59: [46094 - 46875]
  └── Count: 65833
  └── Example Value: 46173
Bin 60: [46876 - 47656]
  └── Count: 65706
  └── Example Value: 47492
Bin 61: [47657 - 48437]
  └── Count: 65697
  └── Example Value: 48146
Bin 62: [48438 - 49218]
  └── Count: 65297
  └── Example Value: 49011
Bin 63: [49219 - 50000]
  └── Count: 65841
  └── Example Value: 49447
Bin 64: [50001 - 50781]
  └── Count: 66287
  └── Example Value: 50467
Bin 65: [50782 - 51562]
  └── Count: 65161
  └── Example Value: 51202
Bin 66: [51563 - 52343]
  └── Count: 65137
  └── Example Value: 51989
Bin 67: [52344 - 53125]
  └── Count: 65292
  └── Example Value: 53035
Bin 68: [53126 - 53906]
  └── Count: 65767
  └── Example Value: 53409
Bin 69: [53907 - 54687]
  └── Count: 65632
  └── Example Value: 54099
Bin 70: [54688 - 55468]
  └── Count: 65580
  └── Example Value: 54862
Bin 71: [55469 - 56250]
  └── Count: 65058
  └── Example Value: 56143
Bin 72: [56251 - 57031]
  └── Count: 65507
  └── Example Value: 56408
Bin 73: [57032 - 57812]
  └── Count: 65303
  └── Example Value: 57386
Bin 74: [57813 - 58593]
  └── Count: 65627
  └── Example Value: 58477
Bin 75: [58594 - 59375]
  └── Count: 65338
  └── Example Value: 58753
Bin 76: [59376 - 60156]
  └── Count: 65625
  └── Example Value: 59966
Bin 77: [60157 - 60937]
  └── Count: 65485
  └── Example Value: 60559
Bin 78: [60938 - 61718]
  └── Count: 66100
  └── Example Value: 61078
Bin 79: [61719 - 62500]
  └── Count: 65433
  └── Example Value: 61901
Bin 80: [62501 - 63281]
  └── Count: 65487
  └── Example Value: 63180
Bin 81: [63282 - 64062]
  └── Count: 65398
  └── Example Value: 63634
Bin 82: [64063 - 64843]
  └── Count: 65111
  └── Example Value: 64798
Bin 83: [64844 - 65625]
  └── Count: 65359
  └── Example Value: 64922
Bin 84: [65626 - 66406]
  └── Count: 65784
  └── Example Value: 65680
Bin 85: [66407 - 67187]
  └── Count: 65690
  └── Example Value: 66698
Bin 86: [67188 - 67968]
  └── Count: 65072
  └── Example Value: 67799
Bin 87: [67969 - 68750]
  └── Count: 65707
  └── Example Value: 68099
Bin 88: [68751 - 69531]
  └── Count: 65403
  └── Example Value: 68797
Bin 89: [69532 - 70312]
  └── Count: 65416
  └── Example Value: 70004
Bin 90: [70313 - 71093]
  └── Count: 65848
  └── Example Value: 71077
Bin 91: [71094 - 71875]
  └── Count: 65116
  └── Example Value: 71473
Bin 92: [71876 - 72656]
  └── Count: 65982
  └── Example Value: 72554
Bin 93: [72657 - 73437]
  └── Count: 65652
  └── Example Value: 72936
Bin 94: [73438 - 74218]
  └── Count: 65767
  └── Example Value: 73702
Bin 95: [74219 - 75000]
  └── Count: 65708
  └── Example Value: 74687
Bin 96: [75001 - 75781]
  └── Count: 65545
  └── Example Value: 75618
Bin 97: [75782 - 76562]
  └── Count: 66217
  └── Example Value: 76506
Bin 98: [76563 - 77343]
  └── Count: 65149
  └── Example Value: 76877
Bin 99: [77344 - 78125]
  └── Count: 65263
  └── Example Value: 78124
Bin 100: [78126 - 78906]
  └── Count: 65806
  └── Example Value: 78694
Bin 101: [78907 - 79687]
  └── Count: 65296
  └── Example Value: 78913
Bin 102: [79688 - 80468]
  └── Count: 65192
  └── Example Value: 79695
Bin 103: [80469 - 81250]
  └── Count: 65723
  └── Example Value: 81087
Bin 104: [81251 - 82031]
  └── Count: 65446
  └── Example Value: 81764
Bin 105: [82032 - 82812]
  └── Count: 65009
  └── Example Value: 82067
Bin 106: [82813 - 83593]
  └── Count: 65628
  └── Example Value: 82854
Bin 107: [83594 - 84375]
  └── Count: 65123
  └── Example Value: 84177
Bin 108: [84376 - 85156]
  └── Count: 65488
  └── Example Value: 84681
Bin 109: [85157 - 85937]
  └── Count: 65802
  └── Example Value: 85308
Bin 110: [85938 - 86718]
  └── Count: 65877
  └── Example Value: 86464
Bin 111: [86719 - 87500]
  └── Count: 65493
  └── Example Value: 86967
Bin 112: [87501 - 88281]
  └── Count: 65711
  └── Example Value: 87640
Bin 113: [88282 - 89062]
  └── Count: 65707
  └── Example Value: 88782
Bin 114: [89063 - 89843]
  └── Count: 65514
  └── Example Value: 89618
Bin 115: [89844 - 90625]
  └── Count: 65658
  └── Example Value: 90047
Bin 116: [90626 - 91406]
  └── Count: 65946
  └── Example Value: 90886
Bin 117: [91407 - 92187]
  └── Count: 65478
  └── Example Value: 91795
Bin 118: [92188 - 92968]
  └── Count: 65005
  └── Example Value: 92278
Bin 119: [92969 - 93750]
  └── Count: 65463
  └── Example Value: 93019
Bin 120: [93751 - 94531]
  └── Count: 66016
  └── Example Value: 94458
Bin 121: [94532 - 95312]
  └── Count: 65661
  └── Example Value: 94696
Bin 122: [95313 - 96093]
  └── Count: 65470
  └── Example Value: 95686
Bin 123: [96094 - 96875]
  └── Count: 65103
  └── Example Value: 96635
Bin 124: [96876 - 97656]
  └── Count: 65492
  └── Example Value: 97033
Bin 125: [97657 - 98437]
  └── Count: 65316
  └── Example Value: 97749
Bin 126: [98438 - 99218]
  └── Count: 65620
  └── Example Value: 98919
Bin 127: [99219 - 100000]
  └── Count: 65673
  └── Example Value: 99473
```
</details>

##### CUDA Program Performance
- Kernel Execution Time: 0.0135063770 seconds

#### Results for 128 bins, N = \( 2^{23} \), block size = 128, grid size = 65536
<details>
<summary>Click to view full histogram results</summary>

```
CUDA Histogram Results
============================

Total elements: 8388608 | Range: 1-100000 | Number of Bins: 128
Block Size (Threads Per Block): 128 | Grid Size (Number of Blocks): 65536
Execution Time (including device sync & copy): 0.0135047550 seconds

Bin 0: [1 - 781]
  ├── Count: 65657
  └── Example Value: 470
Bin 1: [782 - 1562]
  ├── Count: 65500
  └── Example Value: 1306
Bin 2: [1563 - 2343]
  ├── Count: 65822
  └── Example Value: 1666
Bin 3: [2344 - 3125]
  ├── Count: 65479
  └── Example Value: 2458
Bin 4: [3126 - 3906]
  ├── Count: 65381
  └── Example Value: 3869
Bin 5: [3907 - 4687]
  ├── Count: 65570
  └── Example Value: 4492
Bin 6: [4688 - 5468]
  ├── Count: 65741
  └── Example Value: 5357
Bin 7: [5469 - 6250]
  ├── Count: 65360
  └── Example Value: 5980
Bin 8: [6251 - 7031]
  ├── Count: 65671
  └── Example Value: 6735
Bin 9: [7032 - 7812]
  ├── Count: 65842
  └── Example Value: 7301
Bin 10: [7813 - 8593]
  ├── Count: 65372
  └── Example Value: 7987
Bin 11: [8594 - 9375]
  ├── Count: 65626
  └── Example Value: 8861
Bin 12: [9376 - 10156]
  ├── Count: 65314
  └── Example Value: 9645
Bin 13: [10157 - 10937]
  ├── Count: 65591
  └── Example Value: 10885
Bin 14: [10938 - 11718]
  ├── Count: 65905
  └── Example Value: 10978
Bin 15: [11719 - 12500]
  ├── Count: 66003
  └── Example Value: 12159
Bin 16: [12501 - 13281]
  ├── Count: 65433
  └── Example Value: 12844
Bin 17: [13282 - 14062]
  ├── Count: 65846
  └── Example Value: 13523
Bin 18: [14063 - 14843]
  ├── Count: 65653
  └── Example Value: 14418
Bin 19: [14844 - 15625]
  ├── Count: 65577
  └── Example Value: 15508
Bin 20: [15626 - 16406]
  ├── Count: 65661
  └── Example Value: 15876
Bin 21: [16407 - 17187]
  ├── Count: 65120
  └── Example Value: 16615
Bin 22: [17188 - 17968]
  ├── Count: 65548
  └── Example Value: 17358
Bin 23: [17969 - 18750]
  ├── Count: 65523
  └── Example Value: 18447
Bin 24: [18751 - 19531]
  ├── Count: 65544
  └── Example Value: 19516
Bin 25: [19532 - 20312]
  ├── Count: 65467
  └── Example Value: 20169
Bin 26: [20313 - 21093]
  ├── Count: 65639
  └── Example Value: 20697
Bin 27: [21094 - 21875]
  ├── Count: 65279
  └── Example Value: 21121
Bin 28: [21876 - 22656]
  ├── Count: 65321
  └── Example Value: 22270
Bin 29: [22657 - 23437]
  ├── Count: 65543
  └── Example Value: 22790
Bin 30: [23438 - 24218]
  ├── Count: 65849
  └── Example Value: 24216
Bin 31: [24219 - 25000]
  ├── Count: 65476
  └── Example Value: 24658
Bin 32: [25001 - 25781]
  ├── Count: 65582
  └── Example Value: 25206
Bin 33: [25782 - 26562]
  ├── Count: 65930
  └── Example Value: 25835
Bin 34: [26563 - 27343]
  ├── Count: 65487
  └── Example Value: 27242
Bin 35: [27344 - 28125]
  ├── Count: 66058
  └── Example Value: 27879
Bin 36: [28126 - 28906]
  ├── Count: 65442
  └── Example Value: 28586
Bin 37: [28907 - 29687]
  ├── Count: 65429
  └── Example Value: 29520
Bin 38: [29688 - 30468]
  ├── Count: 65292
  └── Example Value: 30305
Bin 39: [30469 - 31250]
  ├── Count: 65624
  └── Example Value: 30774
Bin 40: [31251 - 32031]
  ├── Count: 65438
  └── Example Value: 31636
Bin 41: [32032 - 32812]
  ├── Count: 65164
  └── Example Value: 32622
Bin 42: [32813 - 33593]
  ├── Count: 65378
  └── Example Value: 33023
Bin 43: [33594 - 34375]
  ├── Count: 65940
  └── Example Value: 34296
Bin 44: [34376 - 35156]
  ├── Count: 65776
  └── Example Value: 34528
Bin 45: [35157 - 35937]
  ├── Count: 64887
  └── Example Value: 35626
Bin 46: [35938 - 36718]
  ├── Count: 65923
  └── Example Value: 36259
Bin 47: [36719 - 37500]
  ├── Count: 65692
  └── Example Value: 36990
Bin 48: [37501 - 38281]
  ├── Count: 65470
  └── Example Value: 37526
Bin 49: [38282 - 39062]
  ├── Count: 65447
  └── Example Value: 39004
Bin 50: [39063 - 39843]
  ├── Count: 65309
  └── Example Value: 39138
Bin 51: [39844 - 40625]
  ├── Count: 65604
  └── Example Value: 39917
Bin 52: [40626 - 41406]
  ├── Count: 65633
  └── Example Value: 40866
Bin 53: [41407 - 42187]
  ├── Count: 65341
  └── Example Value: 42044
Bin 54: [42188 - 42968]
  ├── Count: 65168
  └── Example Value: 42571
Bin 55: [42969 - 43750]
  ├── Count: 65576
  └── Example Value: 43038
Bin 56: [43751 - 44531]
  ├── Count: 65178
  └── Example Value: 44267
Bin 57: [44532 - 45312]
  ├── Count: 64987
  └── Example Value: 44862
Bin 58: [45313 - 46093]
  ├── Count: 65444
  └── Example Value: 45821
Bin 59: [46094 - 46875]
  ├── Count: 65833
  └── Example Value: 46677
Bin 60: [46876 - 47656]
  ├── Count: 65706
  └── Example Value: 47330
Bin 61: [47657 - 48437]
  ├── Count: 65697
  └── Example Value: 48248
Bin 62: [48438 - 49218]
  ├── Count: 65297
  └── Example Value: 48600
Bin 63: [49219 - 50000]
  ├── Count: 65841
  └── Example Value: 49832
Bin 64: [50001 - 50781]
  ├── Count: 66287
  └── Example Value: 50255
Bin 65: [50782 - 51562]
  ├── Count: 65161
  └── Example Value: 51324
Bin 66: [51563 - 52343]
  ├── Count: 65137
  └── Example Value: 51900
Bin 67: [52344 - 53125]
  ├── Count: 65292
  └── Example Value: 52889
Bin 68: [53126 - 53906]
  ├── Count: 65767
  └── Example Value: 53348
Bin 69: [53907 - 54687]
  ├── Count: 65632
  └── Example Value: 53915
Bin 70: [54688 - 55468]
  ├── Count: 65580
  └── Example Value: 55367
Bin 71: [55469 - 56250]
  ├── Count: 65058
  └── Example Value: 56056
Bin 72: [56251 - 57031]
  ├── Count: 65507
  └── Example Value: 56701
Bin 73: [57032 - 57812]
  ├── Count: 65303
  └── Example Value: 57052
Bin 74: [57813 - 58593]
  ├── Count: 65627
  └── Example Value: 58574
Bin 75: [58594 - 59375]
  ├── Count: 65338
  └── Example Value: 59190
Bin 76: [59376 - 60156]
  ├── Count: 65625
  └── Example Value: 59399
Bin 77: [60157 - 60937]
  ├── Count: 65485
  └── Example Value: 60184
Bin 78: [60938 - 61718]
  ├── Count: 66100
  └── Example Value: 61354
Bin 79: [61719 - 62500]
  ├── Count: 65433
  └── Example Value: 61764
Bin 80: [62501 - 63281]
  ├── Count: 65487
  └── Example Value: 62560
Bin 81: [63282 - 64062]
  ├── Count: 65398
  └── Example Value: 64027
Bin 82: [64063 - 64843]
  ├── Count: 65111
  └── Example Value: 64683
Bin 83: [64844 - 65625]
  ├── Count: 65359
  └── Example Value: 65558
Bin 84: [65626 - 66406]
  ├── Count: 65784
  └── Example Value: 66067
Bin 85: [66407 - 67187]
  ├── Count: 65690
  └── Example Value: 66822
Bin 86: [67188 - 67968]
  ├── Count: 65072
  └── Example Value: 67539
Bin 87: [67969 - 68750]
  ├── Count: 65707
  └── Example Value: 68233
Bin 88: [68751 - 69531]
  ├── Count: 65403
  └── Example Value: 69433
Bin 89: [69532 - 70312]
  ├── Count: 65416
  └── Example Value: 70312
Bin 90: [70313 - 71093]
  ├── Count: 65848
  └── Example Value: 70381
Bin 91: [71094 - 71875]
  ├── Count: 65116
  └── Example Value: 71711
Bin 92: [71876 - 72656]
  ├── Count: 65982
  └── Example Value: 71885
Bin 93: [72657 - 73437]
  ├── Count: 65652
  └── Example Value: 72785
Bin 94: [73438 - 74218]
  ├── Count: 65767
  └── Example Value: 73765
Bin 95: [74219 - 75000]
  ├── Count: 65708
  └── Example Value: 74603
Bin 96: [75001 - 75781]
  ├── Count: 65545
  └── Example Value: 75359
Bin 97: [75782 - 76562]
  ├── Count: 66217
  └── Example Value: 75954
Bin 98: [76563 - 77343]
  ├── Count: 65149
  └── Example Value: 77208
Bin 99: [77344 - 78125]
  ├── Count: 65263
  └── Example Value: 78004
Bin 100: [78126 - 78906]
  ├── Count: 65806
  └── Example Value: 78182
Bin 101: [78907 - 79687]
  ├── Count: 65296
  └── Example Value: 79193
Bin 102: [79688 - 80468]
  ├── Count: 65192
  └── Example Value: 79710
Bin 103: [80469 - 81250]
  ├── Count: 65723
  └── Example Value: 81004
Bin 104: [81251 - 82031]
  ├── Count: 65446
  └── Example Value: 81852
Bin 105: [82032 - 82812]
  ├── Count: 65009
  └── Example Value: 82670
Bin 106: [82813 - 83593]
  ├── Count: 65628
  └── Example Value: 83191
Bin 107: [83594 - 84375]
  ├── Count: 65123
  └── Example Value: 84316
Bin 108: [84376 - 85156]
  ├── Count: 65488
  └── Example Value: 84762
Bin 109: [85157 - 85937]
  ├── Count: 65802
  └── Example Value: 85288
Bin 110: [85938 - 86718]
  ├── Count: 65877
  └── Example Value: 86682
Bin 111: [86719 - 87500]
  ├── Count: 65493
  └── Example Value: 87347
Bin 112: [87501 - 88281]
  ├── Count: 65711
  └── Example Value: 88161
Bin 113: [88282 - 89062]
  ├── Count: 65707
  └── Example Value: 88838
Bin 114: [89063 - 89843]
  ├── Count: 65514
  └── Example Value: 89328
Bin 115: [89844 - 90625]
  ├── Count: 65658
  └── Example Value: 90597
Bin 116: [90626 - 91406]
  ├── Count: 65946
  └── Example Value: 91310
Bin 117: [91407 - 92187]
  ├── Count: 65478
  └── Example Value: 91841
Bin 118: [92188 - 92968]
  ├── Count: 65005
  └── Example Value: 92332
Bin 119: [92969 - 93750]
  ├── Count: 65463
  └── Example Value: 93081
Bin 120: [93751 - 94531]
  ├── Count: 66016
  └── Example Value: 94011
Bin 121: [94532 - 95312]
  ├── Count: 65661
  └── Example Value: 95155
Bin 122: [95313 - 96093]
  ├── Count: 65470
  └── Example Value: 95540
Bin 123: [96094 - 96875]
  ├── Count: 65103
  └── Example Value: 96620
Bin 124: [96876 - 97656]
  ├── Count: 65492
  └── Example Value: 96915
Bin 125: [97657 - 98437]
  ├── Count: 65316
  └── Example Value: 97763
Bin 126: [98438 - 99218]
  ├── Count: 65620
  └── Example Value: 99089
Bin 127: [99219 - 100000]
  ├── Count: 65673
  └── Example Value: 99532
```
</details>

##### CUDA Program Performance
- Kernel Execution Time: 0.0135047550 seconds

#### Results for 128 bins, N = \( 2^{23} \), block size = 256, grid size = 32768
<details>
<summary>Click to view full histogram results</summary>

```
CUDA Histogram Results
============================

Total elements: 8388608 | Range: 1-100000 | Number of Bins: 128
Block Size (Threads Per Block): 256 | Grid Size (Number of Blocks): 32768
Execution Time (including device sync & copy): 0.0135035340 seconds

Bin 0: [1 - 781]
  └── Count: 65657
  └── Example Value: 353
Bin 1: [782 - 1562]
  └── Count: 65500
  └── Example Value: 1071
Bin 2: [1563 - 2343]
  └── Count: 65822
  └── Example Value: 1864
Bin 3: [2344 - 3125]
  └── Count: 65479
  └── Example Value: 2815
Bin 4: [3126 - 3906]
  └── Count: 65381
  └── Example Value: 3328
Bin 5: [3907 - 4687]
  └── Count: 65570
  └── Example Value: 4063
Bin 6: [4688 - 5468]
  └── Count: 65741
  └── Example Value: 4828
Bin 7: [5469 - 6250]
  └── Count: 65360
  └── Example Value: 5772
Bin 8: [6251 - 7031]
  └── Count: 65671
  └── Example Value: 6995
Bin 9: [7032 - 7812]
  └── Count: 65842
  └── Example Value: 7305
Bin 10: [7813 - 8593]
  └── Count: 65372
  └── Example Value: 7841
Bin 11: [8594 - 9375]
  └── Count: 65626
  └── Example Value: 9051
Bin 12: [9376 - 10156]
  └── Count: 65314
  └── Example Value: 10121
Bin 13: [10157 - 10937]
  └── Count: 65591
  └── Example Value: 10777
Bin 14: [10938 - 11718]
  └── Count: 65905
  └── Example Value: 11284
Bin 15: [11719 - 12500]
  └── Count: 66003
  └── Example Value: 12308
Bin 16: [12501 - 13281]
  └── Count: 65433
  └── Example Value: 12903
Bin 17: [13282 - 14062]
  └── Count: 65846
  └── Example Value: 13935
Bin 18: [14063 - 14843]
  └── Count: 65653
  └── Example Value: 14653
Bin 19: [14844 - 15625]
  └── Count: 65577
  └── Example Value: 15223
Bin 20: [15626 - 16406]
  └── Count: 65661
  └── Example Value: 16369
Bin 21: [16407 - 17187]
  └── Count: 65120
  └── Example Value: 17076
Bin 22: [17188 - 17968]
  └── Count: 65548
  └── Example Value: 17882
Bin 23: [17969 - 18750]
  └── Count: 65523
  └── Example Value: 18279
Bin 24: [18751 - 19531]
  └── Count: 65544
  └── Example Value: 18943
Bin 25: [19532 - 20312]
  └── Count: 65467
  └── Example Value: 20215
Bin 26: [20313 - 21093]
  └── Count: 65639
  └── Example Value: 20934
Bin 27: [21094 - 21875]
  └── Count: 65279
  └── Example Value: 21673
Bin 28: [21876 - 22656]
  └── Count: 65321
  └── Example Value: 22382
Bin 29: [22657 - 23437]
  └── Count: 65543
  └── Example Value: 22751
Bin 30: [23438 - 24218]
  └── Count: 65849
  └── Example Value: 23724
Bin 31: [24219 - 25000]
  └── Count: 65476
  └── Example Value: 24260
Bin 32: [25001 - 25781]
  └── Count: 65582
  └── Example Value: 25444
Bin 33: [25782 - 26562]
  └── Count: 65930
  └── Example Value: 26268
Bin 34: [26563 - 27343]
  └── Count: 65487
  └── Example Value: 26568
Bin 35: [27344 - 28125]
  └── Count: 66058
  └── Example Value: 27844
Bin 36: [28126 - 28906]
  └── Count: 65442
  └── Example Value: 28469
Bin 37: [28907 - 29687]
  └── Count: 65429
  └── Example Value: 28985
Bin 38: [29688 - 30468]
  └── Count: 65292
  └── Example Value: 29937
Bin 39: [30469 - 31250]
  └── Count: 65624
  └── Example Value: 30692
Bin 40: [31251 - 32031]
  └── Count: 65438
  └── Example Value: 31970
Bin 41: [32032 - 32812]
  └── Count: 65164
  └── Example Value: 32292
Bin 42: [32813 - 33593]
  └── Count: 65378
  └── Example Value: 33062
Bin 43: [33594 - 34375]
  └── Count: 65940
  └── Example Value: 34137
Bin 44: [34376 - 35156]
  └── Count: 65776
  └── Example Value: 34432
Bin 45: [35157 - 35937]
  └── Count: 64887
  └── Example Value: 35707
Bin 46: [35938 - 36718]
  └── Count: 65923
  └── Example Value: 36227
Bin 47: [36719 - 37500]
  └── Count: 65692
  └── Example Value: 37343
Bin 48: [37501 - 38281]
  └── Count: 65470
  └── Example Value: 38098
Bin 49: [38282 - 39062]
  └── Count: 65447
  └── Example Value: 38726
Bin 50: [39063 - 39843]
  └── Count: 65309
  └── Example Value: 39472
Bin 51: [39844 - 40625]
  └── Count: 65604
  └── Example Value: 40029
Bin 52: [40626 - 41406]
  └── Count: 65633
  └── Example Value: 40898
Bin 53: [41407 - 42187]
  └── Count: 65341
  └── Example Value: 42178
Bin 54: [42188 - 42968]
  └── Count: 65168
  └── Example Value: 42300
Bin 55: [42969 - 43750]
  └── Count: 65576
  └── Example Value: 43456
Bin 56: [43751 - 44531]
  └── Count: 65178
  └── Example Value: 44096
Bin 57: [44532 - 45312]
  └── Count: 64987
  └── Example Value: 44726
Bin 58: [45313 - 46093]
  └── Count: 65444
  └── Example Value: 45819
Bin 59: [46094 - 46875]
  └── Count: 65833
  └── Example Value: 46620
Bin 60: [46876 - 47656]
  └── Count: 65706
  └── Example Value: 47045
Bin 61: [47657 - 48437]
  └── Count: 65697
  └── Example Value: 47870
Bin 62: [48438 - 49218]
  └── Count: 65297
  └── Example Value: 48712
Bin 63: [49219 - 50000]
  └── Count: 65841
  └── Example Value: 49651
Bin 64: [50001 - 50781]
  └── Count: 66287
  └── Example Value: 50416
Bin 65: [50782 - 51562]
  └── Count: 65161
  └── Example Value: 51234
Bin 66: [51563 - 52343]
  └── Count: 65137
  └── Example Value: 51946
Bin 67: [52344 - 53125]
  └── Count: 65292
  └── Example Value: 52631
Bin 68: [53126 - 53906]
  └── Count: 65767
  └── Example Value: 53515
Bin 69: [53907 - 54687]
  └── Count: 65632
  └── Example Value: 54370
Bin 70: [54688 - 55468]
  └── Count: 65580
  └── Example Value: 54812
Bin 71: [55469 - 56250]
  └── Count: 65058
  └── Example Value: 55608
Bin 72: [56251 - 57031]
  └── Count: 65507
  └── Example Value: 56616
Bin 73: [57032 - 57812]
  └── Count: 65303
  └── Example Value: 57545
Bin 74: [57813 - 58593]
  └── Count: 65627
  └── Example Value: 58548
Bin 75: [58594 - 59375]
  └── Count: 65338
  └── Example Value: 59374
Bin 76: [59376 - 60156]
  └── Count: 65625
  └── Example Value: 60117
Bin 77: [60157 - 60937]
  └── Count: 65485
  └── Example Value: 60799
Bin 78: [60938 - 61718]
  └── Count: 66100
  └── Example Value: 61065
Bin 79: [61719 - 62500]
  └── Count: 65433
  └── Example Value: 61824
Bin 80: [62501 - 63281]
  └── Count: 65487
  └── Example Value: 63122
Bin 81: [63282 - 64062]
  └── Count: 65398
  └── Example Value: 63771
Bin 82: [64063 - 64843]
  └── Count: 65111
  └── Example Value: 64731
Bin 83: [64844 - 65625]
  └── Count: 65359
  └── Example Value: 65556
Bin 84: [65626 - 66406]
  └── Count: 65784
  └── Example Value: 65892
Bin 85: [66407 - 67187]
  └── Count: 65690
  └── Example Value: 66468
Bin 86: [67188 - 67968]
  └── Count: 65072
  └── Example Value: 67862
Bin 87: [67969 - 68750]
  └── Count: 65707
  └── Example Value: 68552
Bin 88: [68751 - 69531]
  └── Count: 65403
  └── Example Value: 68997
Bin 89: [69532 - 70312]
  └── Count: 65416
  └── Example Value: 70013
Bin 90: [70313 - 71093]
  └── Count: 65848
  └── Example Value: 70911
Bin 91: [71094 - 71875]
  └── Count: 65116
  └── Example Value: 71854
Bin 92: [71876 - 72656]
  └── Count: 65982
  └── Example Value: 72294
Bin 93: [72657 - 73437]
  └── Count: 65652
  └── Example Value: 73318
Bin 94: [73438 - 74218]
  └── Count: 65767
  └── Example Value: 73943
Bin 95: [74219 - 75000]
  └── Count: 65708
  └── Example Value: 74268
Bin 96: [75001 - 75781]
  └── Count: 65545
  └── Example Value: 75401
Bin 97: [75782 - 76562]
  └── Count: 66217
  └── Example Value: 76321
Bin 98: [76563 - 77343]
  └── Count: 65149
  └── Example Value: 77134
Bin 99: [77344 - 78125]
  └── Count: 65263
  └── Example Value: 78033
Bin 100: [78126 - 78906]
  └── Count: 65806
  └── Example Value: 78331
Bin 101: [78907 - 79687]
  └── Count: 65296
  └── Example Value: 79570
Bin 102: [79688 - 80468]
  └── Count: 65192
  └── Example Value: 79754
Bin 103: [80469 - 81250]
  └── Count: 65723
  └── Example Value: 80527
Bin 104: [81251 - 82031]
  └── Count: 65446
  └── Example Value: 81601
Bin 105: [82032 - 82812]
  └── Count: 65009
  └── Example Value: 82525
Bin 106: [82813 - 83593]
  └── Count: 65628
  └── Example Value: 83032
Bin 107: [83594 - 84375]
  └── Count: 65123
  └── Example Value: 84113
Bin 108: [84376 - 85156]
  └── Count: 65488
  └── Example Value: 85138
Bin 109: [85157 - 85937]
  └── Count: 65802
  └── Example Value: 85446
Bin 110: [85938 - 86718]
  └── Count: 65877
  └── Example Value: 86546
Bin 111: [86719 - 87500]
  └── Count: 65493
  └── Example Value: 86852
Bin 112: [87501 - 88281]
  └── Count: 65711
  └── Example Value: 87601
Bin 113: [88282 - 89062]
  └── Count: 65707
  └── Example Value: 88377
Bin 114: [89063 - 89843]
  └── Count: 65514
  └── Example Value: 89124
Bin 115: [89844 - 90625]
  └── Count: 65658
  └── Example Value: 89993
Bin 116: [90626 - 91406]
  └── Count: 65946
  └── Example Value: 91142
Bin 117: [91407 - 92187]
  └── Count: 65478
  └── Example Value: 91530
Bin 118: [92188 - 92968]
  └── Count: 65005
  └── Example Value: 92781
Bin 119: [92969 - 93750]
  └── Count: 65463
  └── Example Value: 93011
Bin 120: [93751 - 94531]
  └── Count: 66016
  └── Example Value: 94224
Bin 121: [94532 - 95312]
  └── Count: 65661
  └── Example Value: 94779
Bin 122: [95313 - 96093]
  └── Count: 65470
  └── Example Value: 95808
Bin 123: [96094 - 96875]
  └── Count: 65103
  └── Example Value: 96684
Bin 124: [96876 - 97656]
  └── Count: 65492
  └── Example Value: 97278
Bin 125: [97657 - 98437]
  └── Count: 65316
  └── Example Value: 98336
Bin 126: [98438 - 99218]
  └── Count: 65620
  └── Example Value: 98850
Bin 127: [99219 - 100000]
  └── Count: 65673
  └── Example Value: 99278
```
</details>

##### CUDA Program Performance
- Kernel Execution Time: 0.0135035340 seconds

Since the kernel execution time doesn't seem to change when the block and grid sizes change and only when the value of `N` changes, the results gathered for the range of N = \( 2^{12} \) to \( 2^{23} \) will be condensed below.

#### Program Performance
Defaulted execution with 128 bins and a fixed block size of 128

##### N = \( 2^{12} \)
- Kernel Execution Time: 0.0001732260 seconds

##### N = \( 2^{13} \)
- Kernel Execution Time: 0.0001391720 seconds

##### N = \( 2^{14} \)
- Kernel Execution Time: 0.0001972220 seconds

##### N = \( 2^{15} \)
- Kernel Execution Time: 0.0002147840 seconds

##### N = \( 2^{16} \)
- Kernel Execution Time: 0.0003064160 seconds

##### N = \( 2^{17} \)
- Kernel Execution Time: 0.0003872090 seconds

##### N = \( 2^{18} \)
- Kernel Execution Time: 0.0006032680 seconds

##### N = \( 2^{19} \)
- Kernel Execution Time: 0.0010184730 seconds

##### N = \( 2^{20} \)
- Kernel Execution Time: 0.0018621220 seconds

##### N = \( 2^{21} \)
- Kernel Execution Time: 0.0035205460 seconds

##### N = \( 2^{22} \)
- Kernel Execution Time: 0.0068548530 seconds

##### N = \( 2^{23} \)
- Kernel Execution Time: 0.0135498230 seconds

![Kernel Execution Time vs N](CUDA/output/Kernel%20Execution%20Time%20(s)%20vs.%20N%20(Number%20of%20Elements).png)


## Part (b)

The OpenMP implementation was executed on an `Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz` on `Node c2194` of the Explorer system.




## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```