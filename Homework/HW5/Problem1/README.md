# Problem 1 (40 Points)

Let us revisit the histogramming problem assigned in Homework 4. As in Homework 4, your input to this program will be integers in the range 1-100,000 this time (use a random number generator that generates the numbers on the host). Your host-based data set should contain N integers, where N can be varied in the program.

a. This time you will implement a histogramming kernel in CUDA and run on a single GPU (P100 or V100). You can choose how to compute each class of the data set on the GPU. Attempt to adjust the grid size to get the best performance as you vary N. Experiment with \( N = 2^{12} \) to \( 2^{23} \). When the GPU finishes, print one element from each class in class ascending order on the host (do not include the printing time in the timing measurements, though do include the device-to-host communication in the timing measurement). Plot your results in a graph.

b. Compare your GPU performance with running this same code on a CPU using OpenMP.

## Part (a)

This program was run on the `NVIDIA Tesla P100 GPU` on `Node c2187` of the Explorer System.

### Results for 128 bins, N = \( 2^{23} \), block size = 256, grid size = 32768
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

##### CUDA Program Performance:
- Kernel Execution Time: 0.0135035340 seconds

### Results for 128 bins, N = \( 2^{23} \), block size = 64, grid size = 131072
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

##### CUDA Program Performance:
- Kernel Execution Time: 0.0135063770 seconds


## Part (b)



## Miscellaneous
- The program was compiled and run using the following command within the makefile directory:
```make```