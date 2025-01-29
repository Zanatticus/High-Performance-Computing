# Problem 3 (10 Points)

Select a node on the Explorer Cluster and report the following information about that node: (please do not ask the system admins for this information)

&nbsp;&nbsp;&nbsp;&nbsp;a. The CPU model
&nbsp;&nbsp;&nbsp;&nbsp;b. The cache memory hierarchy, including the size and associativity.
&nbsp;&nbsp;&nbsp;&nbsp;c. The main memory size of this node.
&nbsp;&nbsp;&nbsp;&nbsp;d. The version of Linux installed.

*Answers to this question should be included in your homework 1 write-up in pdf format.


*Obtaining Node **c0745** on the Explorer Cluster*

## Part (a)
Node CPU Model: `Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz`

*```lscpu | grep "Model name"```*

## Part (b)
#### Node cache memory hierarchy
**Caches:**
- L1d:
    - Size: 896 KiB (28 instances)
    - Associativity: 8-way set associative
- L1i:
    - Size: 896 KiB (28 instances)
    - Associativity: 8-way set associative
- L2:
    - Size: 7 MiB (28 instances)
    - Associativity: 8-way set associative
- L3:
    - Size: 70 MiB (2 instance)
    - Associativity: 20-way set associative

*```lscpu -C```*

## Part (c)
**Node Main Memory Size:** 251 GiB (263358456 bytes)

*```free``` and ```free -h```*

## Part (d)
**Node Linux Kernel Version:** Rocky Linux 9.3 (Blue Onyx)

*```cat /etc/os-release```*