# Problem 3

Obtaining Node **c0745** on the Explorer Cluster

## Part (a)
Node CPU Model: `Intel(R) Xeon(R) CPU E5-2680 v4 @ 2.40GHz`

*```lscpu | grep "Model name"```*

## Part (b)
#### Node cache memory hierarchy
**Caches:**
- L1d:
    - Size: 896 KiB (28 instances)
    - Associativity: 8
- L1i:
    - Size: 896 KiB (28 instances)
    - Associativity: 8
- L2:
    - Size: 7 MiB (28 instances)
    - Associativity: 8
- L3:
    - Size: 70 MiB (2 instance)
    - Associativity: 20

*```lscpu -C```*

## Part (c)
**Node Main Memory Size:** 251 GiB (263358456 bytes)

*```free``` and ```free -h```*

## Part (d)
**Node Linux Kernel Version:** Rocky Linux 9.3 (Blue Onyx)

*```cat /etc/os-release```*