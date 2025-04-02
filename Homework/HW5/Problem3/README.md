# Problem 3 (20 Points)

Read the Ampere whitepaper provided, and then identify the key features that were introduced in the Ampere A100 architecture and compare those features against the Hopper-based H100 architecture (make sure to identify the source for the information you obtained on the H100). Please do not just repeat what you read in the Ampere whitepaper, go into more detail on each of the features you identify.

## Answers

Information for both architectures were taken from their respective whitepapers.

### Key Features of Ampere A100 Architecture
- **Third-Generation Tensor Cores**
    - New Data Types (**TF32, BF16, IEEE FP64**)

|       Precision       |        Performance       |
|:---------------------:|:------------------------:|
| Peak FP64             | 9.7 TFLOPS               |
| Peak FP64 Tensor Core | 19.5 TFLOPS              |
| Peak FP32             | 19.5 TFLOPS              |
| Peak FP16             | 78 TFLOPS                |
| Peak BF16             | 39 TFLOPS                |
| Peak TF32 Tensor Core | 156 TFLOPS \| 312 TFLOPS |
| Peak FP16 Tensor Core | 312 TFLOPS \| 624 TFLOPS |
| Peak BF16 Tensor Core | 312 TFLOPS \| 624 TFLOPS |
| Peak INT8 Tensor Core | 624 TOPS \| 1,248 TOPS   |
| Peak INT4 Tensor Core | 1,248 TOPS \| 2,496 TOPS |

- **Sparsity Support** to exploit structured sparsity in deep learning networks
    - Increases Tensor Core operation throughput
- **40 GB HBM2 and 40 MB L2 Cache**
    - Provides L2 cache residency controls
    - Adds Compute Data Compression to improve memory bandwidth and capacity
- **Multi-Instance GPU (MIG)**
    - 7 GPU Instances that can each be independentaly managed and scheduled
- **Third-Generation NVLink**
    - 50 Gbit/s per signal pair
    - 12 total NVLink connections
    - 600 GB/s total bandwidth
- Support for NVIDIA Magnium IO and Mellanox Interconnect Solutions
- **PCIe Gen 4** with SR-IOV
- Improved Error and Fault Detection, Isolation, and Containment
- **Asynchronous Copy** instructions for loading data into SM shared memory from global memory
- **Asynchronous Barrier** instructions for synchronizing threads at different granularities
- **Task Graph Acceleration** for efficient GPU scheduling

### Key Features of Hopper H100 Architecture
- **Fourth-Generation Tensor Cores**

|       Precision       |         NVIDIA H100 SXM5        |       NVIDIA H100 PCIe      |
|:---------------------:|:-------------------------------:|:---------------------------:|
| Peak FP64             | 33.5 TFLOPS                     | 25.6 TFLOPS                 |
| Peak FP64 Tensor Core | 66.9 TFLOPS                     | 51.2 TFLOPS                 |
| Peak FP32             | 66.9 TFLOPS                     | 51.2 TFLOPS                 |
| Peak FP16             | 133.8 TFLOPS                    | 102.4 TFLOPS                |
| Peak BF16             | 133.8 TFLOPS                    | 102.4 TFLOPS                |
| Peak TF32 Tensor Core | 494.7 TFLOPS \| 989.4 TFLOPS¹   | 378 TFLOPS \| 756 TFLOPS¹   |
| Peak FP16 Tensor Core | 989.4 TFLOPS \| 1978.9 TFLOPS¹  | 756 TFLOPS \| 1513 TFLOPS¹  |
| Peak BF16 Tensor Core | 989.4 TFLOPS \| 1978.9 TFLOPS¹  | 756 TFLOPS \| 1513 TFLOPS²  |
| Peak FP8 Tensor Core  | 1978.9 TFLOPS \| 3957.8 TFLOPS¹ | 1513 TFLOPS \| 3026 TFLOPS¹ |
| Peak INT8 Tensor Core | 1978.9 TOPS \| 3957.8 TOPS¹     | 1513 TOPS \| 3026 TOPS¹     |

- Also has **Sparsity Support**
    - Increases Tensor Core operation throughput
    - 3x Faster **IEEE FP64** and **FP32 processing rates** over the A100 architecture
- **DPX Instructions** to accelerate Dynamic Programming algorithms 7x compared to the A100 architecture
- **Asynchronous Execution** features with **Tensor Memory Accelerator (TMA)** units
    - Transfers large blocks of data between global and shared memory.
- **Asynchronous Transaction Barrier**
    - Atomic data movement and synchronization
- **Thread Block Cluster** feature to control locality across streaming multiprocessors
- **Distributed Shared Memory** for direct SM to SM communication over multiple shared memory blocks


### Detailed Key Features Comparison

#### Tensor Core Evolution
- **Ampere A100** introduced TF32 and improved mixed-precision support to balance performance and accuracy.
- **Hopper H100** adds FP8 support and a Transformer Engine that dynamically chooses FP8 vs. FP16, enabling up to **9x faster training** and **30x faster inference** for transformer-based models.

#### Memory and Cache Architecture
- **A100**: 40 GB HBM2 + 40 MB L2 cache, with compute data compression and L2 residency controls.
- **H100**: 80 GB HBM3 + 50 MB L2 cache, delivering **almost 2x bandwidth** and larger on-chip memory to keep more model data close to compute units.

#### Multi-Instance GPU (MIG)
- **A100 MIG** enables safe GPU partitioning into 7 isolated instances.
- **H100 MIG** improves performance per instance and introduces **Confidential Computing** via **Trusted Execution Environments (TEEs)** - critical for cloud deployments with sensitive data.

#### Asynchronous Execution
- **A100**: Introduced async copy/barrier operations.
- **H100**: Adds **Tensor Memory Accelerator (TMA)** for efficient multidimensional data transfers and **Asynchronous Transaction Barriers** for synchronized operations across SMs - leading to better utilization of compute pipelines.

#### Thread Scheduling and Clustering
- **A100**: Introduced basic cooperative groups and task graph acceleration.
- **H100**: Adds **Thread Block Clusters** and **Distributed Shared Memory**, allowing **cross-SM coordination**, better suited for tightly-coupled parallel workloads like graph algorithms and dynamic programming.

#### Specialized Instruction Sets
- **Only in H100**: **DPX instructions** provide hardware acceleration for algorithms like Smith-Waterman (bioinformatics) and Floyd-Warshall (graph analysis), offering **7x speedups** for dynamic programming tasks.

#### NVLink and PCIe
- **A100**: 3rd-gen NVLink (600 GB/s) and PCIe Gen 4.
- **H100**: 4th-gen NVLink with 900 GB/s bandwidth, **NVLink Switch System** for scaling up to **256 GPUs**, and PCIe Gen 5 for double the I/O bandwidth.


## Miscellaneous
- The Ampere whitepaper is available at: https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- The Hopper whitepaper is available at: https://resources.nvidia.com/en-us-tensor-core