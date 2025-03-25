# Problem 4 (25 Extra Credit Points)

(Extra credit for everyone) Part of your weekly reading included a paper titled "MPI on Millions of Cores." Given that this paper was published in 2010 (15 years ago), can you comment on what changes have occurred since 2010 that could positively and/or negatively impact our ability to fully exploit parallelism on millions of cores? Many of the papers today discuss exascale-computing. Select a recent paper on exascale-computing and compare/contrast the barriers identified in the two papers that impact our ability to achieve these milestones.

**This problem is worth 25 points of extra credit for the undergraduates and PlusOne students in the class and 15 points of extra credit for the graduate students.**

## Answers

### Positive Developments

##### High-Speed Interconnects and Network Topologies
Modern supercomputers now incorporate much faster and more efficient interconnects like NVIDIA’s NVLink, AMD's Infinity Fabric, and Cray’s Slingshot. These reduce latency and increase bandwidth between nodes. Dragonfly and fat-tree topologies have become more common in large-scale systems, offering reduced hop counts and congestion for inter-node communication.

##### Advanced Communication Protocols and MPI Enhancements
MPI itself has matured, with later versions introducing improvements like non-blocking collectives, Remote Memory Access (RMA), and persistent communication requests. These changes help reduce synchronization overhead and improve scalability.

##### Larger and Smarter Caches
Cache hierarchies have grown deeper and smarter, with hardware prefetching, non-uniform cache architectures (e.g. NUMA-aware design), and larger shared L3 caches. These enhancements reduce cache misses and memory access latencies, especially important when scaling across many threads or cores.

##### Manycore Architectures and Heterogeneous Computing
CPUs now often include dozens of cores, and GPUs with thousands of cores have become integral to exascale-class systems. The rise of heterogeneous computing (e.g. CPU+GPU/APU) has led to the development of unified programming frameworks (e.g. SYCL, Kokkos, RAJA) that abstract hardware complexity and improve productivity.

##### Programming Models and Abstractions
In addition to traditional MPI+OpenMP, new models such as OpenACC, OpenMP offload, CUDA, and oneAPI have made it easier to write portable, scalable parallel code. These models help reduce the programming burden and improve resource utilization on hybrid systems.

##### Resilience and Fault Tolerance Mechanisms
As system sizes have grown, the probability of faults has increased. To mitigate this, systems now include hardware-level ECC, software-level checkpoint/restart mechanisms, and proactive fault-tolerant runtime environments.

### Negative Developments

##### Memory Bandwidth and Latency Bottlenecks
While compute capabilities have grown, memory bandwidth has not scaled at the same rate. This memory wall becomes especially problematic on manycore systems where many threads compete for shared memory access.

##### Energy and Power Constraints
Power consumption becomes a bottleneck at exascale. Techniques like dynamic voltage/frequency scaling (DVFS) and power-aware scheduling are critical, but they can limit performance if not used effectively.

##### Software Scalability and Legacy Code
Many legacy HPC applications do not scale efficiently beyond tens of thousands of cores. Refactoring or rewriting for modern parallelism (e.g. to utilize GPUs) remains a non-trivial and time-consuming process.

##### Load Balancing and Communication Overhead
At massive scale, even small load imbalances or communication inefficiencies get amplified. MPI synchronization and collective operations become increasingly expensive at scale.

### Comparison of Barriers Between Papers
|            **Barrier**           |                            **MPI on Millions (2010)**                            |                                           **Recent Exascale Papers (2024)**                                           |
|:--------------------------------:|:--------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|
| **Scalability of communication** | Point-to-point latency and synchronization cost increase at scale.               | Still a challenge - requires new communication models, lightweight OS kernels, and improved MPI collectives             |
| **Software complexity**          | Difficulty in designing applications that scale to millions of processes.        | Exascale software must manage heterogeneity (CPU+GPU/APU), massive concurrency, and fault tolerance                       |
| **Load imbalance**               | Irregular communication and computation patterns impact performance.             | Heterogeneous hardware adds more complexity-balancing workloads across CPUs, GPUs, accelerators remains a problem     |
| **Fault tolerance**              | Not a major focus in 2010, but becoming critical with scale.                     | A top challenge now. Node failures are expected frequently; requires resilient programming models and runtime systems |
| **Power and energy efficiency**  | Minimally addressed in 2010.                                                     | Power is a primary bottleneck. Systems now must be energy-aware (targeting <20–30MW for exascale)                     |
| **Memory and I/O bottlenecks**   | Memory latency and bandwidth constraints affect scalability.                     | Heterogeneous hardware adds more complexity - balancing workloads across CPUs, GPUs, accelerators remains a problem.    |
| **Programming models**           | MPI-only approach leads to excessive synchronization and communication overhead. | Emphasis now on hybrid models: MPI+X (OpenMP, CUDA, SYCL), performance-portable libraries, task-based runtimes        |

#### Sources Cited

Both links are of the same general research by the same authors:
- https://www.researchgate.net/publication/381392387_Study_of_exascale_computing_Advancements_challenges_and_future_directions
- https://pubs.aip.org/aip/acp/article/3149/1/030035/3308522/Exploring-the-potential-of-exascale-computing