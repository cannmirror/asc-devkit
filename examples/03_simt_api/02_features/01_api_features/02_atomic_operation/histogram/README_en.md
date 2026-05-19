# Histogram Sample Using SIMT Programming Model

## Overview

This sample demonstrates how to use the `asc_atomic_add` interface of Ascend C SIMT to efficiently count the frequency of each byte value in an input byte sequence. The functionality is illustrated below:</br>
</br><img src="figure/introduction.png" alt="intro" style="width: 50%; height: auto;"></br>

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions

- \> CANN 9.0.0

## Directory Structure

```text
├── histogram
│   ├── CMakeLists.txt         # Sample build script
│   ├── histogram.asc          # Histogram sample implementation
│   └── README.md
```

## Sample Description

- Sample functionality:
  The Histogram sample processes a fixed-size input byte stream and counts the frequency of each byte value (0-255). The input data shape is `294912`.

- Sample specifications:
  <table>
  <tr><td align="center">Sample Type (OpType)</td><td colspan="4" align="center">histogram</td></tr>
  <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">294912</td><td align="center">uint8</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Sample Output</td></tr>
  <tr><td align="center">actual_histogram</td><td align="center">256</td><td align="center">uint32</td><td align="center">ND</td></tr>
  </table>

- Data partitioning:
  * Phase 1 Kernel `accumulate_block_local_histogram`:
    * Number of ThreadBlocks: Uses the maximum physical core count of `72`.
    * Threads per ThreadBlock: Uses the default value of 1024.
    * Per-thread processing: Traverses `uint32` input using grid-stride, parsing 4 bytes at a time and updating the local histogram of the associated Warp.
  * Phase 2 Kernel `merge_block_local_histogram`:
    * Number of ThreadBlocks: `256`, with each ThreadBlock responsible for one bin (bucket). A bucket represents a single counting unit in the histogram array.
    * Threads per ThreadBlock: To use the reduction algorithm, uses `128`, which is the first power of 2 greater than 72.
    * Per-thread processing: Each thread reads the count for the same bucket from multiple local histograms and accumulates them, then performs reduction within the ThreadBlock to obtain the final count.
  * Expected result: The 256 bucket counts output by the NPU should match the results computed by the CPU using the same input.

## Sample Implementation
The overall process consists of two Kernels. In the first phase, the input is partitioned among ThreadBlocks, with each ThreadBlock generating a local histogram. In the second phase, each ThreadBlock merges all local histograms by bucket to produce the final 256-bucket result.

### Phase 1

Overall process: Retrieve input from GM, then partition and distribute the input data to corresponding ThreadBlocks. Each ThreadBlock further partitions the input data and distributes it to corresponding Warps. Each Warp is responsible for computing a local histogram. The count for each bucket in the local histogram is accumulated atomically in UB using `asc_atomic_add`. Finally, the local histograms computed by each Warp are merged to produce the ThreadBlock's local histogram, which is then written back to GM. The flow diagram is as follows:

<img src="figure/local_his.png" alt="local_his" style="width: 50%; height: auto;">

Notes:
  1. Since UB has lower read/write latency, histogram count write operations are performed in UB.
  2. The maximum available UB space is 216KB. The current configuration uses 72 cores, with each core using the default 1024 threads (32 Warps). Since each Warp requires 1KB of space, 32 Warps need a total of 32KB, which can fit entirely in UB.
  3. If all Warps in a ThreadBlock maintain a single local histogram, severe thread conflicts would occur, affecting performance. Therefore, each Warp maintains its own local histogram. Threads within the current Warp only compete to write to their corresponding local histogram. Threads within a Warp read input using grid-stride and increment the count for each byte's corresponding bucket by 1 using `asc_atomic_add`. Then, all Warp local histograms are merged within the ThreadBlock. The `asc_atomic_add` is used for atomic accumulation of bucket counts in a Warp's local histogram, ensuring correct updates when multiple threads simultaneously hit the same bin.</br>

### Phase 2

Overall process: Retrieve local histogram data from GM. Each ThreadBlock is responsible for summing one bucket across all local histograms. Each thread writes the count for the corresponding bucket from local histograms to UB, then performs reduction summation in UB, and finally writes the result to GM. The flow diagram is as follows: The following figure uses Block0 as an example. Block0 is responsible for computing the count for bucket index 0, that is, bin[0]. Each thread in this Block writes the bin[0] from the corresponding local histogram to UB. For example, thread 0 writes bin[0] from local histogram 0 to index 0 in UB, thread 1 writes bin[0] from local histogram 1 to index 1 in UB, and so on. Finally, UB stores the bin[0] counts from each local histogram, and reduction summation is performed on the UB data to obtain the total bin[0], which is then written back to GM.

<img src="figure/merge_his.png" alt="merge_his" style="width: 50%; height: auto;"></br>

## Compilation and Execution
Follow the steps below in the sample root directory to compile and execute the sample.
- Configure environment variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
  - Default path, CANN software package installed by root user
     ```bash
     source /usr/local/Ascend/cann/set_env.sh
     ```

  - Default path, CANN software package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installed
    ```bash
    source ${install_path}/cann/set_env.sh
    ```


- Sample execution

  ```bash
  mkdir -p build && cd build      # Create and enter build directory
  cmake ..                        # Configure project
  make -j                         # Compile sample
  ./histogram                     # Execute sample
  ```

  If the execution result is as follows, the accuracy comparison has passed.
  
  ```text
  Running histogram256 on Ascend C SIMT for fixed xxx bytes
  Validation passed
  ```