# Reduction Sum Sample Using SIMT Programming

## Overview

This sample implements reduction sum for one-dimensional `float` input using Ascend C SIMT programming, demonstrating typical usage of intra-thread-block synchronization and inter-thread-block memory ordering control through 2 progressive scenarios.

The 2 scenarios correspond to reduction sum for small shape input and large shape input, respectively, focusing on the usage of `asc_syncthreads()` and `asc_threadfence()` under different reduction scales.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \> CANN 9.0.0

## Directory Structure

```text
├── memory_fence
│   ├── sync_barrier.asc      # Scenario 1: Small shape reduction sample, demonstrating intra-thread-block synchronization usage of asc_syncthreads()
│   ├── memory_fence.asc      # Scenario 2: Large shape reduction sample, demonstrating inter-thread-block coordination usage of asc_threadfence() and atomic counting
│   ├── CMakeLists.txt        # Build project file
│   └── README.md
```

## Sample Description

### Sample Functionality

<table border="1" align="center">
  <tr>
    <td align="center">SCENARIO_NUM Value</td>
    <td align="center">Functional Scenario</td>
    <td align="center">Scenario Description</td>
    <td align="center">Corresponding File</td>
  </tr>
  <tr>
    <td align="center">1</td>
    <td align="center">Small shape reduction scenario</td>
    <td align="center">Use asc_syncthreads() for intra-thread-block reduction synchronization</td>
    <td align="center">sync_barrier.asc</td>
  </tr>
  <tr>
    <td align="center">2</td>
    <td align="center">Large shape reduction scenario</td>
    <td align="center">Use asc_threadfence() to merge partial sums across multiple thread blocks</td>
    <td align="center">memory_fence.asc</td>
  </tr>
</table>

This sample controls build branches through `SCENARIO_NUM`. The 2 scenarios are categorized by input scale: Scenario 1 demonstrates reduction sum for small shape input and `asc_syncthreads()` synchronization usage. When input elements are few, a single block is sufficient to cover all data, eliminating the need for multi-block synchronization. Scenario 2 extends to multi-block segmented reduction for large shape input, introducing `asc_threadfence()` for inter-block merging.

### Sample Specifications

#### SCENARIO_NUM=1 (Small shape reduction scenario)

- Sample functionality:

  Perform **reduction sum** on 1024 `float` input elements within a single thread block.

- Sample specifications:

  <table border="1" align="center">
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">SyncBarrierSingleBlock</td></tr>
  <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">[1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">output</td><td align="center">[1]</td><td align="center">float</td><td align="center">ND</td></tr>
  </table>

- Data partitioning:

  - gridDim: (1, 1, 1)
  - blockDim: (1024, 1, 1)
  - Per-thread processing: 1 input element

#### SCENARIO_NUM=2 (Large shape reduction scenario)

- Sample functionality:

  Perform **segmented reduction sum** on `1024 * 1024` `float` input elements. Each thread block first completes a **partial sum** for its assigned segment, then a single thread block performs a second reduction sum on all thread blocks' partial sums to obtain the final total.

- Sample specifications:

  <table border="1" align="center">
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">ThreadFenceMultiBlock</td></tr>
  <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">[1024 * 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">output</td><td align="center">[1]</td><td align="center">float</td><td align="center">ND</td></tr>
  </table>

- Data partitioning:

  - gridDim: (1024, 1, 1)
  - blockDim: (1024, 1, 1)
  - Per-thread processing: 1 input element

### Sample Implementation

#### 1: Small shape reduction scenario

Each thread in a single thread block first reads one input element and obtains the final result through **two-phase reduction sum**:

- **Phase 1 (intra-warp reduction)**: Call the `asc_reduce_add()` interface to reduce values from all threads in the current warp, obtain the sum of data within the warp, and write it to shared memory.
- **Phase 2 (block-level sequential accumulation)**: After cross-warp synchronization, thread 0 sequentially accumulates the partial sums from each warp to obtain the final result.

Taking 128 elements (4 warps) as an example, the two-phase reduction process is shown in Figure 1:

<p align="center">
  <img src="./figure/两阶段归约求和.png" width="50%">
   </p>
<p align="center">
Figure 1: Two-phase reduction process diagram
</p>

In Phase 1, each warp completes intra-warp summation through `asc_reduce_add()` and writes the result to shared memory. In Phase 2, thread 0 sequentially reads each warp's partial sum and accumulates them to obtain the final result. `asc_syncthreads()` must be called for synchronization between the two phases.

`asc_syncthreads()` is used to block all threads in the current thread block until all threads have reached the synchronization point. In this scenario, `asc_syncthreads()` ensures that the first thread of each warp has written the intra-warp reduction result to shared memory before the subsequent block-level sequential accumulation can read the complete shared memory data.

#### 2: Large shape reduction scenario

When the total number of input elements is large, **multiple thread blocks** are needed for segmented processing.

Each thread block first completes the above two-phase local reduction within the block, then the thread with `tid = 0` writes the thread block's partial sum to `block_sums[blockIdx.x]`. After writing, execute `asc_threadfence()`, then increment the global counter by 1. The last thread block to increment the counter (that is, the thread block with `ticket = gridDim.x - 1`) reads `block_sums` and performs the second two-phase reduction, outputting the final result.

Taking 8 thread blocks as an example, the cross-thread-block coordination process is shown in Figure 2:

<p align="center">
  <img src="./figure/跨线程块协作.png" width="75%">
   </p>
<p align="center">
Figure 2: Cross-thread-block coordination process diagram
</p>

In the above process, multiple thread blocks need to read and write the same global memory `block_sums`, which may cause data races.

`asc_threadfence()` serves as an inter-core memory barrier, forcibly guaranteeing that memory write operations before and after it are visible to other cores and that their order is not reordered. Therefore, each thread block must execute in the following order: write partial sum to `block_sums` → increment atomic counter. This order ensures that when the last thread block reads `block_sums` to perform global reduction, it will definitely see the partial sums already written by all other thread blocks. If `asc_threadfence()` is omitted, dirty data may be read, resulting in incorrect results.

## Compilation and Execution

Follow the steps below in the sample root directory to compile and execute the operator.
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
  SCENARIO_NUM=1 # Value can be 1 or 2
  mkdir -p build && cd build;   # Create and enter build directory
  cmake .. -DSCENARIO_NUM=$SCENARIO_NUM; make -j;            # Build project
  ./demo                      # Execute sample
  ```
  If the execution result is as follows, the accuracy comparison has passed.
  ```
  [Success] Case accuracy is verification passed.
  ```