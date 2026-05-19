# MrgSort Example

## Overview

This example implements multi-way merge sort functionality in a sorting scenario using Sort32 and MrgSort interfaces. First, Sort32 is called to preprocess data in parallel into multiple sorted subsequences (each group of 32 elements is sorted in descending order, forming sorted queues stored in an alternating (score, index) structure); then MrgSort is called to merge these subsequences into a globally sorted result.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── mrg_sort
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── mrg_sort.asc            // Ascend C implementation & invocation
```

## Scenario Description

This example selects different scenarios via the compilation parameter `SCENARIO_NUM`. All scenarios use ND data format with kernel function name `vec_mrgsort_kernel`.

**Scenario 1: 4-Queue Sorting**
- Input: 128 float elements (score) + 128 uint32 elements (index)
- Output: [1, 256] float elements (128*2 sorted results)
- Implementation: Sort32 repeat=4, divides 128 elements into 4 groups sorted in descending order; MrgSort validBit=0b1111, ifExhaustedSuspension=false, repeatTimes=1, merges 4 queues into 1 sorted queue
- Description: Demonstrates the scenario of completely merging 4 queues into 1 sorted queue

**Scenario 2: 3-Queue Non-4-Aligned Merge**
- Input: 96 float elements (score) + 96 uint32 elements (index)
- Output: [1, 192] float elements (96*2 sorted results)
- Implementation: Sort32 repeat=3, divides 96 elements into 3 groups sorted in descending order; MrgSort validBit=0b0111, ifExhaustedSuspension=false, repeatTimes=1, merges 3 queues into 1 sorted queue
- Description: Demonstrates merging in non-4-aligned situations, validBit=0b0111 indicates the first 3 queues are valid, the 4th queue length is set to 0

**Scenario 3: 32-Queue Multi-Round Merge Sort**
- Input: 1024 float elements (score) + 1024 uint32 elements (index)
- Output: [1, 2048] float elements (1024*2 sorted results)
- Implementation: Sort32 repeat=32, divides 1024 elements into 32 groups sorted in descending order; First round MrgSort repeatTimes=8, every 4 queues merge into 1, resulting in 8 sorted queues; Second round MrgSort repeatTimes=2, every 4 queues merge into 1, resulting in 2 sorted queues; Third round MrgSort2 merges 2 queues into 1 globally sorted queue
- Description: Demonstrates multi-way merge sort scenario, merging 32 sorted queues into 1 globally sorted queue through multiple rounds of merging

## Example Specifications

<table border="2">
<caption>Table 1: Example Input/Output Specifications (Scenario 1)</caption>
<tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">[1, 128]</td><td align="center">uint32</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Example Output</td><td align="center">output</td><td align="center">[1, 256]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">vec_mrgsort_kernel</td></tr>
</table>

<table border="2">
<caption>Table 2: Example Input/Output Specifications (Scenario 2)</caption>
<tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 96]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">[1, 96]</td><td align="center">uint32</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Example Output</td><td align="center">output</td><td align="center">[1, 192]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">vec_mrgsort_kernel</td></tr>
</table>

<table border="2">
<caption>Table 3: Example Input/Output Specifications (Scenario 3)</caption>
<tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">uint32</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Example Output</td><td align="center">output</td><td align="center">[1, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">vec_mrgsort_kernel</td></tr>
</table>

## Build and Run

Execute the following steps in the example root directory to build and run the example.

- Configure environment variables  
  Select the appropriate command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit in your current environment.
  - Default path, CANN package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Example execution

  ```bash
  SCENARIO_NUM=1
  mkdir -p build && cd build;                             # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j;         # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # Generate test input data
  ./demo                                                   # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM output/output.bin output/golden.bin   # Verify output correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # cpu debug mode
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build options description

  | Option | Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2`, `3` | Scenario number |

- Execution result

  The following result indicates successful accuracy comparison.

  ```bash
  test pass!
  ```