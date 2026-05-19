# PairReduceSum Example

## Overview

This example implements pair-wise reduction in a reduction scenario using the PairReduceSum interface. It sums adjacent pairs of elements (a1, a2, a3, a4, a5, a6...) into (a1+a2, a3+a4, a5+a6, ...), i.e., summing each pair (adjacent element pair consisting of even and odd indices), with the output element count being half of the input.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── pair_reduce_sum
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── pair_reduce_sum.asc     // Ascend C implementation & invocation
```

## Example Description

- Example functionality:  
  This example calls the PairReduceSum interface to sum all adjacent element pairs in the input. The example specifications are shown in the table below:

  <table border="2">
  <caption>Table 1: PairReduceSum Example Specifications</caption>
  <tr>
  <td rowspan="1" align="center">Example Type (OpType)</td>
  <td colspan="4" align="center">PairReduceSum</td></tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
   <tr><td rowspan="2" align="center">Example Output</td></tr>
   <tr><td align="center">y</td><td align="center">[1, 64]</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">pair_reduce_sum_custom</td></tr>
  </table>

- Example implementation:  
  This example implements PairReduceSum with fixed shape: input x[1, 128], output y[1, 64].

  The Compute task is responsible for summing each pair of adjacent elements (even index and odd index) in srcLocal and storing the results in dstLocal.

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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # cpu debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build options description

  | Option | Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution result

  The following result indicates successful accuracy comparison.

  ```bash
  test pass!
  ```