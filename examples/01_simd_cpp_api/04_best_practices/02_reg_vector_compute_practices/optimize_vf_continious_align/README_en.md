# optimize_vf_continious_align Sample

## Overview

This sample demonstrates operator implementation using continuous non-aligned transfer interfaces LoadUnAlign/StoreUnAlign for transfer optimization in SIMD scenarios. The optimization includes the following two points:

1. Move LoadUnAlignPre outside the loop
2. Move StoreUnAlignPost outside the loop

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── optimize_vf_continious_align
│   ├── CMakeLists.txt                           // Build project file
│   └── optimize_vf_continious_align.asc         // AscendC operator implementation & call sample
```

## Operator Description

- Operator Function:
  Transfer 1024 float numbers by calling continuous non-aligned transfer interfaces. During the transfer, two optimization points are involved:
  1. Move LoadUnAlignPre outside the loop
  2. Move StoreUnAlignPost outside the loop

- Operator Specification:
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="3" align="center">AIV Operator</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">y</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">optimize_vf_continious_align_kernel</td></tr>
  </table>

- Operator Implementation:
  Transfer 1024 float numbers by calling continuous non-aligned transfer interfaces. During the transfer, two optimization points are involved:
  1. Move LoadUnAlignPre outside the loop
  2. Move StoreUnAlignPost outside the loop

  - Call Implementation
    Use the kernel call operator <<<>>> to call the kernel function.

## Build and Run

Execute the following steps in the sample root directory to build and run the operator.

- Configure Environment Variables
  Select the corresponding environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in the current environment.

  - Default path, root user installed CANN software package
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN software package
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN software package
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample Execution
  ```bash
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake ..;make -j;                                                         # Build project
  ./demo                                                                    # Execute the compiled executable program to run the sample
  ```
  The execution result shown below indicates the accuracy comparison succeeded.
  ```bash
  test pass!
  ```