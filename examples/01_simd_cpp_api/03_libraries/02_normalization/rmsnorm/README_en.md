# RmsNorm Sample

## Overview

This sample is based on Kernel direct call sample project, introducing calling RmsNorm high-level API to implement rmsnorm single sample, implementing RmsNorm normalization on input data with shape size [B, S, H].

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── rmsnorm
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── rmsnorm.asc             // Ascend C sample implementation & call sample
```

## Sample Description

- Sample Function:
  Implement RmsNorm normalization on input data with shape size [B, S, H], with computation formula as follows:
  $$
  y_i = RmsNorm(x_i)\\
  y_i=\frac{x_i}{\sqrt{\frac{1}{N}\sum_{i = 1}^{N}x_i^2+\varepsilon}}\times\gamma
  $$


- Sample Specification:
  <table>
  <tr><td rowspan="1" align="center">Sample Type(OpType)</td><td colspan="4" align="center"> rmsnorm </td></tr>

  <tr><td rowspan="4" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[4, 8, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">gamma</td><td align="center">[64]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="2" align="center">Sample Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[4, 8, 64]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">rmsnorm_custom</td></tr>
  </table>

- Sample Implementation:

  This sample does not use basic block mode (isBasicBlock=false), please refer to RmsNorm API documentation for detailed explanation.

  - Kernel Implementation
    Computation logic is:
    Input data needs to be transferred to on-chip storage first, then use RmsNorm high-level API interface to complete rmsnorm computation, obtain final result, then transfer out to external storage.

  - Tiling Implementation

    Sample Tiling implementation flow is as follows:
    1. AscendC::GetRmsNormMaxMinTmpSize gets RmsNorm interface computation required maximum and minimum temporary space size, then calculate usage size stackSize according to mode, then call AscendC::GetRmsNormTilingInfo to get kernel side interface required Tiling parameters based on input shape and workspace size.
    2. Encapsulate Tiling parameters into RmsNormTilingData structure, pass to Kernel side for use.

  - Call Implementation
    Use kernel call operator <<<>>> to call kernel function.

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..;make -j;             # Build project (default npu mode)
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute compiled executable program to run sample
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build mode, need to clear cmake cache. Can execute `rm CMakeCache.txt` in build directory and then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

  The execution result shown below indicates the accuracy comparison succeeded.
  ```bash
  test pass!
  ```