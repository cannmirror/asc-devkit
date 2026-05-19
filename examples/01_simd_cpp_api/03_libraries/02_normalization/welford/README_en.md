# Welford Example

## Overview

This example is based on the Kernel direct call example project and demonstrates how to continuously call the WelfordUpdate and WelfordFinalize high-level APIs in one kernel function to implement the complete Welford online algorithm. Welford is a method for online computation of mean and variance. It can incrementally compute mean and variance in a single data pass, reducing memory access次数 and improving computational performance.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── welford
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── welford.asc             // Ascend C example implementation & call example
```

## Example Description

- Example Function:  
  This example calls the WelfordUpdate and WelfordFinalize high-level APIs sequentially in one kernel function to implement the complete Welford online algorithm. Advantages of the Welford algorithm:
  - Can incrementally compute mean and variance of all samples without storing all samples
  - Only requires one pass through the data, reducing memory access count and improving computational performance
  - WelfordUpdate: Online update of mean and variance, computation formula is $M_t = M_{t-1} + (x_t - M_{t-1}) / n$, $S_t = S_{t-1} + (x_t - M_{t-1}) \times (x_t - M_t)$
  - WelfordFinalize: Aggregate results from multiple blocks to get final mean and variance, computation formula is $Mean = \frac{\sum(M_i \cdot n_i)}{\sum n_i}$, $Var = \frac{\sum(S_i + (Mean_i - Mean)^2 \cdot n_i)}{\sum n_i}$

- Example Specifications:  
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> welford </td></tr>

  <tr><td rowspan="5" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">srcGm</td><td align="center">[1, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">inMeanGm</td><td align="center">[1, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">inVarGm</td><td align="center">[1, 64]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="5" align="center">Example Output</td></tr>
  <tr><td align="center">outMeanGm</td><td align="center">[1, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outVarGm</td><td align="center">[1, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">finalMeanGm</td><td align="center">[8]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">finalVarGm</td><td align="center">[8]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">welford_coop_custom</td></tr>
  </table>

- Example Implementation:  
  This example implements a fused Welford example with fixed shape (RN=1, AB=64).

  This example calls two APIs in a single kernel function:
    1. WelfordUpdate: Compute local mean and variance for each data block
    2. WelfordFinalize: Aggregate mean and variance from all blocks to get global mean and variance

  The Welford algorithm formulas have been described in the example function. For details, refer to the WelfordUpdate API documentation and WelfordFinalize API documentation.

  - Kernel Implementation

    The computation logic is:  
    Call the WelfordUpdate and WelfordFinalize high-level APIs sequentially within one kernel function to implement the complete Welford online algorithm, and move the computation results to external storage.

  - Tiling Implementation

    The tiling implementation flow of the example is as follows:
    1. Call AscendC::GetWelfordUpdateMaxMinTmpSize to get the minimum temporary space size required for WelfordUpdate interface computation, and use the minimum value as the workspace size to ensure correct functionality.
    2. Copy Tiling data from GM to kernel side through the CopyTilingData function.

  - Invocation Implementation  
    Use the kernel call operator <<<>>> to call the kernel function.

## Build and Run  

Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables  
  Select the corresponding command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
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
    
- Example Execution
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..;make -j;             # Build project (default npu mode)
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

  The following output indicates that the accuracy comparison passed.
  ```bash
  test pass!
  ```