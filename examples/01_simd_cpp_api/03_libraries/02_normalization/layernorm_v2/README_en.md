# LayerNormV2 Example

## Overview

This example is based on the Kernel direct call example project and demonstrates how to continuously call the LayerNorm and Normalize high-level APIs in one kernel function to perform row-wise normalization on the input tensor, and compare the output results of both APIs.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure
 
```
├── layernorm_v2
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── layernorm_v2.asc        // Ascend C example implementation & call example
```

## Example Description

- Example Function:  
  This example calls the LayerNorm and Normalize high-level APIs sequentially in one kernel function. Both APIs work together to implement the complete normalization computation. LayerNorm computes the mean and reciprocal of standard deviation (rstd), then computes the variance based on rstd using the formula:
  $$
  var = 1/(rstd*rstd) - \epsilon
  $$
  Finally, Normalize uses the original input inputX from LayerNorm, the mean computed by LayerNorm, and the variance calculated from rstd as inputs to perform normalization computation. The LayerNorm computation formula is as follows:
  $$
  y_i = \gamma_i \cdot \frac{x_i - \mu}{\sqrt{var + \epsilon}} + \beta_i
  $$

- Example Specifications:  
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> layernorm_v2 </td></tr>

  <tr><td rowspan="5" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">inputXGm</td><td align="center">[32, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">gammaGm</td><td align="center">[32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">betaGm</td><td align="center">[32]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="6" align="center">Example Output</td></tr>
  <tr><td align="center">outputGm</td><td align="center">[32, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outputMeanGm</td><td align="center">[32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outputRstdGm</td><td align="center">[32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">normalizeOutputGm</td><td align="center">[32, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">normalizeRstdGm</td><td align="center">[32]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">layernormv2_custom</td></tr>
  </table>

- Example Implementation:  
  This example uses the **AR format** (shape is [A, R]) and calls the LayerNorm and Normalize high-level APIs sequentially in one kernel function to perform normalization computation. A is the batch axis, and R is the normalization axis. For details, refer to the LayerNorm API documentation and Normalize API documentation.

  - Kernel Implementation

    This example implements complete data flow computation within one kernel function:
    1. LayerNorm forward computation: Input inputX, gamma, beta, compute output y, mean, and reciprocal of standard deviation rstd.
    2. Normalize computation: Use the same input to compute the normalization result and rstd.
    3. Result comparison: Compare the outputs of LayerNorm and Normalize to verify their mathematical equivalence.

  - Tiling Implementation

    The tiling implementation flow of the example is as follows:
    1. Obtain and set the minimum temporary space size required for LayerNorm and Normalize interfaces to complete computation.
    2. Obtain the tiling parameters required for both API kernel-side interfaces based on input shape, remaining available computation space size, and other information.

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