# DeepNorm Example

## Overview

This example demonstrates how to call the DeepNorm high-level API provided by Ascend C to implement DeepNorm normalization in deep neural network training scenarios. This API can improve the training stability of deep Transformer networks by scaling the residual connection (α coefficient) when performing LayerNorm normalization at the layer level.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── deepnorm
│   ├── scripts
│   │   ├── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── deepnorm.asc            // Ascend C example implementation & call example
```

## Example Description

- Example Function:  
  This example implements DeepNorm normalization for input data with shape [B, S, H]. The computation formula is:
  $$
  DeepNorm(x) = LayerNorm(α * X + SubLayer(X))
  $$

- Example Specifications:  
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> deepnorm </td></tr>

  <tr><td rowspan="6" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">inputX</td><td align="center">[4, 16, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">inputGx</td><td align="center">[4, 16, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">beta</td><td align="center">[1, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">gamma</td><td align="center">[1, 64]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="4" align="center">Example Output</td></tr>
  <tr><td align="center">output</td><td align="center">[4, 16, 64]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outputMean</td><td align="center">[4, 16]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">outputVariance</td><td align="center">[4, 16]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">deepnorm_custom</td></tr>
  </table>

- Example Implementation:  

  - Kernel Implementation

    The computation logic is:  
    Use the DeepNorm high-level API interface to complete the deepnorm computation and obtain the final result, then move it to external storage. For details about the API used, refer to DeepNorm.
    
  - Tiling Implementation

    The tiling implementation flow of the example is as follows:
    1. Call AscendC::GetDeepNormMaxMinTmpSize to get the minimum temporary space size required for DeepNorm interface computation.
    2. Call AscendC::GetDeepNormTilingInfo to get the Tiling parameters required for the kernel-side interface based on the input shape and workspace size.

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