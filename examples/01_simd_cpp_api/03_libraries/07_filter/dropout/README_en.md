# DropOut Example

## Overview

This example implements dropout functionality based on the DropOut high-level API in neural network training scenarios, providing the ability to filter source operands based on mask to obtain destination operands. DropOut is a commonly used regularization technique that prevents overfitting by randomly dropping some neurons. This example uses DropOut byte mode, supporting selective retention or dropping of input data based on mask.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── dropout
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── dropout.asc             // Ascend C example implementation & call example
```

## Example Description

- Example Function:  
  This example filters the input source operand based on the input mask in DropOut byte mode to obtain the output destination operand.
- Example Specifications:  
  <table>
  <caption>Table 1: Example Input/Output Specifications</caption>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> dropout </td></tr>

  <tr><td rowspan="4" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[4, 256]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">mask</td><td align="center">[4, 32]</td><td align="center">uint8_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[4, 256]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">dropout_custom</td></tr>
  </table>

- Example Implementation:  
    This example implements dropout functionality with fixed shape: input src[4, 256], mask[4, 32], output dst[4, 256].

  - Kernel Implementation:  
    The computation logic is: Input data needs to be moved to on-chip storage first, then use the DropOut high-level API interface to complete dropout computation, and finally move the result out.

  - Tiling Implementation:  
    The tiling implementation flow of this example is as follows: Use the GetDropOutMaxMinTmpSize interface to calculate the required maximum/minimum temporary space size, use the minimum temporary space, then determine the required tiling parameters based on input length.

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
    
- Example Execution (NPU Mode)
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201;make -j;             # Build project
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
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The following output indicates that the accuracy comparison passed.
  ```bash
  test pass!
  ```