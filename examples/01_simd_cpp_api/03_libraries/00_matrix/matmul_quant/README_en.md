# Matmul Dequantization Scenario Direct Call Example

## Overview

This is a Matmul example with output dequantization, supporting both scalar dequantization mode and vector dequantization mode.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure
```
├── matmul_quant
│   └── scripts
│       ├── gen_data.py         // Script for generating input data and golden data
│       └── verify_result.py    // Golden data verification file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_quant.asc        // Ascend C example implementation & invocation example
```

## Example Description
- Example Functionality:
  This Matmul example calls the Matmul API with int8_t type input, and outputs the computation result as half type with dequantization. It supports both scalar dequantization mode and vector dequantization mode. In this scenario, when moving matrix C data from CO1 to Global Memory, dequantization is performed using either a scalar or vector for all values of the output matrix.

- Example Specifications:
  In this example: M = 1024, N = 1024, K = 1024.
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">int8_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">int8_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">int32_t</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_quant_custom</td></tr>
  </table>

- Example Implementation:
  - Kernel Key Steps
    - Set dequantization parameters.
      When the compilation option QUANT_MODE value is 1, set the compilation macro CUSTOM_QUANT_VECTOR to enable vector dequantization mode.
      Based on whether the macro CUSTOM_QUANT_VECTOR is defined, set the corresponding dequantization parameters.
      ```cpp
      #if defined(CUSTOM_QUANT_VECTOR)
          matmulObj.SetQuantVector(quantGlobal);
      #else
          float quantFloat = 0.1f;
          uint64_t quantValue = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&quantFloat));
          matmulObj.SetQuantScalar(quantValue);
      #endif
      ```

  - Tiling Key Steps
    - Set Matmul dequantization mode.
      ``` cpp
      #if defined(CUSTOM_QUANT_VECTOR)
          tilingApi.SetDequantType(matmul_tiling::DequantType::TENSOR); // set TENSOR quant mode
      #else
          tilingApi.SetDequantType(matmul_tiling::DequantType::SCALAR); // set SCALAR quant mode
      #endif
      ```

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run
Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Please select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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

- Example Execution

  ```bash
  # -DQUANT_MODE=0: Enable scalar dequantization mode;
  # -DQUANT_MODE=1: Enable vector dequantization mode;
  # -m=0: Enable scalar dequantization mode;
  # -m=1: Enable vector dequantization mode;
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DQUANT_MODE=0 -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, using scalar dequantization mode as example, default npu mode
  python3 ../scripts/gen_data.py -m=0   # Generate test input data, using scalar dequantization mode as example
  ./demo                        # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DQUANT_MODE=0 -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DQUANT_MODE=0 -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Parameter | Description | Available Values | Default Value |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```