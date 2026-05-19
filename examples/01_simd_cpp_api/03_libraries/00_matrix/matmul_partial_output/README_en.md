# Matmul Direct Call Example with Partial Output Enabled

## Overview

This is a Matmul high-level API example that demonstrates enabling the Partial Output feature. The Partial Output feature applies to scenarios where matrix multiplication results do not need to be accumulated, and only the baseM*baseN computation result from baseM*baseK and baseK*baseN needs to be output.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure
```
├── matmul_partial_output
│   └── scripts
│       ├── gen_data.py         // Script for generating input data and golden data
│       └── verify_result.py    // Golden data verification file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_partial_output.asc              // Ascend C example implementation & invocation example
```

## Example Description
- Example Functionality:
  This example calls the Matmul high-level API with Partial Output enabled to perform matrix multiplication on input matrices A and B. Non-accumulated results are moved to VECIN (Unified Buffer), where users can perform custom operations on the data, such as dequantization. Finally, the basic blocks are accumulated to obtain the final result.

  - Constraints
    - Only supports the MDL template.
    - Only supports the continuous write mode using Iterate and GetTensorC interfaces. The IterateAll interface and non-continuous write mode are not supported.
    - Matmul computation with a Bias matrix is not supported, meaning Bias matrix input is not supported.

- Example Specifications:
  In this example: M = 128, N = 128, K = 256.
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_partial_output_custom</td></tr>
  </table>
- Example Implementation:

  - Kernel Key Steps
    - Specific Steps:
      - Create a Matmul object.
        When creating a Matmul object, customize the MatmulConfig parameter by setting the isPartialOutput parameter to true to enable the Partial Output feature, obtaining a customized Matmul object using the MDL template.
          ```cpp
          __aicore__ inline constexpr MatmulConfig GetCustomMDLCFG()
          {
              auto mmCfg = CFG_MDL;
              mmCfg.isPartialOutput = true;
              return mmCfg;
          }
          constexpr static MatmulConfig CUSTOM_CFG_MDL = GetCustomMDLCFG();
          AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
                    AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CUSTOM_CFG_MDL>
              matmulObj;
          ```
      - Initialization operations.
      - Set left matrix A and right matrix B.
      - Complete matrix multiplication operations.
        Call the Iterate and GetTensorC interfaces in continuous write mode.
          ```cpp
          while (matmulObj.Iterate()) {
              matmulObj.GetTensorC(workspace[offset], 0, true);
              offset += tiling.baseM * tiling.baseN;
          }
          ```
      - End matrix multiplication operations.

  - Tiling Key Steps
    - Ascend C provides a set of Matmul Tiling APIs to help users obtain the Tiling parameters required for Matmul kernel computation. Simply pass in the A/B/C matrix information and call the API interfaces to obtain the relevant parameters in the TCubeTiling structure.
    - The flow for obtaining Tiling parameters is as follows:
      - Create a Tiling object.
      - Set parameter type information for A, B, C, Bias, and shape information such as M, N, Ka, Kb.
      - Call the GetTiling interface to obtain Tiling information.

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
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Parameter | Description | Available Values | Default Value |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following execution result indicates successful precision comparison:
  ```bash
  test pass!
  ```