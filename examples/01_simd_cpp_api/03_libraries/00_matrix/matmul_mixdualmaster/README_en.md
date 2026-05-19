# Matmul Dual-Master Mode Direct Call Sample

## Overview

This sample enables MixDualMaster mode for Matmul, where AIC and AIV run code independently without relying on message-driven mechanisms to improve performance.

In the default MIX mode of the Matmul API, the Matmul API drives AIC execution through a message mechanism by AIV. In dual-master mode, AIC and AIV run code independently without relying on message-driven mechanisms, resulting in better performance.

You can enable dual-master mode when one of the following conditions is met:
- The kernel type is MIX, and the ratio of AIC cores to AIV cores is 1:1
- The kernel type is MIX, and the ratio of AIC cores to AIV cores is 1:2, and both matrix A and matrix B have the IBSHARE parameter enabled

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_mixdualmaster
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_mixdualmaster.asc              // Ascend C sample implementation & call sample
```

## Sample Description

- Sample Function:
  This sample has a kernel type of MIX, and the ratio of AIC cores to AIV cores is 1:2. Both matrix A and matrix B have the IBSHARE parameter enabled. The input AB matrix data for two AIVs corresponding to the same AIC is identical in L1 Buffer.
  Enable dual-master mode by setting the MatmulConfig parameter enableMixDualMaster.

- Sample Specifications:
  In this sample: M = 12288, N = 256, K = 128.
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float32</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float32</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_mixdualmaster_custom</td></tr>
  </table>

- Sample Implementation:
  - Kernel Key Steps
    - Create a Matmul object: Call the GetNormalConfig interface to set the enableMixDualMaster parameter to true, obtain the custom template MM_CFG, and create a Matmul object by passing in the template parameters.
      ```cpp
      static constexpr auto MM_CFG = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, 
      true, IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, true, true, false); 
      // Set the second-to-last parameter enableMixDualMaster to true to create the Matmul object. Matrix A and Matrix B have IBSHARE parameter enabled
      AscendC::Matmul<
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, false, LayoutMode::NONE, true>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false, LayoutMode::NONE, true>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
          AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, MM_CFG> matmulObj;
      ```

  - Call Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
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
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The execution result is as follows, indicating successful precision comparison:
  ```bash
  test pass!
  ```