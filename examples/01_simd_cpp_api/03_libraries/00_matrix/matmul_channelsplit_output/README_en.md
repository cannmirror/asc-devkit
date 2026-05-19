# Matmul Output Channel Split Direct Invocation Sample

## Overview

This sample demonstrates Matmul with output channel split functionality. When the Matmul computation result matrix C has NZ format, matrix C uses fractal storage. If matrix C has float data type, the default fractal size is 16 * 16. The channel split functionality can split each 16 * 16 fractal of matrix C in this scenario into 16 * 8 fractals, enabling matrix C to be stored in 16 * 8 fractals.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_channelsplit_output
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_channelsplit_output.asc   // Ascend C sample implementation & invocation sample
```

## Sample Description

- **Sample Function:**
  When the Matmul sample calls the Matmul API for computation, it enables the matrix multiplication output channel split functionality by setting the isEnableChannelSplit parameter in MatmulConfig to true, performing matrix multiplication and adding bias offset on input matrices A and B.

- **Sample Specifications:**
  In this sample: M = 64, N = 7680, K = 64.

  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_channelsplit_output_custom</td></tr>
  </table>

- **Sample Implementation:**
  - Kernel Key Steps
    - When creating a Matmul object, customize the MatmulConfig parameter by setting the isEnableChannelSplit parameter to true to enable the matrix multiplication output channel split functionality, obtaining a customized Matmul object using the Norm template.
      ```cpp
      __aicore__ inline constexpr MatmulConfig GetCustomNORMCFG()
      {
          auto mmCfg = CFG_NORM;
          mmCfg.isEnableChannelSplit = true;
          return mmCfg;
      }
      constexpr static MatmulConfig CUSTOM_CFG_NORM = GetCustomNORMCFG();
      AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
                      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
                      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::NZ, CType>,
                      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CUSTOM_CFG_NORM>
          matmulObj;
      ```

  - Invocation Implementation
    Use the kernel invocation operator <<<>>> to call the kernel function.

## Compilation and Execution

Execute the following steps in the sample root directory to compile and run the sample.

- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in your current environment.

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

- Sample Execution

  ```bash
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled binary to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```