# Matmul UnitFlag Feature Direct Invocation Sample

## Overview

This sample demonstrates Matmul with the UnitFlag feature enabled, which parallelizes the CUBE computation pipeline with the FIXPIPE data output pipeline. After enabling the UnitFlag feature, fine-grained synchronization between MMAD and FIXPIPE instructions is implemented internally in the Matmul API, enabling parallel execution of the computation pipeline and data output pipeline, thus improving sample performance.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_unitflag
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden data verification file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── matmul_unitflag.asc              // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  When the Matmul sample calls the Matmul API for computation, the UnitFlag feature for the MDL template is enabled by setting the enUnitFlag parameter in MatmulConfig to true. The Norm template and IBShare template have UnitFlag enabled by default, while the MDL template has UnitFlag disabled by default.

- Constraints
  - The UnitFlag feature only supports Norm, IBShare, and MDL templates.
  - When UnitFlag is enabled, the sample cannot have both L0C-to-Global Memory and L1-to-Global Memory output pipelines simultaneously.
  - When UnitFlag is enabled together with L0C accumulation feature, multiple Iterate computations with one GetTensorC output are not supported.

- Sample Specifications:
  In this sample: M = 1024, N = 4096, K = 1024.
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
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_unitflag_custom</td></tr>
  </table>

- Sample Implementation:
  - Key Kernel Steps
    - When creating the Matmul object, customize the MatmulConfig parameter by setting the enUnitFlag parameter to true to enable the UnitFlag feature, obtaining a customized Matmul object using the MDL template.
      ```cpp
      __aicore__ inline constexpr MatmulConfig GetUnitFlagCfg()
      {
          auto mmCfg = CFG_MDL;
      #ifdef ENABLE_UNITFLAG_FEATURE
          // enable UnitFlag feature
          mmCfg.enUnitFlag = true;
      #endif
          return mmCfg;
      }
      constexpr static MatmulConfig CFG_MDL_UNITFLAG = GetUnitFlagCfg();

      using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
      using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>;
      using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
      using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
      AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL_UNITFLAG> matmulObj;
      ```

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Compilation and Execution

Execute the following steps in the root directory of this sample to compile and run the sample.

- Configure Environment Variables
  Select the corresponding environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
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
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output correctness, confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
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

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```