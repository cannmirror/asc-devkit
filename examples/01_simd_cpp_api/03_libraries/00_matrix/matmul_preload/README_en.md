# M-direction Preload Matmul Example

## Overview

This is a Matmul example with M/N-direction preloading, which can reduce MTE2 gaps. When MTE2 pipeline gaps are large and M/N values are significant, you can enable the corresponding M/N-direction preload function to load matrix A/B data in advance.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure
```
├── matmul_preload
│   └── scripts
│       ├── gen_data.py         // Script for generating input data and golden data
│       └── verify_result.py    // Golden data verification file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_preload.asc      // Ascend C example implementation & invocation example
```

## Example Description

- Example Functionality:
  This example implements two scenarios: Preload M-direction pipeline parallel and Preload N-direction pipeline parallel, controlled by the MatmulConfig doMTE2Preload parameter. When the preloadMode value in the code is 1, M-direction pipeline parallel is enabled. When the preloadMode value is 2, N-direction pipeline parallel is enabled.

- Constraints
  - The preload function is only valid for the MDL template
  - When enabling M/N preload function, ensure K is fully loaded and M/N has Double buffer enabled
  - The condition for K full load is singleK <= baseK * stepK
  - The condition for M Double buffer is depthA1 = stepM * stepK * 2
  - The condition for N Double buffer is depthB1 = stepN * stepK * 2

- Example Specifications:
  In this example: M = 128, N = 24576, K = 512.
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">matmul_preload_custom</td></tr>
  </table>

- Example Implementation
  - Kernel Key Steps
    - Configure MatmulConfig template parameters, enable the doMTE2Preload switch with value 1 (M-direction) or 2 (N-direction), and create a Matmul object.
      ```cpp
      static constexpr MatmulConfig MM_CFG_PRELOAD = GetMDLConfig(false, false, 2); 
      AscendC::Matmul<
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>,
        MM_CFG_PRELOAD> matmulObj;
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

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```