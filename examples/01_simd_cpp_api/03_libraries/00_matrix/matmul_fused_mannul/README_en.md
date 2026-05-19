# Fused Programming matmul_fused_mannul Example

## Overview

This example demonstrates AIC and AIV split-core fused programming implementation, mainly introducing the pure Cube mode of the Matmul high-level API, which requires calling related interfaces to manually control inter-core synchronization between AIC and AIV.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products
- Atlas inference series products AI Core

## Directory Structure

```
├── matmul_fused_mannul
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden data comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_fused_mannul.asc // Ascend C example implementation & call example
```

## Example Description

- Example Function:
  This example creates a Matmul object in pure Cube mode on the AIC core, implements Matmul computation, and outputs intermediate results to GM. After the AIC core computation is complete, inter-core synchronization is manually controlled by calling CrossCoreSetFlag and CrossCoreWaitFlag interfaces, then LeakyRelu computation is performed on the intermediate results on GM in the AIV core.

- Example Specifications:
  In this example: M = 128, N = 128, K = 256
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_fused_mannul_custom</td></tr>
  </table>

- Example Implementation:

  - Kernel Key Steps
    - AIC side specific steps:
      - Create and initialize Matmul object.
        Configure the ASCENDC_CUBE_ONLY macro before #include "lib/matmul_intf.h" to create a pure Cube mode Matmul object
          ```cpp
          #define ASCENDC_CUBE_ONLY
          #include "lib/matmul_intf.h"
          ```
      - Complete Matmul computation.
      - Set inter-core synchronization.
          ```cpp
          if ASCEND_IS_AIC {
            AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(3);
          }
          ```
    - AIV side specific steps:
      - Create and initialize LeakyRelu object.
      - Wait for inter-core synchronization.
          ```cpp
          if ASCEND_IS_AIV {
            AscendC::CrossCoreWaitFlag(3);
          }
          ```
      - Complete LeakyRelu computation.

  - Tiling Key Steps
    - Set custom MatmulConfig parameters to synchronize parameters configured on the Kernel side (such as scheduleType) to the Tiling side.
      ```cpp
      matmul_tiling::MatmulConfigParams matmulConfigParams(1, false, matmul_tiling::ScheduleType::OUTER_PRODUCT,
          matmul_tiling::MatrixTraverse::FIRSTM, false);
      cubeTiling.SetMatmulConfigParams(matmulConfigParams);
      ```

  - Call Implementation
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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness and confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

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
  The following execution result indicates successful precision comparison.
  ```bash
  test pass!
  ```