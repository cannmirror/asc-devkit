# Matmul IBShareAB Feature Example

## Overview

This example calls the Matmul high-level API to enable the IBShare feature, which reuses the same A matrix or B matrix data in L1 Buffer. This example demonstrates the scenario where both A matrix and B matrix are reused simultaneously. Enabling this feature can reduce data搬运 overhead. Two scenarios are supported and can be selected via environment variables.
    <table>
 	<tr>
 		<td>scenarioNum</td>
 		<td>Scenario Type</td>
 	</tr>
 	<tr>
 		<td>1</td>
 		<td>Enable AB matrix IBShare (A and B matrices are not split)</td>
 	</tr>
 	<tr>
 		<td>2</td>
 		<td>Disable AB matrix IBShare (A and B matrices are split along K-axis)</td>
 	</tr>
 </table>

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products
- Atlas inference series products AI Core

## Directory Structure

```
├── matmul_ibshareAB
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden data comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_ibshareAB.asc    // Ascend C example implementation & call example
```

## Example Description

- Example Function:
  When calling the Matmul high-level API, the IBShare feature for A and B matrices is enabled. During computation, the A and B matrix data in L1 Buffer used by two AIVs corresponding to the same AIC in each iteration are consistent.

  When both A matrix and B matrix have IBShare enabled, the K column is not split for computation; when neither has IBSHARE enabled, computation is performed by splitting the K column. By comparing the execution time of the two scenarios, the performance improvement of this feature can be observed.
  Data processing diagram for enabled AB matrix ibshare scenario (A matrix and B matrix are not split): ![alt text](./pictures/matmul_ABshare.png)
  Data processing diagram for disabled AB matrix ibshareAB scenario (A matrix and B matrix are split): ![alt text](./pictures/matmul_noABshare.png)


- Example Specifications:
  In this example: M = 128, N = 256, K = 384.
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">matmul_ABshare_custom</td></tr>
  </table>

- Example Implementation:

  - Kernel Key Steps
    - Create Matmul object:
        Configure the A and B matrix IBSHARE parameters to true or false based on the SCENARIO_NUM build option in CMakeLists.
        ```cpp
        #if SCENARIO_NUM == 1
        constexpr bool isABshare = true;
        #else
        constexpr bool isABshare = false;
        #endif

        Matmul<
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType, false, LayoutMode::NONE, isABshare>,
            MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false, LayoutMode::NONE, isABshare>,
            MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, CType>> matmulObj;
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
  SCENARIO=1                                                                    # Set scenario number
  mkdir -p build && cd build;                                                   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO ..;make -j; # Build project, default npu mode
  python3 ../scripts/gen_data.py                                                # Generate test input data
  ./demo                                                                        # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin       # Verify output correctness and confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=1 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=1 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes or scenarios, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201`, `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 training series products/Atlas A2 inference series products and Atlas A3 training series products/Atlas A3 inference series products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2` | Scenario number: 1=enable AB matrix IBShare, 2=disable AB matrix IBShare |

- Execution Result

  The following execution result indicates successful precision comparison:
  ```bash
  test pass!
  ```