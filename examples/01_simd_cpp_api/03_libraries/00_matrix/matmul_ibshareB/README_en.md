# Matmul IBShareB Direct Call Example

## Overview

This example enables the IBShare feature to reuse the same A matrix or B matrix data in L1 Buffer. This example demonstrates the scenario where only the B matrix is reused. Enabling this feature can reduce data transfer overhead. Multiple scenarios are supported and can be selected via environment variables.
    <table>
	 	<tr>
	 		<td>scenarioNum</td>
	 		<td>Scenario Type</td>
	 	</tr>
	 	<tr>
	 		<td>1</td>
	 		<td>Default implementation, enable IBShare template</td>
	 	</tr>
	 	<tr>
	 		<td>2</td>
	 		<td>Enable pure Cube mode + IBShareB</td>
	 	</tr>
	 </table>

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 training series products/Atlas A3 inference series products
- Atlas A2 training series products/Atlas A2 inference series products

## Directory Structure

```
├── matmul_ibshareB
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden data comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_ibshareB.asc     // Ascend C example implementation & call example
```

## Example Description

- Example Function:
  When calling the Matmul high-level API, the IBShare feature for the B matrix is enabled. During computation, the B matrix data in L1 Buffer used by two AIVs corresponding to the same AIC in each iteration are consistent.

- Constraints:
  - For scenarios where only the A matrix or B matrix has IBShare enabled, the reused matrix must be fully loaded in L1 Buffer.

- Example Specifications:
  In this example: M = 64, N = 256, K = 384.
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="5" align="center">MatmulIBShareBCustom</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_ibshareb_custom</td></tr>
  </table>

- Example Implementation:
  - Kernel Key Steps
    - Create a Matmul object and set the IBShare parameter for the B matrix to true.
      - Method 1: Default implementation, create a Matmul object using the default IBShare template CFG_IBSHARE_NORM.
        ```cpp
        #include "lib/matmul_intf.h"
    
        using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
        using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false, LayoutMode::NONE, true>;
        using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
        using BIAS_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
        AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_IBSHARE_NORM> matmulObj;
        ```
      - Method 2: Enable pure Cube mode implementation. In the code defining the Matmul object, set the ASCENDC_CUBE_ONLY macro, which must be set before #include "lib/matmul_intf.h".
        ```cpp
        #define ASCENDC_CUBE_ONLY // Set ASCENDC_CUBE_ONLY macro
        #include "lib/matmul_intf.h"

        using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>;
        using B_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType, false, LayoutMode::NONE, true>;
        using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
        using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
        AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_NORM> matmulObj;
        ```

  - Kernel Function
    When using pure Cube mode, specify __cube__ at the entry point of the kernel function implementation.

  - Call Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.
    ```cpp
    matmul_ibshareb_custom<<<tilingData.usedCoreNum / MIX_RATIO, nullptr, stream>>>(x1, x2, bias, y, workspaceDevice,
                          tilingDevice);       // Method 1: Non-pure Cube mode, SetDim is set to the number of AIV:AIC combined cores
    matmul_ibshareb_custom<<<tilingData.usedCoreNum, nullptr, stream>>>(x1, x2, bias, y, workspaceDevice,
                          tilingDevice);       // Method 2: Pure Cube mode, SetDim is set to the number of AIC cores                          
    ```

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
  SCENARIO=1                                                                    # Default implementation. Enable IBShare template
  mkdir -p build && cd build;                                                   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO ..;make -j; # Build project, default npu mode
  python3 ../scripts/gen_data.py                                                # Generate test input data
  ./demo                                                                        # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin       # Verify output correctness and confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  SCENARIO=1
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  | ----------------| -----------------------------| ---------------------------------------------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201`, `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 training series products/Atlas A2 inference series products and Atlas A3 training series products/Atlas A3 inference series products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2` | Scenario number: 1=default implementation, enable IBShare template, 2=enable pure Cube mode + IBShareB |

- Execution Result

  The following execution result indicates successful precision comparison:
  ```bash
  test pass!
  ```