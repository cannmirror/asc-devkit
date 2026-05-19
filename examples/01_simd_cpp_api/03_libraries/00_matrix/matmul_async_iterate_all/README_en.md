# Matmul IterateAll Asynchronous Direct Invocation Sample

## Overview

A Matmul sample in asynchronous scenario, implemented by calling IterateAll to output to GM.

Asynchronous scenario means that during program execution, the next operation can be performed without waiting for a specific operation to complete. Asynchronous scenarios can reduce synchronization wait times and improve parallelism. Developers can choose this approach when they have high requirements for computation performance.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_async_iterate_all
│   └── scripts
│       ├── gen_data.py         // Input data and ground truth data generation script
│       └── verify_result.py    // Ground truth comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_async_iterate_all.asc              // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  The Matmul sample implements asynchronous matrix multiplication by calling IterateAll to output to GM.

- Sample Specifications:

  In this sample: M = 640, N = 1024, K = 512.
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td>
  <td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_async_iterate_all_custom</td></tr>
  </table>

- Sample Implementation:
  - Kernel Key Steps
    - Create a Matmul object with the output C matrix's TPosition set to GM.
      ```cpp
      AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL> matmulObj;
      ```
    - Get the matrix multiplication result.
      ```cpp
      matmulObj.template IterateAll<false>(cGlobal, 0, false, true);
      matmulObj.WaitIterateAll();
      ```

  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
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

- Run the Sample

  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness and confirm algorithm logic
  ```

  For CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then run cmake again.

- Build Options Description

  | Parameter | Description | Options | Default |
  |-----------|-------------|---------|---------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The following output indicates successful accuracy verification:
  ```bash
  test pass!
  ```