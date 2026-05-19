# Matmul Template Parameter MatmulCallbackFunc Direct Invocation Sample

## Overview

This sample demonstrates custom usage of the Matmul API template parameter MatmulCallbackFunc. MatmulCallbackFunc is used to configure custom functions for copying the left matrix from Global Memory to A1 (L1 Buffer), copying the right matrix from Global Memory to B1 (L1 Buffer), and copying computation results from CO1 (L0C Buffer) to Global Memory. This sample uses the callback function for custom transfer from Global Memory to A1 (L1 Buffer) as an example to demonstrate how to use this template parameter.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_callback
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_callback.asc     // Ascend C sample implementation & invocation sample
```

## Sample Description

- **Sample Function:**
  The Matmul sample performs matrix multiplication and adds bias offset on input matrices A and B. The custom left matrix transfer function CustomDataCopyInA is passed as a parameter to the Matmul template parameter MatmulCallbackFunc to implement custom transfer of the left matrix from Global Memory to A1 (L1 Buffer). This sample demonstrates the callback functionality using input matrix A as an example. The callback functionality for input matrix B and output matrix C can be implemented similarly.

- **Sample Specifications:**
  In this sample: M = 2560, N = 128, K = 512.

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
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_callback_custom</td></tr>
  </table>

- **Sample Implementation:**
  - Computation Logic: In this sample, matrix A uses non-contiguous transfer, where every two base blocks require one address jump (i.e., the first and second blocks are contiguous, there is an address offset between the second and third blocks, the third and fourth blocks are contiguous, and so on). Before writing the custom callback function, you need to determine the sizes of SingleM, SingleK, baseM, and baseK after tiling, as well as the distribution of base blocks. In this sample, the single-core computation size SingleShape is set on the Tiling side: SingleM=128, SingleK=512, SingleN=128. Then during the debugging phase, the GetBaseM and GetBaseK interfaces are called to print the parameter information: baseM=128, baseK=128. This indicates that each single core has 4 base blocks, which is used for address offset calculation on the Kernel side to write the callback function transfer. The variable offsetListGlobal stores the starting addresses of the 0th and 2nd base blocks of matrix A on each single core, and each single core needs to pass in 2 addresses.
  - Kernel Key Steps
    - Define a custom left matrix transfer function CustomDataCopyInA, which implements the transfer of baseM * baseK base block of the left matrix from Global Memory to logical position A1 (L1 Buffer) through matrix starting address, offset address, etc.
    - Pass the custom CustomDataCopyInA to the template parameter MatmulCallBackFunc to create a Matmul object.
      ```cpp
      AscendC::Matmul<
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>,
        AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>,
        CFG_NORM, AscendC::MatmulCallBackFunc<nullptr, CustomDataCopyInA, nullptr>> matmulObj;
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