# Matmul SplitM Strategy Direct Invocation Sample

## Overview

This sample demonstrates Matmul in a multi-core M-split scenario, where the input matrix is split along the M axis and distributed across multiple cores for parallel processing. This is applicable in AscendC separation mode with AIC:AIV=1:N architecture, where after one Iterate computation on the AIC side, the intermediate results are split along the M axis and processed separately by N AIVs.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── matmul_splitm
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden data verification file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read and write functions
│   └── matmul_splitm.asc       // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  This sample implementation operates in AscendC separation mode with AIC:AIV=1:2 architecture. The sample process calls the Matmul high-level API with both matrix A and matrix B having the IBShare parameter enabled. The Iterate intermediate results are output to Unified Buffer on the AIC, and then each of the two AIVs processes half of the intermediate result data in the Unified Buffer.

- Sample Specifications:
  In this sample: M = 127, N = 127, K = 63.
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">matmul_splitm_custom</td></tr>
  </table>

- Sample Implementation:
 
  - Key Kernel Steps
    - Use the Matmul API with the following key configurations: 1) IBSHARE of both matrix A and matrix B MatmulType are set to true, 2) Use the NORM template, 3) Use the SplitM template strategy.
    - Use a for loop with Iterate and GetTensorC interfaces to retrieve the data from each Iterate computation result, split along the M direction of the matrix, to the current AIV core.
    - After the current AIV core retrieves the data for processing, output the results to the corresponding GM based on whether GetSubBlockIdx is 0 or 1.

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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness, confirm algorithm logic
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-3510 | dav-3510 |

- Execution Result

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```