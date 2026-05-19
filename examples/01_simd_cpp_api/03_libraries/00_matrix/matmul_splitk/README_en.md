# Matmul Multi-Core Split-K Direct Call Example
## Overview
A Matmul example in the multi-core split-K scenario, where input matrices are split along the K-axis and distributed to multiple cores for parallel processing. This applies to multi-core Matmul scenarios where the M and N dimensions of the input matrices are small and cannot be split across multiple cores in the M and N directions.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products
## Directory Structure
```
├── matmul_splitk
│   └── scripts
│       ├── gen_data.py         // Input data and ground truth data generation script
│       └── verify_result.py    // Ground truth comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_splitk.asc       // Ascend C example implementation & call example
```
## Example Description
- Example Function:  
  This example enables multi-core split-K by calling EnableMultiCoreSplitK when invoking the Matmul Tiling API, obtaining tiling computation parameters for multi-core split-K, and distributing a single Iterate computation across multiple cores. In the Kernel implementation, first clear the output Global Memory to zero, then enable AtomicAdd accumulation. After the computations distributed to multiple cores for the same Iterate complete, the results are accumulated to the output Global Memory.

- Constraints
  - In the multi-core split-K scenario, when obtaining the C matrix result, only output to Global Memory is supported.
  - In the multi-core split-K scenario, before writing the C matrix slice result to Global Memory for the first time in the Kernel-side code, you must first clear the Global Memory to zero, and then enable AtomicAdd accumulation when obtaining the C matrix slice result.  
    If you do not clear the Global Memory beforehand, precision issues may occur due to accumulating invalid original data in Global Memory.
  - In the multi-core split-K scenario, enabling Bias is not supported.

- Example Specifications:  
  In this example: M = 16, N = 16, K = 1024.
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_splitk_custom</td></tr>
  </table>
- Example Implementation: 
  - Kernel Key Steps  
    - Clear the output Global Memory address of the C matrix to zero.
      ```cpp
      Fill(cGlobal, tiling.M * tiling.N, (cType)0);
      ```
    - Enable AtomicAdd accumulation to complete the matrix multiplication operation.
      ```cpp
      uint8_t enAtomic = 1; // set AtomicAdd
      matmulObj.IterateAll(cGlobal, enAtomic);
      ```

  - Tiling Key Steps
    - Set the parameter type information for A, B, C, and Bias; set M, N, Ka, Kb shape information and so on, then call EnableMultiCoreSplitK to enable multi-core split-K.
      ```cpp
      cubeTiling->EnableMultiCoreSplitK(true);
      ```

  - Invocation Implementation  
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run
Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables  
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
  - Default path, CANN software package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```
    
  - Specified path install_path, CANN software package installed
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution Result

  The execution result is as follows, indicating successful precision comparison.

  ```bash
  test pass!
  ```