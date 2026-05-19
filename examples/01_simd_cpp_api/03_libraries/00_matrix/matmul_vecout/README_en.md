# Matmul Example with VECOUT Input Matrix Direct Call Example

## Overview
A Matmul example using user-defined VECOUT input, allowing developers to manage Unified Buffer autonomously for efficient hardware resource utilization.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── matmul_vecout
│   └── scripts
│       ├── gen_data.py         // Input data and ground truth data generation script
│       └── verify_result.py    // Ground truth comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_vecout.asc       // Ascend C example implementation & call example
```

## Example Description
- Example Function:  
  The Matmul example calls the Matmul API to perform matrix multiplication and bias addition on input A and B matrices. The input position of matrix A is VECOUT.

- Example Specifications:  
  In this example: M = 31, N = 31, K = 31.
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">matmul_vecout_custom</td></tr>
  </table>

- Example Implementation: 
  - Kernel Key Steps  
    - Create a Matmul object. In the MatmulType of left matrix A, POSITION is VECOUT.
      ```cpp
      AscendC::MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, AType>
      ```
    - Customize the data copy of left matrix A from GM to VECOUT, and set left matrix A as VECOUT input.
      ```cpp
      AscendC::LocalTensor<AType> vecinTensor = vecin.AllocTensor<AType>();
      // A matrix copy-in parameters
      DataCopyPad(vecinTensor, aGlobal, {blockCount, blockLen, srcStride, dstStride, 0}, {false, 0, 0, 0});
      vecin.EnQue(vecinTensor);
      AscendC::LocalTensor<AType> vecinLocal = vecin.DeQue<AType>();

      AscendC::LocalTensor<AType> vecoutTensor = vecout.AllocTensor<AType>();
      DataCopy(vecoutTensor, vecinLocal, singleSize); // Direct copy of entire size for convenience
      vecout.EnQue(vecoutTensor);
      AscendC::LocalTensor<AType> vecoutLocal = vecout.DeQue<AType>();
      vecin.FreeTensor(vecinLocal);

      matmulObj.SetTensorA(vecoutLocal, isTransA);
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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Parameter | Description | Options | Default |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-3510 | dav-3510 |

- Execution Result

  The execution result is as follows, indicating successful precision comparison.

  ```bash
  test pass!
  ```