# Matmul Multi-Core Unaligned Tiling Direct Call Example

## Overview
Multi-core unaligned tiling refers to a Matmul example where the actual computation of tail blocks in multi-core tiling is smaller than the corresponding parameters in tiling. In this scenario, one of the M, N, K dimensions cannot divide singleCoreM, singleCoreN, or singleCoreK evenly. Without changing the original tiling, you need to call the SetTail interface on the Kernel side to reset singleCoreM/singleCoreN/singleCoreK for the current Matmul computation.

## Supported Products
- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure
```
├── matmul_multi_core_unaligned
│   └── scripts
│       ├── gen_data.py         // Input data and ground truth data generation script
│       └── verify_result.py    // Ground truth comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_multi_core_unaligned.asc              // Ascend C example implementation & call example
```
## Example Description
- Example Function:  
  The Matmul example calls the Matmul API to perform matrix multiplication and Bias addition on multi-core unaligned A and B matrices.

- Example Specifications:  
  In this example: M = 1000, N = 700, K = 500.
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
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_multi_core_unaligned_custom</td></tr>
  </table>

- Example Implementation: 

  - Kernel Key Steps
    - Calculate tailM, tailN, tailK. When tailM < singleCoreM || tailN < singleCoreN || tailK < singleCoreK, process the tail block by calling the SetTail interface to set the tail block size.
      ```cpp
      auto temp0 = AscendC::Ceil(tiling.M, tiling.singleCoreM);
      auto temp1 = AscendC::Ceil(tiling.N, tiling.singleCoreN);
      auto temp2 = AscendC::Ceil(tiling.Ka, tiling.singleCoreK);

      auto divideKCoreNum = tiling.usedCoreNum / temp2;
      auto mCoreIndex = (blockIdx % divideKCoreNum) % temp0;
      auto nCoreIndex = (blockIdx % divideKCoreNum) / temp0;
      auto subKIndex = blockIdx / divideKCoreNum;

      uint32_t gmUseM = tiling.M - mCoreIndex * tiling.singleCoreM;
      uint32_t tailM = gmUseM < tiling.singleCoreM ? gmUseM : tiling.singleCoreM;
      uint32_t gmUseN = tiling.N - nCoreIndex * tiling.singleCoreN;
      uint32_t tailN = gmUseN < tiling.singleCoreN ? gmUseN : tiling.singleCoreN;
      uint32_t gmUseK = tiling.Ka - subKIndex * tiling.singleCoreK;
      uint32_t tailK = gmUseK < tiling.singleCoreK ? gmUseK : tiling.singleCoreK;

      if (tailM < tiling.singleCoreM || tailN < tiling.singleCoreN || tailK < tiling.singleCoreK) {
          matmulObj.setTail(tailM, tailN, tailK);
      }
      matmulObj.IterateAll(cGlobal);
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
  mkdir -p build && cd build;    # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py    # Generate test input data
  ./demo                        # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin    # Verify output result correctness, confirm algorithm logic is correct
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

  The execution result is as follows, indicating successful precision comparison:
  ```bash
  test pass!
  ```