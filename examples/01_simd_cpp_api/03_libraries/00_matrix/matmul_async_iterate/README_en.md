# Matmul Iterate Asynchronous Scenario Direct Invocation Sample

## Overview

This sample demonstrates Matmul in an asynchronous scenario, implemented by calling Iterate and GetTensorC to output to VECIN.

An asynchronous scenario refers to program execution where the next operation can proceed without waiting for the previous operation to complete. Asynchronous scenarios can reduce synchronization waits and improve parallelism. Developers with high computational performance requirements can choose this approach.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_async_iterate
│   └── scripts
│       ├── gen_data.py         // Input data and golden data generation script
│       └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_async_iterate.asc              // Ascend C sample implementation & invocation sample
```

## Sample Description

- **Sample Function:**
  The Matmul sample implements matrix multiplication in an asynchronous scenario by calling Iterate and GetTensorC to output to VECIN.

- **Sample Specifications:**
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
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_async_iterate_custom</td></tr>
  </table>

- **Sample Implementation:**
  - Kernel Key Steps
    - Create a Matmul object with the TPosition of output matrix C set to VECIN.
      ```cpp
      AscendC::Matmul<AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
      AscendC::MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, CType>,
      AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>, CFG_MDL> matmulObj;
      ```
    - Initialize operations.
    - Set the left matrix A, right matrix B, and Bias.
    - Obtain the matrix multiplication result.
      ```cpp
      matmulObj.SetWorkspace(workspaceGlobal);
      matmulObj.template Iterate<false>();
      uint32_t baseM = this->tiling.baseM;
      uint32_t baseN = this->tiling.baseN;
      pipe->InitBuffer(cInQueue, 1, baseM * baseN * sizeof(CType));
      pipe->InitBuffer(cOutQueue, 1, baseM * baseN * sizeof(CType));
      AscendC::DataCopyParams copyParams = {
          (uint16_t)baseM,
          (uint16_t)(baseN * sizeof(CType) / AscendC::DEFAULT_C0_SIZE),
          (uint16_t)0,
          (uint16_t)((this->tiling.N - baseN) * sizeof(CType) / AscendC::DEFAULT_C0_SIZE)
      };
      uint32_t iterateTimes = Ceiling(this->tiling.singleCoreM, baseM) * Ceiling(this->tiling.singleCoreN, baseN);
      for (uint32_t i = 0; i < iterateTimes; ++i) {
          // compute
          auto cInLocal = cInQueue.AllocTensor<CType>();
          matmulObj.template GetTensorC<false>(cInLocal);
          cInQueue.EnQue(cInLocal);
          // any vector operator
          auto src = cInQueue.DeQue<CType>();
          auto dst = cOutQueue.AllocTensor<CType>();
          DataCopy(dst, src, baseM * baseN);
          cOutQueue.EnQue(dst);
          cInQueue.FreeTensor(src);
          // copy out
          auto cOutLocal = cOutQueue.DeQue<CType>();
          DataCopy(cGlobal[CalcDstOffset(i)], cOutLocal, copyParams);
          cOutQueue.FreeTensor(cOutLocal);
      }
      ```

  - Tiling Key Steps
    - Set the TPosition of C to VECIN.
      ```cpp
      cubeTiling->SetCType(matmul_tiling::TPosition::VECIN, matmul_tiling::CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
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
  mkdir -p build && cd build;   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;             # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                        # Execute the compiled binary to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output correctness
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