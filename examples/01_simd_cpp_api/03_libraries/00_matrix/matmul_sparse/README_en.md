# Sparse Matmul Direct Call Example

## Overview
A Matmul example for 4:2 sparse matrix multiplication (Sparse Matmul), which reduces memory usage and computation during matrix multiplication. Sparse matrix multiplication skips zero elements in the sparse matrix B and only performs data transfer, storage, and computation on non-zero elements.  
In this scenario, the input original left matrix A is a regular matrix, and the right matrix is a sparse matrix. In the sparse matrix B, at least 2 out of every 4 elements are zeros. Before performing Matmul computation, users need to perform 4:2 densification on matrix B themselves. This means filtering out 2 zero elements from every 4 elements based on the original sparse matrix B, making matrix B a dense matrix. The Sparse Matmul scenario calls the Matmul API to complete matrix multiplication between matrix A and the 4:2 densified matrix B.  
> **Note:** 4:2 sparse matrix multiplication (Sparse Matmul) currently only supports matrix B transposition.

## Supported Products
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure
```
├── matmul_sparse
│   └── scripts
│       ├── gen_data.py         // Input data and ground truth data generation script
│       └── verify_result.py    // Ground truth comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_sparse.asc       // Ascend C example implementation & call example
```
## Example Description
- Example Function:  
  The Matmul example calls the high-level Matmul API to perform matrix multiplication and bias addition on the input left matrix A and the 4:2 densified right matrix B.   
  During the data preparation phase before computation, the ground truth data generation script completes the densification of matrix B and generates the index matrix. When implementing Matmul computation, the high-level Matmul API is called with the 4:2 densified matrix B and index matrix to complete the Sparse Matmul scenario matrix multiplication computation.

- Example Specifications:  
  In this example: M = 128, N = 7680, K = 64.
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[M, K]</td><td align="center">int8_t</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[K, N]</td><td align="center">int8_t</td><td align="center">ND</td><td align="center">true</td></tr>
  <tr><td align="center">bias</td><td align="center">[1, N]</td><td align="center">int32_t</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[M, N]</td><td align="center">int32_t</td><td align="center">ND</td><td align="center">-</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">matmul_sparse_custom</td></tr>
  </table>

- Example Implementation: 
  - Kernel Key Steps
    - When creating a Matmul object, define the parameter type information of matrix B through SparseMatmulType.
      ```cpp
      using A_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, ATYPE, false>;
      // Use SparseMatmulType to define matrix B parameter type information
      using B_TYPE = AscendC::SparseMatmulType<AscendC::TPosition::GM, AscendC::TPosition::GM, CubeFormat::ND, BType, true>;
      using C_TYPE = AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>;
      using BIAS_TYPE =  AscendC::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasType>;
      AscendC::Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, CFG_MDL> matmulObj;
      ```
    - Set the index matrix.
      ```cpp
      matmulObj.SetSparseIndex(gm_index); // Set the index matrix gm_index generated during matrix B densification
      ```

  - Tiling Key Steps
    - Enable the Sparse Matmul sparse matrix computation scenario.
      ```cpp
      auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
      matmul_tiling::MatmulApiTiling tiling(ascendcPlatform);
      tiling.SetSparse(true); // Enable Sparse Matmul sparse matrix computation scenario
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
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201 | dav-2201 |

- Execution Result

  The execution result is as follows, indicating successful precision comparison.

  ```bash
  test pass!
  ```