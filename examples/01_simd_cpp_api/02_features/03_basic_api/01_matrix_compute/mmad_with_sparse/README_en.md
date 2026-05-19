# mmad_with_sparse Example

## Overview

This example introduces the basic API MmadWithSparse calling example.

## Supported Products

- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── mmad_with_sparse
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for output data and golden data
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   └── mmad_with_sparse.asc        // Ascend C example implementation & calling example
```

## Operator Description

- Operator Function:

  This example implements a sparse matrix multiplication operator with fixed [m, n, k] = [128, 128, 64] using the Ascend C basic API MmadWithSparse interface. 4-of-2 sparseMatMul is a special matrix multiplication that requires at most 2 non-zero values in a continuous group of 4 weights or activation values, with the other 2 forced to be zero. The mathematical expression of the operator is:
  ```
  C = A * B
  ```
  where matrix B is the densified matrix. The original B matrix contains at least 2 zero elements in every 4 elements, compressed and stored through the 4-of-2 densification strategy. Matrix B must be transposed, i.e., input as [N,K]. Matrices A and B only support int8_t data type. Index matrix data type is uint8_t.

- Operator Specification:
  <table>
  <tr><td rowspan="4" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">128 * 128</td><td align="center">int8</td><td align="center">NZ</td></tr>
  <tr><td align="center">b</td><td align="center">64 * 128</td><td align="center">int8</td><td align="center">NZ</td></tr>
  <tr><td align="center">idx</td><td align="center">128 * 8</td><td align="center">uint8</td><td align="center">ZN</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">c</td><td align="center">128 * 128</td><td align="center">int32</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">mmad_with_sparse_custom</td></tr>
  </table>

- Operator Implementation:

  The operator implementation process includes the following steps:
  - **CopyIn**: Move input data from Global Memory to Local Memory L1, and load the index matrix to L1. The Index fractal and B matrix fractal must be Zn fractal, meaning the B matrix on GM must be transposed. Index offline generation must be in Zn layout.
  - **SplitB**: Use LoadDataWithSparse to move the B matrix and index matrix from L1 to L0B and the built-in index buffer.
  - **SplitA**: Move the A matrix from L1 to L0A.
  - **Compute**: Use MmadWithSparse to complete sparse matrix multiplication computation. The computation result is stored in Local Memory L0C.
  - **CopyOut**: Move output data from L0C to Global Memory output GM.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure environment variables

  Please select the corresponding environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.

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

- Example execution
  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program, run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output result correctness, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build option description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |

- Execution result

  The execution result is shown below, indicating the precision comparison passed.
  ```bash
  test pass!
  ```

## Data Generation and Verification Description

### gen_data.py Script Function

1. **Construct Sparse Matrix B**: Generate a sparse matrix of specified shape, where each row contains at least 2 zero elements in every 4-element block.
2. **Densification Processing**: Densify the sparse matrix B through the 4-of-2 strategy to generate the dense matrix dense_B.
3. **Index Matrix Generation**:
   - Generate index_matrix: Records the relative positions of selected elements in each block (used for NPU computation).
   - Generate index_mask_matrix: Records the absolute indices of selected elements (used for golden computation).
4. **Generate Golden Data**: Compute the ground truth for sparse matrix multiplication using the densified matrix and index matrix.
5. **Data Format Conversion**:
   - Convert the index matrix from uint8 to uint2 format.
   - Perform ND to NZ layout transposition for input matrices.