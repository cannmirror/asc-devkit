# Batch Mmad Example

## Overview

This example introduces batch matrix multiplication with float data type input and both left and right matrices non-transposed. The three pathways GM-->L1, L0C-->GM, and L0C-->L1 use DataCopy ND2NZ and Fixpipe batch data movement respectively, while the steps from L1-->L0A/L0B and Mmad matrix multiplication execution loop batch times, processing one pair of left and right matrices in each loop iteration.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── batch_mmad
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for checking output data against golden data
│   ├── CMakeLists.txt              // Build configuration file
│   ├── data_utils.h                // Data read/write functions
│   └── batch_mmad.asc              // Ascend C example implementation & invocation example
```

## Operator Description

### Batch Mmad Definition

Batch matrix multiplication (batch mmad) is an extension of standard matrix multiplication on the batch dimension. The core logic is: for batch data containing multiple matrices, execute standard matrix multiplication for each matrix in the batch one by one, ultimately outputting result matrices with the same batch count.

Simply speaking, if there are two batch matrices A and B with shapes [B, M, K] and [B, K, N] respectively (where B is the batch size, M/K/N are matrix dimensions), batch matrix multiplication will take A[i] (shape [M, K]) and B[i] (shape [K, N]) for each batch index i (i ∈ [0, B-1]) and execute standard matrix multiplication, ultimately obtaining a batch result matrix C with shape [B, M, N]. For any batch i (0 ≤ i < B), the i-th matrix of C satisfies:
C[i]=A[i]×B[i].

Note that matrices from different batches do not compute with each other.

### Example Specifications

The specifications of input and output matrices in this example are shown in Table 1 below:

<table border="2">
<caption>Table 1: Input/Output Specifications</caption>
  <tr>
    <td >Input/Output</td>
    <td>Data Type</td>
    <td>Shape</td>
    <td>Transposed</td>
  </tr>
  <tr>
    <td>Input Matrix A</td>
    <td>float</td>
    <td>[4, 30, 40]</td>
    <td>false</td>
  </tr>
  <tr>
    <td>Input Matrix B</td>
    <td>float</td>
    <td>[4, 40, 70]</td>
    <td>false</td>
  </tr>
  <tr>
    <td>Output Matrix C</td>
    <td>float</td>
    <td>[4, 30, 70]</td>
    <td>-</td>
  </tr>
</table>

### Matrix Batch Load (GM->L1)

According to the batch mmad definition, there are B pairs of A and B matrices for matrix multiplication. When data moves through the GM-->L1 pathway, as shown below, when calling the inline conversion ND2NZ movement interface, by configuring `nd2nzA1Params.ndNum = B`, you can load B pairs of A and B matrices at once.

```cpp
// GM-->L1, move A matrix
AscendC::Nd2NzParams nd2nzA1Params;
// Number of ND matrices to transfer
nd2nzA1Params.ndNum = B;
// Number of rows in ND matrix
nd2nzA1Params.nValue = m;
// Number of columns in ND matrix
nd2nzA1Params.dValue = k;
// Offset between starting addresses of adjacent ND matrices in source operand, in elements
nd2nzA1Params.srcNdMatrixStride = m * k;
// Offset between starting addresses of adjacent rows in the same ND matrix in source operand, in elements
nd2nzA1Params.srcDValue = k;

// After ND conversion to NZ format, one row of source operand will be converted to multiple rows in destination operand.
// This parameter represents the offset between starting addresses of adjacent rows that come from the same row of source operand in destination NZ matrix, unit: C0_SIZE (32B).
// Data is aligned when moved to L1
nd2nzA1Params.dstNzC0Stride = CeilAlign(m, cubeShape[0]);
// Offset between starting addresses of adjacent rows in Z-type matrix in destination NZ matrix, unit: C0_SIZE (32B)
nd2nzA1Params.dstNzNStride = 1;
// Offset between starting addresses of adjacent NZ matrices in destination NZ matrix, in elements
nd2nzA1Params.dstNzMatrixStride = aSizeAlignL0;
```

### L1->L0A/L0B Movement and Matrix Multiplication Mmad Loop Execution B Times

For loop B times, each time moving each batch's A and B matrices from L1->L0A/L0B, the mmad instruction calculates the result of matrix multiplication for one pair of A and B matrices each time.

```cpp
for (int32_t batchIndex = 0; batchIndex < B; batchIndex++) {
        SplitA(a1Local[batchIndex * aSizeAlignL0]);
        SplitBTranspose(b1Local[batchIndex * bSizeAlignL0]);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

        Compute(batchIndex, c1Local);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
}
```

The for loop calculates the matrix computation result for each batch, stored in the corresponding position on L0C. After the loop ends, the complete result on L0C is moved out to GM.

```cpp
AscendC::Mmad(c1Local[batchIndex * CeilAlign(m, cubeShape[0]) * CeilAlign(n, cubeShape[0])],
              a2Local, b2Local, mmadParams);
```

### Matrix Batch Store

When data moves through the L0C-->GM pathway, as shown below, when calling the fixpipe movement interface, by configuring fixpipeParams.ndNum = B, you can store B pairs of C matrices at once. Note that C matrices in L0C are aligned, while C matrices moved out to GM have the original non-aligned shape.

```cpp
// Size of source NZ matrix in N direction
fixpipeParams.nSize = n;
// Size of source NZ matrix in M direction
fixpipeParams.mSize = m;
// Starting address offset of adjacent Z layouts in source NZ matrix, unit: C0_Size (16*sizeof(T), T is src data type)
fixpipeParams.srcStride = CeilAlign(m, cubeShape[0]);
// When NZ2ND function is enabled, represents the number of elements in each row of destination ND matrix, value is not 0, unit: element
fixpipeParams.dstStride = n;
// Number of source NZ matrices, i.e., number of ND matrices to transfer
fixpipeParams.ndNum = B;
// Interval between starting addresses of different NZ matrices, unit: 1024B
fixpipeParams.srcNdStride = (CeilAlign(m, cubeShape[0]) * CeilAlign(n, cubeShape[0])) 
                                / (cubeShape[0] * cubeShape[0]);
// Offset between starting addresses of adjacent destination ND matrices, unit: element
```

### Avoiding Data Occupying Total Memory Exceeding Storage Space Limits

Users should ensure that the total memory occupied by data during the entire batch mmad process does not exceed storage space limits.
Users can use the PlatformAscendC class member function [GetCoreMemSize](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/ascendcopapi/atlasascendc_api_07_1034.html) to obtain the memory size of L1, L0A, L0B, and L0C storage spaces on the hardware platform.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure Environment Variables
  Please select the corresponding command to configure environment variables according to the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
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

- Example Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify if output result is correct, confirm algorithm logic is correct
  ```

  When using CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. Execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products/Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

  The execution result is as follows, indicating precision comparison succeeded.

  ```bash
  test pass!
  ```