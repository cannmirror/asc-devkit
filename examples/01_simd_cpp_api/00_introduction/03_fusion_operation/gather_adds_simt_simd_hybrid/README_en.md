# SIMT and SIMD Hybrid Programming for Gather and Adds Computation

## Overview

This sample implements gather and adds computation using SIMT and SIMD hybrid programming mode. It uses SIMT programming for discrete memory access operations (gather) and SIMD programming for continuous memory access operations (adds).

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions

- >= CANN 9.0.0-beta.2

## Directory Structure

```
├── gather_adds_simt_simd_hybrid
│   ├── scripts
│   │   ├── gen_data.py         # Input data and golden data generation script
│   │   └── verify_result.py    # Golden data comparison file
│   ├── CMakeLists.txt          # cmake build file
│   ├── gather_and_adds.asc     # Ascend C sample implementation & invocation sample
│   └── README.md
```

## Sample Description

- Sample Function:
  Computation formula:

  ```
  output[i] = input[index[i]] + 1
  ```

- Sample Specifications:
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">gather & adds</td></tr>
  <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">[100000]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">index</td><td align="center">[8192]</td><td align="center">uint32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">output</td><td align="center">[8192]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">gather_and_adds_kernel</td></tr>
  </table>

- Sample Implementation:
  The SIMT unit and SIMD unit in the Vector Core share on-chip memory, which can be used for SIMT and SIMD hybrid programming. In this sample, the input index has a shape of [8192]. You can set the number of cores to 8, with each core processing 1024 data elements. Set the thread count THREAD_COUNT to 1024, with each thread processing 1 data element. A single core only needs to call the simt_gather function once to complete the gather operation.

  > **Note**: When the data volume processed by a single core exceeds the set thread count, the data needs to be split into multiple thread blocks. You can use asc_vf_call to call the simt_gather function multiple times to start multiple thread blocks to complete the operation of fetching data at specified indices.

  Based on the above data partitioning, the simd_adds function performs the add-one operation on 1024 data elements.

  > **Note**: The add-one operation in simd_adds can actually be implemented directly in the simt_gather function. This sample aims only to demonstrate the hybrid programming approach of SIMT and SIMD programming modes through a simple use case, and is not the best practice for this sample.

  The implementation flow of the gather & adds sample consists of 3 main steps: simt_gather, simd_adds, and DataCopy.

  (1) simt_gather fetches data at specified indices from GM (Global Memory).
  ```
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  ...
  uint32_t gatherIdx = index[idx];
  ...
  gatherOutput[threadIdx.x] = input[gatherIdx];
  ```

  (2) simd_adds performs the add-one operation on data in UB (Unified Buffer). Call Reg::LoadAlign to move data from UB (Unified Buffer) to registers, call Reg::Adds to complete the add-one operation and output to the destination register, and finally call Reg::StoreAlign to move data from registers to UB. Repeat the above operation to complete the add-one operation on 1024 data elements.
  ```
  for (uint16_t i = 0; i < repeatTimes; i++) {
      AscendC::Reg::LoadAlign(srcReg0, input + i * oneRepeatSize);
      AscendC::Reg::Adds(dstReg0, srcReg0, ADDS_ADDEND, maskReg);
      AscendC::Reg::StoreAlign(output + i * oneRepeatSize, dstReg0, maskReg);
  }
  ```

  (3) DataCopy is responsible for moving output data from UB (Unified Buffer) to GM (Global Memory).

- Invocation Implementation:
  Use the kernel invocation operator <<<>>> to call the kernel function.

## Compilation and Execution

Execute the following steps in the root directory of this sample to compile and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment.
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
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                      # Compile project (default npu mode)
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results are correct
  ```

  When using NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note**: Before switching compilation modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then run cmake again.

- Compilation Options Description

| Option | Available Values | Description |
|--------|------------------|-------------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU execution, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result
  The execution result is as follows, indicating that the accuracy comparison passed.
  ```
  [Success] Case accuracy is verification passed.
  ```