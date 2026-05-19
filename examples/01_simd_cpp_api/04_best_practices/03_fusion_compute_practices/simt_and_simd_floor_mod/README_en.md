# SIMT and SIMD Hybrid Programming floor_mod Operator Sample

## Overview

This sample uses the implementation of the floor_mod function operator as an example to demonstrate the operator development approach using SIMT and SIMD hybrid programming. The operator uses SIMD-based datacopy for data transfer in and out, float scenarios complete computation logic through SIMD, while int32_t scenarios complete computation functionality based on SIMT, improving integer scenario performance.


## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \>= CANN 9.0.0-beta.2

## Directory Structure

```
├── 14_simt_and_simd_floor_mod
│   ├── CMakeLists.txt         # cmake build file
│   ├── floor_mod.asc    # Ascend C operator implementation & invocation sample
|   └── README.md
```

## Operator Description

- Operator Functionality:  
  Input two tensors self and other, divide each element of self by the corresponding element of other to get the remainder. The result has the same sign as the divisor other, and the absolute value is less than the absolute value of other. The calculation formula for the i-th data element in operator output is:

  ```
  output[i] = self[i] - floor(self[i]/other[i]) * other[i]
  ```

- Operator Specification:  
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">floor_mod</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">self</td><td align="center">6400</td><td align="center">float/int32_t</td><td align="center">ND</td></tr>
  <tr><td align="center">other</td><td align="center">6400</td><td align="center">float/int32_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">output</td><td align="center">6400</td><td align="center">float/int32_t</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">floor_mod</td></tr>
  </table>

- Basic Knowledge:  
  SIMD programming provides register-based (Regbase) programming APIs that can directly operate registers in Vector Core. The upper limit of data processed by a single API call is the register size, which can be obtained through the AscendC::GetVecLen function. During computation, multiple micro-instruction API calls are needed to complete single-core data processing. In SIMT programming, data on Global Memory can be directly read and used. In Vector Core, SIMT unit and SIMD unit share on-chip storage, so on-chip storage can be used for SIMT and SIMD hybrid programming.
  In this example, float scenarios use SIMD to complete operator functionality, while in integer scenarios, SIMD completes data transfer in and out, and SIMT implements computation functionality.

- Data Splitting:  
  In this example, the operator has a total of 6400 input elements, each core processes 1024 elements, so the core count is 7.
  - In SIMT scenarios, set the actual number of threads started per core THREAD_COUNT to 1024, each thread is responsible for processing one element. Each thread's data position is indexed through threadIdx, then subsequently indexed by total thread count stride.
    ```cpp
    uint32_t index = threadIdx.x;
    ```
  - In SIMD scenarios, limited by Reg size for micro-instruction computation, a single operation can only process 256B data, so multiple loop iterations are needed to process 1024 single-core data elements.

- Operator Implementation:  
  This operator's implementation flow is mainly divided into 3 steps: CopyIn, Compute, CopyOut. CopyIn and CopyOut follow general SIMD operator development methods, which will not be elaborated here.
  In Compute, different processing methods are determined by data type.

  floor_mod_simt is responsible for computing several elements in integer scenarios.
  ```cpp
    uint32_t index = threadIdx.x;
    auto rem = self[index] % other[index];
    bool signs_differ = ((rem < 0) != (other[index] < 0));
    if (signs_differ && (rem != 0)) {
        out[index] = rem + other[index];
    } else {
        out[index] = rem;
    }
  ```

  floor_mod_simd is responsible for float scenario computation.
  ```cpp
  for (uint16_t j = 0; j < loopTimes; j++) {
        preg = AscendC::Reg::UpdateMask<T>(sregMask);
        AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_NORM>(fmodResValue, fmodResAddr + VL_T * j);
        AscendC::Reg::Compare<T, AscendC::CMPMODE::NE>(negValue, fmodResValue, zeroValue, preg);

        AscendC::Reg::And(fmodSignValue, (AscendC::Reg::RegTensor<uint32_t>&)fmodResValue, signValue, preg);
        AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_NORM>(inputX2Value, otherAddr + VL_T * j);
        AscendC::Reg::Add(addValue, fmodResValue, inputX2Value, preg);
        AscendC::Reg::And(inputX2signValue, (AscendC::Reg::RegTensor<uint32_t>&)inputX2Value, signValue, preg);
        AscendC::Reg::Compare<uint32_t, AscendC::CMPMODE::NE>(signNegValue, fmodSignValue, inputX2signValue, preg);

        AscendC::Reg::MaskAnd(resMaskValue, signNegValue, negValue, preg);
        AscendC::Reg::Select(resValue, addValue, fmodResValue, resMaskValue);
        AscendC::Reg::DataCopy<T, AscendC::Reg::StoreDist::DIST_NORM>(dstAddr + VL_T * j, resValue, preg);
    }
  ```

- Invocation Implementation:  
  Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the operator.
- Configure environment variables  
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in the current environment.
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
    
- Sample execution
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..; make -j;            # Build project
  ./demo                        # Execute sample
  ```
  Execution result shown below indicates accuracy comparison passed.
  ```
  [Success] Case accuracy is verification passed.
  ```
  The current sample code executes integer scenario functionality by default. You can execute float scenario functionality by modifying the `process_float()` method call in the main() function.
  ```
  int32_t main()
  {
      // process_float:  test float function
      // process_int:  test int32_t function
      return process_float();
  }
  ```