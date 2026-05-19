# SIMT Printf Implementation Example Based on Gather Operator

## Overview

This example demonstrates how to use the `printf()` interface in SIMT programming to implement on-board printing for function debugging.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \> CANN 9.0.0

## Directory Structure

```
├── 00_printf
│   ├── CMakeLists.txt         # cmake build file
│   ├── printf.asc             # Ascend C operator implementation with printf printing example
|   └── README.md
```

## Operator Description

- Operator function:

  This example demonstrates the practical usage of the `printf()` interface in SIMT implementation functions, enabling printing of variable information for each thread during operator execution.


- Operator implementation:
  ```cpp
  __global__ void  simt_printf(float* input, uint32_t in_shape)
  {
      // Calculate global thread ID
      int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (threadIdx.x < 3) {
      printf("[SIMT %s] thread index[%u], input data shape: %u\n", "print 1", idx, in_shape);
      printf("[SIMT %s] input addr: %p value[%u]: %f\n", "print 2",  input, idx, input[idx]);
      }
  }
  ```

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure environment variables
  Please select the appropriate environment variable configuration command based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, root user installation
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installation
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, custom installation
    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Execute the example
  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..; make -j;            # Build the project
  ./demo                        # Run the example
  ```
  After execution, the following output indicates that the printing function works correctly.
  ```
  [SIMT print 1] thread index[34], input data shape: 128
  [SIMT print 1] thread index[2], input data shape: 128
  [SIMT print 1] thread index[33], input data shape: 128
  [SIMT print 1] thread index[1], input data shape: 128
  [SIMT print 1] thread index[32], input data shape: 128
  [SIMT print 1] thread index[0], input data shape: 128
  [SIMT print 2] input addr: 0x120000016000 value[2]: 3.118000
  [SIMT print 2] input addr: 0x120000016000 value[1]: 2.118000
  [SIMT print 2] input addr: 0x120000016000 value[0]: 1.118000
  [SIMT print 2] input addr: 0x120000016000 value[34]: 35.118000
  [SIMT print 2] input addr: 0x120000016000 value[33]: 34.118000
  [SIMT print 2] input addr: 0x120000016000 value[32]: 33.118000
  ```