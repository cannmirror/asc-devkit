# Gather Operator Sample Using SIMT Programming Model

## Overview

This sample implements a Gather operator for a simple scenario (fixed shape) using Ascend C SIMT programming, collecting specified m rows of data from an input tensor, demonstrating the development of operators with discrete memory access patterns in simplified scenarios.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \>= CANN 9.0.0

## Directory Structure

```
├── basic_gather
│   ├── gather.asc             # SIMT implementation of gather call sample
|   └── README.md
```

## Operator Description

- Operator functionality:
  The gather operator retrieves 12288 rows of data at specified indices from a two-dimensional vector with shape 100000 * 128. The calculation formula for the i-th row of the operator output is:
  
  ```
  output[i] = input[index[i]]
  ```

- Operator specifications:
  <table>
  <tr><td rowspan="1" align="center">Operator Type (OpType)</td><td colspan="4" align="center">gather</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">100000 * 128</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">index</td><td align="center">12288</td><td align="center">uint32_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">output</td><td align="center">12288 * 128</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">gather_custom</td></tr>
  </table>

- Data partitioning:
  * Number of cores: 48 cores
  * Threads per core: 256 threads
  * Per-thread processing: 1 row (128 columns)
  * Total processing capacity: 48×256=12288 rows (covers index length)

- Operator implementation:
  The gather operator implementation retrieves data at specified indices from the input (Global Memory). Based on the above data partitioning, first calculate the index of data that the thread should process, then store one row of data to Global Memory through assignment operation.

- Invocation implementation:
  Use the kernel call operator <<<>>> to invoke the kernel function.

## Compilation and Execution

Follow the steps below in the sample root directory to compile and execute the operator.
- Configure environment variables
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
    
- Sample execution  
```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..; make -j;            # Build project
  ./gather                      # Execute sample
  ```
  If the execution result is as follows, the accuracy comparison has passed.
  ```
  [Success] Case accuracy is verification passed.
  ```