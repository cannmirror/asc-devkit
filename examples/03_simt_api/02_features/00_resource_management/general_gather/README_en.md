# Gather Operator Sample Using SIMT Programming Model

## Overview

This sample implements a Gather operator supporting generalized shapes using Ascend C SIMT programming, including the basic gather and enhanced gather_v2. The gather operator collects row data at specified indices from a two-dimensional input tensor, while the gather_v2 operator supports collecting data from multi-dimensional input tensors along a specified dimension and supports batch_dims batch processing mode. The sample demonstrates the development of operators with discrete memory access patterns in generalized scenarios.

## Supported Products
- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \>= CANN 9.0.0

## Directory Structure
```text
├── general_gather
│   ├── CMakeLists.txt         # cmake build file
│   ├── gather_v2.asc          # SIMT implementation of gather_v2 call sample
│   ├── gather.asc             # SIMT implementation of gather call sample
│   └── README.md
```

## Operator Description
### 1. gather operator

- Operator functionality:
  The gather operator retrieves m rows of data at specified indices from a two-dimensional input tensor with shape M * N. The row indices for these m rows are specified by the input index. The calculation formula for the i-th row of the operator output is:
  ```text
  output[i] = input[index[i]]
  ```

- Operator specifications:
  <table>
  <tr><td rowspan="1" align="center">Operator Type (OpType)</td><td colspan="4" align="center">gather</td></tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">M, N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">index</td><td align="center">m (m < M, m < 65535 * 2048)</td><td align="center">uint32_t</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">output</td><td align="center">m, N</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">gather_custom</td></tr>
  </table>

- Data partitioning:
  * gridDim: Dynamically allocated based on specific input shape, maximum not exceeding 65535
  * blockDim: Dynamically allocated based on specific input shape, maximum not exceeding 2048
  * Per-thread processing: 1 row
  * Maximum processing capacity: 65535 * 2048 = 134215680 rows

- Operator implementation:
  The gather operator implementation retrieves data at specified indices from the input (Global Memory). Based on the above data partitioning, first calculate the index of data that the thread should process, then store one row of data to Global Memory through assignment operation. Since the computation process is relatively simple, the kernel function's maximum thread limit is set to 2048.

- Invocation implementation:
  Use the kernel call operator <<<>>> to invoke the kernel function.

### 2. gather_v2 operator
- Operator functionality:
  The gather_v2 operator collects data from a multi-dimensional input tensor along a specified dimension (axis). The indices tensor specifies the index positions to collect. It supports batch_dims batch processing mode, allowing different batches to use different index sets.
- Processing flow:
  For example, if the input tensor shape is (2, 2, 3, 2) and the indices tensor shape is (2, 2):
  ```text
  input:
   [[[[ 1,  2],
      [ 3,  4],
      [ 5,  6]],

     [[ 7,  8],
      [ 9, 10],
      [11, 12]]],


    [[[13, 14],
      [15, 16],
      [17, 18]],

     [[19, 20],
      [21, 22],
      [23, 24]]]]

  indices:
   [[1, 2],
    [0, 1]]
  ```
  axis=2, batch_dims=1 indicates collecting along dimension 2, with each batch using different indices:
  - batch=0: output[0, :, :, :] = input[0, :, [1, 2], :], that is, collecting slices corresponding to indices[0] along dimension 2 of input[0]
  - batch=1: output[1, :, :, :] = input[1, :, [0, 1], :], that is, collecting slices corresponding to indices[1] along dimension 2 of input[1]
  ```text
  output:
   [[[[ 3,  4],
      [ 5,  6]],

     [[ 9, 10],
      [11, 12]]],

    [[[13, 14],
      [15, 16]],

     [[19, 20],
      [21, 22]]]]
  ```
 
- Operator specifications:
  <table>
  <tr><td rowspan="1" align="center">Operator Type (OpType)</td><td colspan="4" align="center">gather_v2</td></tr>
  <tr><td rowspan="5" align="center">Operator Input</td><td align="center">name</td><td align="center">data type</td><td align="center">format</td><td align="center">description</td></tr>
  <tr><td align="center">input</td><td align="center">float</td><td align="center">ND</td><td align="center">Multi-dimensional input tensor</td></tr>
  <tr><td align="center">indices</td><td align="center">uint32_t / int32_t</td><td align="center">ND</td><td align="center">Indices tensor, specifying collection positions</td></tr>
  <tr><td align="center">axis</td><td align="center">int32_t</td><td align="center">-</td><td align="center">Scalar, used to specify the collection dimension</td></tr>
  <tr><td align="center">batch_dims</td><td align="center">int32_t</td><td align="center">-</td><td align="center">Scalar, used to specify batch processing dimensions</td></tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">output</td><td align="center">float</td><td align="center">ND</td><td align="center">Collected output tensor</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">gather_custom_v2</td></tr>
  </table>
- Constraint description:
  * indices: The first batch_dims dimensions of indices must match the first batch_dims dimensions of input, that is, indices.shape[0:batch_dims] = input.shape[0:batch_dims]
  * axis: The collection dimension axis cannot be less than batch_dims and cannot exceed the number of dimensions of input, that is, batch_dims <= axis < input.rank
  * batch_dims: The number of batch dimensions cannot exceed the smaller of the number of dimensions in input and indices, that is, 0 <= batch_dims <= min(input.rank, indices.rank)
  * output.shape: input.shape[:axis] + indices.shape[batch_dims:] + input.shape[axis+1:]
  * In the sample implementation, axis and batch_dims also support negative values, which are converted to corresponding non-negative dimension indices before computation
- Data partitioning:
  * gridDim: Dynamically allocated based on total collected data volume. Priority is given to calculating the number of blocks as needed based on blockDim. When the required number of blocks exceeds the device AIV core count, the device AIV core count is used as the number of blocks
  * blockDim: Dynamically allocated based on specific total data volume, maximum not exceeding 2048
  * Per-thread processing: Dynamically balanced based on specific collected data volume, with a maximum difference of 1 element per thread. In the kernel function, the total number of threads is used as the loop stride. Each thread starts from position begin = blockIdx.x * blockDim.x + threadIdx.x and traverses and processes elements with stride = gridDim.x * blockDim.x

  Advantages of this partitioning approach:
  * Load balancing: The workload difference among all threads is at most 1 element, avoiding thread idling
  * Memory access friendly: Adjacent threads access consecutive memory addresses, facilitating coalesced memory access
- Operator implementation:
  The gather_v2 operator implementation collects data from a multi-dimensional input tensor along a specified dimension (axis). Based on the above data partitioning strategy, each thread dynamically processes a portion of data, decomposes each output index, converts the one-dimensional output index to logical coordinates, finds the corresponding collection position based on indices, and finally calculates the linear index of input and completes data transfer.
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
  ./gather_v2
  ```
  If the execution result is as follows, the accuracy comparison has passed:
  ```text
  [Success] Case accuracy is verification passed.
  ```