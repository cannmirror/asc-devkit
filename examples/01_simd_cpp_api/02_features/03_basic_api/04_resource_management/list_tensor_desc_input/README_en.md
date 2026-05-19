# ListTensorDescInput Example

## Overview

This example implements an AddN example based on the static Tensor programming model, using the ListTensorDesc structure to handle dynamic input parameters, combined with static memory allocation and event synchronization mechanisms to achieve coordinated scheduling of data transfer and computation tasks.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── list_tensor_desc_input
│   ├── CMakeLists.txt              // Build configuration file
│   └── list_tensor_desc_input.asc  // Ascend C example implementation & invocation example
```

## Example Description

- Example functionality

  This example uses Add computation to demonstrate the usage of the dynamic Tensor programming model, suitable for the following scenarios:

  1. Multi-input parameter dynamic processing: Supports dynamic combination operations of multiple input tensors in the model (e.g., multi-branch network structures).

  2. Memory pipeline optimization: Achieves pipeline parallelism of data transfer and computation through static double buffering and event synchronization mechanisms, reducing memory access latency.

  3. Multi-core parallel computation: Adapts to the multi-core architecture of AI processors, supporting efficient distribution of large-scale tensor operations.

- Example specifications

  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center">AddN</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x (dynamic input parameter srcList[0])</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y (dynamic input parameter srcList[1])</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[8, 2048]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Name</td><td colspan="4" align="center">list_tensor_desc_input_custom</td></tr>
  </table>

- Example implementation

  - Kernel implementation

    The dynamic input feature means that the kernel function input parameters use a ListTensorDesc structure to store input data information. Construct a TensorList data structure as shown below.

    ```cpp
    constexpr uint32_t SHAPE_DIM = 2;
    struct TensorDesc {
      uint32_t dim{SHAPE_DIM};
      uint32_t index;
      uint64_t shape[SHAPE_DIM] = {8, 2048};
    };

    constexpr uint32_t TENSOR_DESC_NUM = 2;
    struct ListTensorDesc {
      uint64_t ptrOffset;
      TensorDesc tensorDesc[TENSOR_DESC_NUM];
      uintptr_t dataPtr[TENSOR_DESC_NUM];
    } inputDesc;
    ```

    Combine the allocated Tensor input parameters into a ListTensorDesc data structure as shown below.

    ```cpp
    inputDesc = {(1 + (1 + SHAPE_DIM) * TENSOR_DESC_NUM) * sizeof(uint64_t),
                {xDesc, yDesc},
                {(uintptr_t)xDevice, (uintptr_t)yDevice}};
    ```

    Parse the corresponding input parameters according to the passed data format as shown below.

    ```cpp
    AscendC::ListTensorDesc keyListTensorDescInit((__gm__ void*)srcList);
    GM_ADDR x = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    GM_ADDR y = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(1);
    ```

  - Invocation implementation

    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure environment variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.

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

- Execute the example

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  ./demo                           # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build options description

  | Parameter | Description | Possible Values | Default Value |
  |------|------|---------|--------|
  | CMAKE_ASC_RUN_MODE | Run mode | npu, cpu, sim | npu |
  | CMAKE_ASC_ARCHITECTURES | NPU hardware architecture | dav-2201, dav-3510 | dav-2201 |

- Execution result

  The following execution result indicates successful precision comparison:

  ```bash
  [Success] Case accuracy is verification passed.
  ```