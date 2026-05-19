# GetTPipePtr Example

## Overview

This example demonstrates how to obtain the global TPipe pointer using the GetTPipePtr interface and perform TPipe-related operations through that pointer.

> **Note:** This example only applies to the programming model based on TPipe and TQue.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── get_tpipe_ptr
│   ├── CMakeLists.txt          // Build configuration file
│   └── get_tpipe_ptr.asc       // Ascend C example implementation & invocation example
```

## Example Description

- Example functionality

  When creating a TPipe object, the object initialization sets a globally unique TPipe pointer. This example calls the GetTPipePtr interface to obtain this pointer, allowing the kernel function to perform TPipe-related operations without explicitly passing the TPipe pointer. The following code snippets demonstrate examples of calling and not calling the GetTPipePtr interface.

  **Calling the GetTPipePtr interface**

  ```cpp
  class KernelAdd {
      // No TPipe member variable

      __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
      {
          // Call GetTPipePtr to get TPipe pointer and use it
          GetTPipePtr()->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
      }
  };

  __global__ __vector__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
  {
      // No need to explicitly pass TPipe pointer
      KernelAdd op;
      op.Init(x, y, z, tiling.totalLength, tiling.tileNum);
      op.Process();
  }
  ```

  **Not calling the GetTPipePtr interface (kernel function explicitly passes TPipe pointer)**

  ```cpp
  class KernelAdd {
      AscendC::TPipe* pipe;  // Requires TPipe pointer member variable

      __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum, AscendC::TPipe* pipeIn)
      {
          pipe = pipeIn;
          pipe->InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
      }
  };

  __global__ __vector__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
  {
      // Need to explicitly pass TPipe pointer
      AscendC::TPipe pipe;
      KernelAdd op;
      op.Init(x, y, z, tiling.totalLength, tiling.tileNum, &pipe);
      op.Process();
  }
  ```

- Example specifications

  <table>
    <tr>
      <td align="center">Category</td>
      <td align="center">name</td>
      <td align="center">shape</td>
      <td align="center">data type</td>
      <td align="center">format</td>
    </tr>
    <tr>
      <td rowspan="2" align="center">Example Input</td>
      <td align="center">x</td>
      <td align="center">[8, 2048]</td>
      <td align="center">float</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">y</td>
      <td align="center">[8, 2048]</td>
      <td align="center">float</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Example Output</td>
      <td align="center">z</td>
      <td align="center">[8, 2048]</td>
      <td align="center">float</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Kernel Name</td>
      <td colspan="4" align="center">get_tpipe_ptr_custom</td>
    </tr>
  </table>

- Example implementation

  - Kernel implementation

    - Calls the GetTPipePtr interface to obtain the global TPipe pointer.

    - Calls the TPipe::InitBuffer interface to allocate memory space for TQue.

    - Calls the DataCopy basic API to transfer data from GM (Global Memory) to UB (Unified Buffer).

    - Calls the Add interface to perform addition operation on two input tensors.

    - Calls the DataCopy basic API to transfer the computation result from UB (Unified Buffer) to GM (Global Memory).

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