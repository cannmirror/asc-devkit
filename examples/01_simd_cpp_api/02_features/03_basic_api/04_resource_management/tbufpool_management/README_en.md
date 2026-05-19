# TBufPool Memory Management Example

## Overview

This example demonstrates TBufPool memory resource management using the `TPipe::InitBufPool` and `TBufPool::InitBufPool` interfaces, showcasing TBufPool resource allocation, memory partitioning, memory reuse, and custom TBufPool implementation. This example provides 3 different test scenarios.

> **Note:** This example only applies to the programming model based on TPipe and TQue.

<table>
  <tr>
    <td align="center">scenarioNum</td>
    <td align="center">Scenario Name</td>
    <td align="center">Description</td>
  </tr>
  <tr>
    <td align="center">1</td>
    <td align="center">TBufPool Memory Reuse</td>
    <td align="center">Use TPipe::InitBufPool to initialize two TBufPools, specifying that the second reuses the starting address and length of the first, implementing memory reuse</td>
  </tr>
  <tr>
    <td align="center">2</td>
    <td align="center">TBufPool Resource Subdivision</td>
    <td align="center">Use TBufPool::InitBufPool to further partition the entire resource block into smaller resource blocks and specify reuse relationships between sub-resource pools</td>
  </tr>
  <tr>
    <td align="center">3</td>
    <td align="center">Custom TBufPool</td>
    <td align="center">Use the EXTERN_IMPL_BUFPOOL macro to assist users in implementing custom TBufPool classes, enabling non-contiguous memory block allocation and memory sharing</td>
  </tr>
</table>

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── tbufpool_management
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── tbufpool_management.asc // Ascend C example implementation & invocation example
```

## Example Description

- Example Functionality

  This example demonstrates various usage patterns for TBufPool memory resource management, including memory reuse, resource subdivision, and custom TBufPool implementation.

- Example Specifications

  **Scenario 1: TBufPool Memory Reuse**
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
      <td align="center">[3, 65536]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">y</td>
      <td align="center">[3, 65536]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Example Output</td>
      <td align="center">z</td>
      <td align="center">[3, 65536]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Kernel Function Name</td>
      <td colspan="4" align="center">tbufpool_management_custom</td>
    </tr>
  </table>

  **Scenario 2: TBufPool Resource Subdivision**
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
      <td align="center">[4, 32768]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">y</td>
      <td align="center">[4, 32768]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Example Output</td>
      <td align="center">z</td>
      <td align="center">[4, 32768]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Kernel Function Name</td>
      <td colspan="4" align="center">tbufpool_management_custom</td>
    </tr>
  </table>

  **Scenario 3: Custom TBufPool**
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
      <td align="center">[1, 65536]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">y</td>
      <td align="center">[1, 65536]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Example Output</td>
      <td align="center">z</td>
      <td align="center">[1, 65536]</td>
      <td align="center">half</td>
      <td align="center">ND</td>
    </tr>
    <tr>
      <td align="center">Kernel Function Name</td>
      <td colspan="4" align="center">tbufpool_management_custom</td>
    </tr>
  </table>

- Example Implementation

  - Kernel Implementation

    **Memory Resource Management (Scenario Differences)**

    <table>
      <tr>
        <td align="center">scenarioNum</td>
        <td align="center">Memory Management Method</td>
        <td align="center">Implementation Description</td>
      </tr>
      <tr>
        <td align="center">1</td>
        <td align="center">TBufPool Memory Reuse</td>
        <td align="center">Call TPipe::InitBufPool to initialize tbufPool1 and tbufPool2, specifying tbufPool2 reuses the starting address and length of tbufPool1; call TBufPool::InitBuffer to allocate memory space for TQue</td>
      </tr>
      <tr>
        <td align="center">2</td>
        <td align="center">TBufPool Resource Subdivision</td>
        <td align="center">Call TPipe::InitBufPool to initialize tbufPool0, call TBufPool::InitBufPool to subdivide tbufPool1 and tbufPool2, specifying tbufPool2 reuses tbufPool1; call TBufPool::InitBuffer to allocate memory space for TQue</td>
      </tr>
      <tr>
        <td align="center">3</td>
        <td align="center">Custom TBufPool</td>
        <td align="center">Use the EXTERN_IMPL_BUFPOOL macro to implement a custom TBufPool class MyBufPool; call TPipe::InitBufPool to allocate memory for MyBufPool, and use InitBuffer to implement memory allocation for TQue and TBuf</td>
      </tr>
    </table>

    **Common Computation Flow**

    - Call the DataCopy basic API to move data from GM (Global Memory) to UB (Unified Buffer).
    - Call the Add interface to perform addition on two input tensors.
    - Call the DataCopy basic API to move computation results from UB (Unified Buffer) to GM (Global Memory).

  - Invocation Implementation

    Use the kernel call operator `<<<>>>` to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure Environment Variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, CANN package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, CANN package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  SCENARIO=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO   # Generate test input data
  ./demo                           # Execute the compiled executable program
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin   # Verify output correctness
  ```

  To use CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:
  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory and re-running cmake.

- Build Options Description

  | Option | Available Values | Description |
  | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1`, `2`, `3` | Scenario number: 1=TBufPool memory reuse, 2=TBufPool resource subdivision, 3=custom TBufPool |

- Execution Result

  The following output indicates successful precision comparison:
  ```bash
  test pass!
  ```