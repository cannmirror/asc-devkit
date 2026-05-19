# Copy Interface Example

## Overview

This example implements UB (Unified Buffer) internal data transfer based on the Copy interface, suitable for scenarios that require transferring data between different TPositions such as VECIN, VECCALC, and VECOUT. The example supports switching between different scenarios through compilation parameters, helping developers understand how to use the Copy interface.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── copy_ub2ub
│   ├── scripts
│   │   ├── gen_data.py             // Input data and golden data generation script
│   │   └── verify_result.py        // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt              // Build project file
│   ├── data_utils.h                // Data read/write functions
│   └── copy.asc                    // Ascend C example implementation & invocation example
```

## Scenario Description

This example selects different scenarios through the compilation parameter `SCENARIO_NUM`. All scenarios use ND data format with kernel function name `copy_custom`.

<table border="2">
<caption>Table 1: Scenario Configuration Reference</caption>
<tr><th>scenarioNum</th><th>Input Shape</th><th>Output Shape</th><th>Computation Mode</th><th>Description</th></tr>
<tr><td>1</td><td>[1, 512]</td><td>[1, 512]</td><td>Tensor high-dimensional slice computation</td><td>Source and destination operand space shared</td></tr>
<tr><td>2</td><td>[18, 64]</td><td>[18, 8]</td><td>Tensor high-dimensional slice computation</td><td>Source and destination operand space different</td></tr>
<tr><td>3</td><td>[18, 64]</td><td>[18, 8]</td><td>Counter mode</td><td>Source and destination operand space different</td></tr>
</table>

### Scenario Parameter Description

**Tensor high-dimensional slice computation**: Controls the number of elements participating in computation within each iteration through the mask parameter. Each DataBlock is 32B in size, containing 8 elements (for int32 type). The repeatTime parameter controls the number of iterations, and stride parameters control the address stride of source and destination operands.

**Counter mode**: The mask parameter represents the number of elements processed per Repeat, with total elements participating in computation being repeatTime * mask. Use SetMaskCount to set the computation mode and SetVectorMask to set the mask.

**Stride parameters**: {dstStride, srcStride, dstRepeatSize, srcRepeatSize} control the address strides of source and destination operands within the same iteration and between adjacent iterations.

- **Scenario 1**: Tensor high-dimensional slice computation, mask=64, repeatTime=8, stride={1, 1, 8, 8}. Source and destination operand space is shared, processing 64 elements per iteration, iterating 8 times, transferring a total of 512 elements.

- **Scenario 2**: Tensor high-dimensional slice computation, mask=8, repeatTime=18, stride={1, 1, 1, 8}. Transferring [18, 8] from [18, 64], srcRepeatSize=8 means source operand skips 64 elements per Repeat (jumping to next row), dstRepeatSize=1 means destination operand is compactly arranged, transferring a total of 144 elements.

- **Scenario 3**: Counter mode, mask=144, repeatTime=1, stride={1, 8, 8, 8}. Transferring [18, 8] from [18, 64], srcStride=8 means source operand address stride per DataBlock is 8 (taking the first 8 elements of each row), transferring a total of 144 elements.

## Example Description

- Example Specifications
  <table border="2">
  <caption>Table 2: Example Specifications</caption>
  <tr>
    <td align="center">Category</td>
    <td align="center">name</td>
    <td align="center">shape</td>
    <td align="center">data type</td>
    <td align="center">format</td>
  </tr>
  <tr>
    <td align="center">Example Input</td>
    <td align="center">x</td>
    <td align="center">[1, 512]/[18, 64]</td>
    <td align="center">int32</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">Example Output</td>
    <td align="center">z</td>
    <td align="center">[1, 512]/[18, 8]</td>
    <td align="center">int32</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">Kernel Function Name</td>
    <td colspan="4" align="center">copy_custom</td>
  </tr>
  </table>

- Example Implementation
  - Kernel Implementation
    - Calls DataCopy basic API to transfer data from GM (Global Memory) to UB (Unified Buffer)
    - Calls Copy interface to transfer data from UB (Unified Buffer) to UB (Unified Buffer), supporting both tensor high-dimensional slice computation and Counter mode
    - Calls DataCopy basic API to transfer data from UB (Unified Buffer) to GM (Global Memory)

- Invocation Implementation
  Uses the kernel call operator <<<>>> to invoke the kernel function.

## Compilation and Execution

Execute the following steps in the root directory of this example to compile and run the example.

- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
  - Default path, CANN software package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installation

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  SCENARIO_NUM=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute compiled executable to run the example
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin  # Verify output correctness
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:

  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching compilation modes, clean the cmake cache by executing `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Compilation Options Description

  | Option | Possible Values | Description |
  |--------|-----------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products; dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1` (default), `2`, `3` | Scenario number |

- Execution Result

  The following result indicates successful precision comparison:

  ```bash
  test pass!
  ```