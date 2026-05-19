# Squeeze Example

## Overview

This example implements the Squeeze operation using the Reg programming interface, primarily calling the Squeeze interface and StoreUnAlign/StoreUnAlignPost interfaces.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── squeeze
│   ├── scripts
│   │   ├── gen_data.py                // Input data and ground truth generation script
│   ├── CMakeLists.txt                 // Build configuration file
│   ├── data_utils.h                   // Data read/write functions
│   ├── squeeze.asc                    // AscendC example implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description

- Example functionality:

  This example uses MaskReg set to MaskPattern::M4. In each iteration, elements at indices that are multiples of 4 from the input vector xReg are sequentially copied to the output vector yReg in consecutive positions, and the remaining positions in the output vector are set to 0.

  - When the template parameter store of the Squeeze interface is configured as STORE_REG, it can record the count of valid elements and store it in the AR special register for use with the StoreUnAlign interface.

  - The StoreUnAlign interface can use the value recorded in the AR special register as the number of elements to transfer, implementing consecutive unaligned transfers to continuously output the Squeeze results.

- Constraints:

  - When the template parameter store of the Squeeze interface is configured as STORE_REG, the Squeeze interface and StoreUnAlign interface must be used alternately to ensure proper enabling of the AR special register.

  - Before calculation, the ClearSpr interface must be called to clear the AR register; otherwise, residual data may cause precision issues.

- Example specifications:

  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[1, 64]</td><td align="center">float</td></tr>
  <tr><td rowspan="1" align="center">Kernel Name</td><td colspan="4" align="center">squeeze</td></tr>
  </table>

- Example implementation:

  - The input vector has shape [1, 256] and float data type. It performs 4 iterations, processing 64 elements per iteration.

  - Squeeze: Each call selects elements at indices that are multiples of 4 from xReg, i.e., xReg[i * 4] is written consecutively to yReg[i]. The result y[0:16] contains 16 valid elements, and the remaining y[16:64] is set to 0. The AR special register value is set to 16.

  - StoreUnAlign: Uses the value in the AR register as the number of elements to transfer, moving the first 16 elements from yReg to the unaligned register ureg or output UB address yAddr.

  - StoreUnAlignPost: After 4 iterations complete, moves the remaining data from ureg.

  - Invocation implementation: Uses the kernel call operator <<<>>> to invoke the kernel function.

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
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;                      # Build project (default npu mode)
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build options description

| Option | Possible Values | Description |
|------|--------|------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution result

  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```