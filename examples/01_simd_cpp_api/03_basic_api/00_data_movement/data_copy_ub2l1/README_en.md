# DataCopy ub2l1 Example

## Overview

This example implements data transfer from UB (Unified Buffer) to L1 (L1 Buffer) based on DataCopy in a Mmad matrix multiplication scenario, covering two scenarios: continuous transfer and inline ND2NZ transfer.

## Supported Products and CANN Versions

| Product | CANN Version |
|---------|-------------|
| Ascend 950PR/Ascend 950DT | >= CANN 9.1.0 |

## Directory Structure

```
├── data_copy_ub2l1
│   ├── scripts
│   │   ├── gen_data.py                   // Input data and ground truth generation script
│   ├── CMakeLists.txt                    // Build configuration file
│   ├── data_utils.h                      // Data read/write functions
│   ├── data_copy_ub2l1.asc               // Ascend C example implementation & invocation example
│   └── README.md                         // Example description document
```

## Example Description

- Example functionality:
  Transfers data from UB (Unified Buffer) to L1 (L1 Buffer), then performs Mmad matrix multiplication computation, and finally transfers the result to GM (Global Memory) via Fixpipe.

- Scenario description:
  Two transfer scenarios are switched via the compile option `SCENARIO_NUM`:

  | Scenario | SCENARIO_NUM | Transfer API | Input Format | Description |
  |----------|-------------|--------------|--------------|-------------|
  | Continuous transfer | 1 | `DataCopy(dst, src, DataCopyParams)` | NZ | UB→L1 data content unchanged, input must be pre-converted to NZ format |
  | Inline ND2NZ transfer | 2 | `DataCopy(dst, src, Nd2NzParams)` | ND | Hardware completes ND→NZ format conversion during UB→L1 transfer |

  Refer to [UBToL1 Continuous Data Transfer](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/SIMD-API/基础API/矩阵计算（ISASI）/矩阵计算的搬入/矩阵数据搬入至L1-Buffer/UBToL1连续数据搬运（DataCopy）.md) and [UBToL1 Inline ND2NZ Transfer](https://gitcode.com/cann/asc-devkit/blob/master/docs/api/SIMD-API/基础API/矩阵计算（ISASI）/矩阵计算的搬入/矩阵数据搬入至L1-Buffer/UBToL1随路转换-ND2NZ搬运（DataCopy）.md) for API documentation.

- Example specifications:
  <table>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format (Scenario 1 / Scenario 2)</td></tr>
  <tr><td align="center">x</td><td align="center">[32, 32]</td><td align="center">half</td><td align="center">NZ / ND</td></tr>
  <tr><td align="center">y</td><td align="center">[32, 32]</td><td align="center">half</td><td align="center">NZ / ND</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[32, 32]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">data_copy_ub2l1</td></tr>
  </table>

- Example implementation:
  1. AIV core: Transfers data from GM (Global Memory) to UB (Unified Buffer). The data layout in GM differs between the two scenarios: Scenario 1 uses NZ format (fractal column-major, intra-fractal row-major, pre-converted by `gen_data.py`); Scenario 2 uses ND format (native row-major, no pre-conversion needed).
  2. AIV core: Transfers data from UB to L1 (L1 Buffer). Scenario 1 uses continuous transfer `DataCopy(dst, src, DataCopyParams)`, where UB→L1 data content is unchanged and L1 holds NZ format; Scenario 2 uses Nd2NzParams inline conversion `DataCopy(dst, src, Nd2NzParams)`, where hardware completes ND→NZ format conversion during transfer, and the L1 result is identical to Scenario 1. Both scenarios produce the same data layout in L1, and the subsequent computation flow is identical.
  3. AIC core: Calls the basic API LoadData to transfer data from L1 to L0A Buffer and L0B Buffer.
  4. AIC core: Calls the basic API Mmad to perform matrix multiplication computation.
  5. AIC core: Calls the basic API Fixpipe to transfer data from L0C Buffer to GM (Global Memory).

- Invocation implementation
  Uses the kernel invocation syntax <<<>>> to call the kernel function, declared as `__mix__(1, 2)` mixed core (1 AIC + 2 AIV).

## Build and Run

Run the following steps in the root directory of this example to build and run it.
- Configure environment variables
  Configure environment variables based on the [installation method](../../../../../docs/quick_start.md#prepare&install) of the CANN development kit on the current environment.
  ```bash
  source ${install_path}/cann/set_env.sh
  ```

  > **Note:** `${install_path}` is the CANN package installation directory. When no installation directory is specified, the default installation path is `/usr/local/Ascend`.

- Run the example

  Run the following commands in the example directory.
  ```bash
  mkdir -p build && cd build;                                               # Create and enter the build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=2 ..;make -j;    # Build the project, default npu mode, scenario 2 (ND2NZ)
  python3 ../scripts/gen_data.py --scenarioNum=2                            # Generate test input data (scenario must match compile option)
  ./demo                                                                    # Run the compiled executable to execute the example
  ```

  > **Note:** The `--scenarioNum` parameter of `gen_data.py` must match the `-DSCENARIO_NUM` of cmake, otherwise the input data format mismatch will cause precision verification failure.

  To use CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=1 ..;make -j; # CPU debug mode, scenario 1
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=2 ..;make -j; # NPU simulation mode, scenario 2
  ```

  > **Notice:** Clear the cmake cache before switching build modes or scenarios. Run `rm CMakeCache.txt` in the build directory and re-run cmake.

- Build option description

  | Option | Values | Description |
  |--------|--------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/950DT |
  | `SCENARIO_NUM` | `1` (default), `2` | Transfer scenario: 1=continuous transfer (NZ input), 2=inline ND2NZ transfer (ND input) |

- Execution result

  The following execution result indicates that the precision comparison is successful.
  ```bash
  test pass!
  ```
