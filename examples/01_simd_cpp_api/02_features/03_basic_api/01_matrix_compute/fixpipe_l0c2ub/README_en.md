# fixpipe_l0c2ub Example

## Overview

This example introduces how to use Fixpipe to move matrix multiplication results from CO1 (L0C Buffer) to UB (Unified Buffer), supporting various output formats (Nz, ND) and dual destination mode (split by M dimension or N dimension) functions. These interfaces are used to efficiently transfer matrix multiplication computation results from L0C to the unified buffer, supporting various data format conversions and split capabilities.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── fixpipe_l0c2ub
│   ├── scripts
│   │   ├── gen_data.py                // Input data and golden data generation script
│   │   └── verify_result.py           // Verification script for checking output data against golden data
│   ├── CMakeLists.txt                 // Build configuration file
│   ├── data_utils.h                   // Data read/write functions
│   └ fixpipe_l0c2ub.asc             // Ascend C example implementation & invocation example
```

## Scenario Detailed Description

This example selects different output scenarios through the build parameter `SCENARIO_NUM`. The meanings corresponding to different values of SCENARIO_NUM are shown in the table below. All scenarios are based on the same matrix multiplication specification: [M, N, K] = [128, 256, 128], kernel function name is `fixpipe_l0c2ub`.

<table>
<caption style="font-weight: normal;">
  	     <span style="font-weight: bold; font-size: 1.2em;">Table 1: Meanings of Different scenarioNum Values</span>
<tr><td rowspan="1" align="center">scenarioNum</td><td align="center">L0C Data Type</td><td align="center">Output Data Type</td><td align="center">Output Format</td><td align="center">Dual Destination Mode</td><td align="center">Split Dimension</td></tr>
<tr><td align="center">1</td><td align="center">float</td><td align="center">float</td><td align="center">Nz</td><td align="center">No</td><td align="center">-</td></tr>
<tr><td align="center">2</td><td align="center">float</td><td align="center">float</td><td align="center">ND</td><td align="center">No</td><td align="center">-</td></tr>
<tr><td align="center">3</td><td align="center">float</td><td align="center">float</td><td align="center">ND</td><td align="center">Yes</td><td align="center">M dimension</td></tr>
<tr><td align="center">4</td><td align="center">float</td><td align="center">float</td><td align="center">ND</td><td align="center">Yes</td><td align="center">N dimension</td></tr>
</table>

**Scenario 1: Output Format Nz, Output Data Type float**
- Input: A [128, 128] half type, ND format; B [128, 256] half type, ND format
- Output: C [128, 256] float type, Nz format
- Implementation: Use `Fixpipe<outputType, l0cType, CFG_NZ_UB>` to move data from CO1 to UB, output as Nz format
- Description: CO1 data is in Nz format directly output to UB as Nz format, data maintains original format unchanged

**Scenario 2: Output Format ND, Output Data Type float**
- Input: A [128, 128] half type, ND format; B [128, 256] half type, ND format
- Output: C [128, 256] float type, ND format
- Implementation: Use `Fixpipe<outputType, l0cType, CFG_ROW_MAJOR_UB>` to specify ROW_MAJOR format conversion
- Description: Convert Nz format data in CO1 to ND format and output to UB

**Scenario 3: Output Format ND, Output Data Type float, Enable Dual Destination Mode, Split by M Dimension, Simultaneously Write to Two Sub Blocks (SUB BLOCK) UB**
- Input: A [128, 128] half type, ND format; B [128, 256] half type, ND format
- Output: Single sub block C [64, 256] float type, ND format (dual destination mode, split by M dimension, each destination outputs 64 rows)
- Implementation: Set `fixpipeParams.dualDstCtl = 0b01`, split by M dimension, M must be a multiple of 2
- Description: Use dual destination mode to split data output to UB, two cores split by M dimension each process half of the data
<p align="center">
  <img src="figures/fixpipe_l0c2ub_split_m.png" width="500">
</p>

**Scenario 4: Output Format ND, Output Data Type float, Enable Dual Destination Mode, Split by N Dimension, Simultaneously Write to Two Sub Blocks (SUB BLOCK) UB**
- Input: A [128, 128] half type, ND format; B [128, 256] half type, ND format
- Output: Single sub block C [128, 128] float type, ND format (dual destination mode, split by N dimension, each destination outputs 128 columns)
- Implementation: Set `fixpipeParams.dualDstCtl = 0b10`, split by N dimension, N must be a multiple of 32
- Description: Use dual destination mode to split data output to UB, two cores split by N dimension each process half of the data
<p align="center">
  <img src="figures/fixpipe_l0c2ub_split_n.png" width="500">
</p>

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
  Please select the corresponding command to configure environment variables according to the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on the current environment.
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

- Example Execution
  ```bash
  SCENARIO_NUM=1
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled executable program to run the example
  python3 ../scripts/verify_result.py output/output.bin ./output/golden.bin  # Verify if output result is correct
  ```
    When using NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:
  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clean the cmake cache. Execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU run, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | 1-4 | Scenario number |

  The execution result is as follows, indicating precision comparison succeeded.
  ```bash
  test pass!
  ```