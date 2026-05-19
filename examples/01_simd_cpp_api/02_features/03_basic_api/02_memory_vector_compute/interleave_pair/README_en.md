# Interleave/DeInterleave Example

## Overview

This example implements element interleaving and de-interleaving functionality using the Interleave and DeInterleave interfaces. Interleave stores elements from two source operands interleaved into two result operands, while DeInterleave stores elements from two source operands de-interleaved into two result operands. The example supports switching between different scenarios via compilation parameters, helping developers understand the usage and implementation differences of these two interfaces.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── interleave_pair
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── interleave_pair.asc     // Ascend C implementation & invocation example
```

## Scenario Details

This example switches between different scenarios via the compilation parameter `SCENARIO_NUM`:

<table border="2">
<caption>Table 1: Scenario Configuration Reference</caption>
<tr><th>scenarioNum</th><th>Interface</th><th>Input Shape</th><th>Output Shape</th><th>Data Type</th><th>Description</th></tr>
<tr><td>1</td><td>Interleave</td><td>[1, 512], [1, 512]</td><td>[1, 512], [1, 512]</td><td>half</td><td>Store elements from two source operands interleaved into two result operands</td></tr>
<tr><td>2</td><td>DeInterleave</td><td>[1, 512], [1, 512]</td><td>[1, 512], [1, 512]</td><td>half</td><td>Store elements from two source operands de-interleaved into two result operands</td></tr>
</table>

**Scenario 1: Interleave Element Interleaving**
- Input shape: src0=[1, 512], src1=[1, 512]
- Output shape: dst0=[1, 512], dst1=[1, 512]
- Data type: half
- Parameters: count=512
- Implementation:

    ```cpp
    AscendC::Interleave(dst0Local, dst1Local, src0Local, src1Local, count);
    ```

- Description: Interleaves elements from src0 and src1 into dst0 and dst1. dst0 contains the interleaved result of the first half of src0 and the first half of src1, dst1 contains the interleaved result of the second half of src0 and the second half of src1.
- Example:
  - Input src0: [1 2 3 ... 512]
  - Input src1: [513 514 515 ... 1024]
  - Output dst0: [1 513 2 514 ... 256 768]
  - Output dst1: [257 769 258 770 ... 512 1024]

**Scenario 2: DeInterleave Element De-interleaving**
- Input shape: src0=[1, 512], src1=[1, 512]
- Output shape: dst0=[1, 512], dst1=[1, 512]
- Data type: half
- Parameters: count=512
- Implementation:

    ```cpp
    AscendC::DeInterleave(dst0Local, dst1Local, src0Local, src1Local, count);
    ```

- Description: De-interleaves elements from src0 and src1 into dst0 and dst1. dst0 contains elements at odd index positions from src0 and src1, dst1 contains elements at even index positions from src0 and src1.
- Example:
  - Input src0: [1 2 3 ... 512]
  - Input src1: [513 514 515 ... 1024]
  - Output dst0: [1 3 5 ... 511 513 515 ... 1023]
  - Output dst1: [2 4 6 ... 512 514 516 ... 1024]

## Build and Run

Execute the following steps in the example root directory to build and run the example.

- Configure environment variables  
  Select the appropriate command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit in your current environment.
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
    
- Example execution

  ```bash
  SCENARIO_NUM=1  # Set scenario number
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py ./output/output_dst0.bin ./output/output_dst1.bin ./output/golden_dst0.bin ./output/golden_dst1.bin  # Verify output correctness
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build options description

  | Option | Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT, this example only supports this architecture |
  | `SCENARIO_NUM` | `1` (default), `2` | Scenario number: 1 (Interleave), 2 (DeInterleave) |

- Execution result

  The following result indicates successful accuracy comparison.

  ```bash
  test pass!
  ```