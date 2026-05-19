# Region Proposal Sort Pipeline Example

## Overview

This example implements a complete sorting pipeline using four interfaces: ProposalConcat, RpSort16, MrgSort4, and ProposalExtract. First, ProposalConcat merges consecutive score values into Region Proposal format, then RpSort16 sorts each group of 16 Region Proposals in descending order by score field, then MrgSort4 merges 4 groups of sorted Region Proposals into 1 group, and finally ProposalExtract extracts the score field from the merged Region Proposals to obtain consecutive score values in global descending order.

## Supported Products

- Atlas Inference Series Products AI Core

## Directory Structure

```
├── region_proposal_sort
│   ├── scripts
│   │   ├── gen_data.py                    // Input data and golden data generation script
│   │   └── verify_result.py              // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt                    // Build configuration file
│   ├── data_utils.h                      // Data read/write functions
│   └── region_proposal_sort.asc          // Ascend C implementation & invocation example
```

## Example Description

- Example functionality:  
  Sorts 64 score values through a complete pipeline. A Region Proposal consists of 8 half elements, where the 5th element (offset=4) is the score field.

- Example specifications:

<table border="2">
<caption>Table 1: Region Proposal Sort Example Specifications</caption>
<tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 64]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">Example Output</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[1, 64]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">region_proposal_sort_custom</td></tr>
</table>

- Example implementation:  
  This example implements the complete Region Proposal sorting pipeline:
  1. **ProposalConcat**: Merges 16 consecutive score values into the score field position (offset=4) of 16 Region Proposals, 4 groups total, each group with 1 repeat
  2. **RpSort16**: Sorts each group of 16 Region Proposals in descending order by score field, 4 groups total
  3. **MrgSort4**: Merges 4 groups of sorted Region Proposals into 1 group, with results sorted in descending order by score
  4. **ProposalExtract**: Extracts the score field from 64 merged Region Proposals to obtain 64 consecutive score values in descending order

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
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2002 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin  # Verify output correctness
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2002 ..;make -j; # cpu debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2002 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build options description

  | Option | Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2002` (default) | NPU architecture: dav-2002 corresponds to Atlas Inference Series Products AI Core |

- Execution result

  The following result indicates successful accuracy comparison.

  ```bash
  test pass!
  ```