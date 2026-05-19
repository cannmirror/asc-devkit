# ld_st_reg_unalign Example

## Overview
This example implements unaligned data transfer operations from UB (Unified Buffer) to RegTensor (Reg vector computation basic unit) based on Reg programming interfaces, supporting multiple scenarios selected through environment variables.
    <table>
 		<tr>
	 		<td>scenarioNum</td>
	 		<td>Scenario Type</td>
	 	</tr>
	 	<tr>
	 		<td>1</td>
	 		<td>Single-core unaligned data transfer: Use LoadUnAlign and StoreUnAlign for data transfer at unaligned addresses</td>
	 	</tr>
	 	<tr>
	 		<td>2</td>
	 		<td>Multi-core unaligned scenario computation and transfer: Use LoadUnAlign to handle unaligned input (uint16_t), use ReduceDataBlock (MAX) for in-block maximum calculation</td>
	 	</tr>
 	 </table>


## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── ld_st_reg_unalign
│   ├── scripts
│   │   ├── gen_data.py                // Input data and ground truth data generation script
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── ld_st_reg_unalign.asc          // AscendC example implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description

### Scenario 1: Single-Core Address Unaligned Data Transfer
- Example Functionality:
  Input a float type matrix with shape [1, 1024], transfer 128 data elements when both source operand and destination operand addresses are not 32B aligned.
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[1, 128]</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">ld_st_reg_unalign_kernel</td></tr>
  <tr><td rowspan="1" align="center">Number of Cores</td><td colspan="4" align="center">1</td></tr>
  </table>
- Example Implementation:
   Offset the source operand and destination operand addresses on UB by one element (4 Byte) length, making the starting address non-32B aligned. Call CopyVF to transfer 128 float numbers from that address and write back to UB (Unified Buffer).

### Scenario 2: Multi-Core Address Unaligned, Data Count Unaligned Scenario Transfer
- Example Functionality:
  Input a uint16_t type matrix with shape [14, 255], demonstrating the natural use case of LoadUnAlign handling unaligned input. Use ReduceDataBlock to compute the maximum value for each DataBlock (16 uint16_t elements), with 4 cores processing in parallel, and StoreUnAlign for contiguous result storage.
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[14, 255]</td><td align="center">uint16_t</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">y</td><td align="center">[14, 16]</td><td align="center">uint16_t</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">ld_st_reg_unalign_kernel</td></tr>
  <tr><td rowspan="1" align="center">Number of Cores</td><td colspan="4" align="center">4</td></tr>
  </table>
- Example Implementation:
  1. **Data Load**:
     - Cores 0, 1, 2 transfer 4×255=1020 elements, core 3 transfers 2×255=510 elements, contiguous block load to UB
     - UB tightly packed, each row spaced by 255 elements
     - Row 1-3 starting addresses are not 32B aligned (510 bytes, 1020 bytes, 1530 bytes)

  2. **LoadUnAlign Natural Use**:
     - All rows uniformly use LoadUnAlignPre + LoadUnAlign to handle aligned addresses (row 0) and unaligned addresses (row 1-3)
     - Aligned addresses can also use StoreAlign interface. Currently this is done to maintain consistent code format for example demonstration, not intended as extreme performance reference

  3. **ReduceDataBlock Computation**:
     - Each row with 255 elements processed in 2 VL blocks (128 + 127)
     - Each VL block produces 8 results (maximum value per DataBlock)

  4. **StoreUnAlign Contiguous Storage**:
     - Use StoreUnAlign version with postUpdateStride
     - Each store offsets by 8 elements (resultsPerRepeat=8)
     - After 2 iterations, automatically offsets to dstAddr+16
     - StoreUnAlignPost handles final boundary data

  5. **Output Layout**:
     - Each row has 16 results, data contiguously arranged
     - Core 0, core 1, core 2 output 64 values (GM[0-63], GM[64-127], GM[128-191]), core 3 outputs 32 values (GM[192-223])
   
  - Invocation Implementation
    Use the kernel call operator <<<>>> to invoke the kernel function, launching 4 cores.

## Build and Run
Execute the following steps in the root directory of this example to build and run the example.
- Configure Environment Variables
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
    
- Example Execution
  ```bash
  SCENARIO=1                                                                    # Set scenario number
  mkdir -p build && cd build;                                                   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j; # Build project (default npu mode)
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO                         # Generate test input data
  ./demo                                                                        # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  SCENARIO=1
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build mode or scenario, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build Option Description

| Option | Available Values | Description |
| | ----------------| -----------------------------| ---------------------------------------------------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
| `SCENARIO_NUM` | `1`, `2` | Scenario number: 1=Single-core address unaligned data transfer, 2=Multi-core address unaligned, data count unaligned data transfer |

- Execution Result
  When the execution result shows the following, it indicates successful precision comparison.
  ```bash
  test pass!
  ```