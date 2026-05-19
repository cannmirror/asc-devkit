# gather_ld_reg Example

## Overview
This example implements the functionality of transferring discrete data from UB (Unified Buffer) to RegTensor (Reg vector computation basic unit) based on Reg programming interfaces, supporting two scenarios selected through compile-time macros.
    <table>
 	 	<tr>
 	 		<td>scenarioNum</td>
 	 		<td>Transfer Scenario</td>
 	 	</tr>
 	 	<tr>
 	 		<td>1</td>
 	 		<td>Gather (collect single-point data from UB to RegTensor by index)</td>
 	 	</tr>
 	 	<tr>
 	 		<td>2</td>
 	 		<td>GatherB (collect DataBlock-sized data from UB to RegTensor by index)</td>
 	 	</tr>
  	 </table>

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── gather_ld_reg
│   ├── scripts
│   │   ├── gen_data.py                // Input data and ground truth data generation script
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── gather_ld_reg.asc              // AscendC example implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description
- Example Functionality:
  Collect elements discretely from source data to destination address based on index.

  **Scenario 1: Gather Mode**
  - Use Gather interface to collect data from UB to RegTensor by element index
  - Element-wise collection: dst[i] = src[index[i]]
  - Source data count is 1024 elements, index data count is 128 elements, output data count is 128 elements.
  - Example Specifications:
    <table>
    <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
    <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">src</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
    <tr><td align="center">index</td><td align="center">[1, 128]</td><td align="center">uint16</td></tr>
    <tr><td rowspan="1" align="center">Example Output</td><td align="center">dst</td><td align="center">[1, 128]</td><td align="center">half</td></tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="3" align="center">gather_ld_reg</td></tr>
    </table>
  - Example Implementation:
    In the GatherVF function, first load index data to indexReg via LoadAlign, then call Gather interface to collect data from source address to dstReg based on index, finally write the result back to UB via StoreAlign.
    - Invocation Implementation
      Use the kernel call operator `<<<>>>` to invoke the kernel function, launching 1 core.

  **Scenario 2: GatherB Mode**
  - Use GatherB interface for 32-byte DataBlock collection
  - Source data count is 1024 elements, index data count is 8 elements (corresponding to 8 DataBlocks), output data count is 128 elements.
  - Example Specifications:
    <table>
    <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
    <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">src</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
    <tr><td align="center">index</td><td align="center">[1, 8]</td><td align="center">uint32</td></tr>
    <tr><td rowspan="1" align="center">Example Output</td><td align="center">dst</td><td align="center">[1, 128]</td><td align="center">half</td></tr>
    <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="3" align="center">gather_ld_reg</td></tr>
    </table>
  - Example Implementation:
    In the GatherBVF function, first load index data to indexReg via LoadAlign, then call GatherB interface to collect data from source address to dstReg by DataBlock, finally write the result back to UB via StoreAlign.
    - Invocation Implementation
      Use the kernel call operator `<<<>>>` to invoke the kernel function, launching 1 core.

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
  SCENARIO=1                                                                     # Select execution scenario (1=Gather, 2=GatherB)
  mkdir -p build && cd build;                                                    # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;  # Build project (default npu mode)
  python3 ../scripts/gen_data.py -scenarioNum $SCENARIO                          # Generate test input data
  ./demo                                                                         # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build mode or scenario, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build Option Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `SCENARIO_NUM` | 1, 2 | Example execution scenario: Scenario 1=Gather mode, Scenario 2=GatherB mode |
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  When the execution result shows the following, it indicates successful precision comparison.
  ```bash
  test pass!
  ```