# ld_st_reg_align Example

## Overview
This example implements contiguous and non-contiguous aligned data transfer operations from UB (Unified Buffer) to RegTensor (Reg vector computation basic unit) based on Reg programming interfaces. This example uses LoadAlign, StoreAlign interfaces, and enables POST_MODE_UPDATE, DATA_BLOCK_COPY modes. This example supports 6 transfer scenarios selected through environment variables.
    <table>
      <tr>
        <td>scenarioNum</td>
        <td>Transfer Scenario</td>
      </tr>
      <tr>
        <td>1</td>
        <td>Using developer-defined inter-iteration offset</td>
      </tr>
      <tr>
        <td>2</td>
        <td>Using PostUpdate mode to represent inter-iteration offset</td>
      </tr>
      <tr>
        <td>3</td>
        <td>Using address register (AddrReg) to represent inter-iteration offset</td>
      </tr>
      <tr>
        <td>4</td>
        <td>Non-contiguous transfer in DataBlock units</td>
      </tr>
      <tr>
        <td>5</td>
        <td>Broadcast mode load</td>
      </tr>
      <tr>
        <td>6</td>
        <td>Upsample mode load</td>
      </tr>
    </table>

## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure
```
├── ld_st_reg_align
│   ├── scripts
│   │   ├── gen_data.py                // Input data and ground truth data generation script
│   ├── CMakeLists.txt                 // Build project file
│   ├── data_utils.h                   // Data read/write functions
│   ├── ld_st_reg_align.asc            // AscendC example implementation & invocation example
│   └── README.md                      // Example introduction
```

## Example Description
**Scenario 1: Using Developer-Defined Inter-Iteration Offset**
- Example Functionality:
  Reconfigure load/store addresses at each iteration to implement contiguous data load/store. Also demonstrates handling non-VL-aligned data in the final iteration using MaskReg when output data count is less than VL.
- Example Implementation:
  - Basic scenario, LoadAlign and StoreAlign use default parameter configuration.
  - Input 1024 elements, output 1021 elements, requires 8 iterations. The final iteration only calculates and stores 125 elements, less than VL.
  - In the VF function for loop, each iteration uses UpdateMask interface to update mask. For half type:
    - When outputLength >= 128, mask processes 128 elements
    - When outputLength < 128, mask processes outputLength elements
    - After UpdateMask execution, outputLength decrements by the number of elements processed by mask.
    ```cpp
    mask = AscendC::Reg::UpdateMask<T>(outputLength);
    ```
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 1021]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">VF Function Name</td><td colspan="4" align="center">CopyWithOffsetVF</td></tr>
  </table>

**Scenario 2: Using PostUpdate Mode to Represent Inter-Iteration Offset**
- Example Functionality:
  Load/store using POST_MODE_UPDATE mode to implement contiguous data load/store.
- Example Implementation:
  - LoadAlign/StoreAlign template parameter postMode = PostLiteral::POST_MODE_UPDATE
  - LoadAlign/StoreAlign input parameter postUpdateStride = 128, i.e., VL/sizeof(half). Using LoadAlign as an example:
    - UB starting address for each iteration is srcAddr
    - After LoadAlign execution, srcAddr automatically updates to srcAddr + postUpdateStride.
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">VF Function Name</td><td colspan="4" align="center">CopyWithPostUpdateVF</td></tr>
  </table>

**Scenario 3: Using Address Register (AddrReg) to Represent Inter-Iteration Offset**
- Example Functionality:
  Load/store using AddrReg (address register) to implement contiguous data load/store.
- Example Implementation:
  - Initialize address register, indicating aReg increments by stride0 when i-axis loop completes
    ```cpp
    uint32_t stride0 = AscendC::GetVecLen() / sizeof(T);
    AddrReg aReg = AscendC::Reg::CreateAddrReg<T>(i, stride0);
    ```
  - LoadAlign/StoreAlign input parameter offset = aReg. Using LoadAlign as an example:
    - UB starting address for each iteration is srcAddr + aReg
    - After iteration, aReg automatically updates to aReg + stride0.
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">VF Function Name</td><td colspan="4" align="center">CopyWithAddrRegVF</td></tr>
  </table>

**Scenario 4: Non-Contiguous Transfer in DataBlock Units**
- Example Functionality:
  Load/store using DataBlock (32-byte) transfer mode. During load, adjacent DataBlocks are separated by 2 DataBlocks, i.e., transfer 32B, skip 32B; during store, adjacent DataBlocks are separated by 1 DataBlock, equivalent to contiguous transfer.
- Example Implementation:
  - LoadAlign/StoreAlign template parameter dataMode = DataCopyMode::DATA_BLOCK_COPY
  - LoadAlign input parameter dataBlockStride is set to 2, StoreAlign input parameter dataBlockStride is set to 1
  - Each repeat processes 256B, i.e., 8 DataBlocks, so LoadAlign input parameter repeatStride is set to 16, StoreAlign input parameter repeatStride is set to 8
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 512]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">VF Function Name</td><td colspan="4" align="center">CopyWithAddrRegVF</td></tr>
  </table>

**Scenario 5: Broadcast Mode Load**
- Example Functionality:
  Load using broadcast mode. Each iteration loads the first element at the UB starting address and broadcasts it to all elements in regTensor.
- Example Implementation:
  - LoadAlign template parameter dist = LoadDist::DIST_BRC_B16
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">VF Function Name</td><td colspan="4" align="center">CopyInBroadcastVF</td></tr>
  </table>

**Scenario 6: Upsample Mode Load**
- Example Functionality:
  Load using upsample mode. Each iteration loads VL/2 data, with each input element repeated twice.
- Example Implementation:
  - LoadAlign template parameter dist = LoadDist::DIST_US_B16
- Example Specifications:
  <table>
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="3" align="center">AIV Example</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[1, 2048]</td><td align="center">half</td></tr>
  <tr><td rowspan="1" align="center">VF Function Name</td><td colspan="4" align="center">CopyInUpsampleVF</td></tr>
  </table>

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
  SCENARIO=1                                                                    # Execute scenario 1
  mkdir -p build && cd build;                                                   # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # Build project (default npu mode)
  python3 ../scripts/gen_data.py -scenarioNum $SCENARIO                         # Generate test input data
  ./demo                                                                        # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DSCENARIO_NUM=1 -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DSCENARIO_NUM=1 -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build mode or scenario, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build Option Description

| Option | Available Values | Description |
| | ----------------| -----------------------------| --------------------------------------------------------------------------------------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
| `SCENARIO_NUM` | `1`, `2`, `3`, `4`, `5`, `6` | Scenario number, see Overview for details |

- Execution Result
  When the execution result shows the following, it indicates successful precision comparison.
  ```bash
  test pass!
  ```