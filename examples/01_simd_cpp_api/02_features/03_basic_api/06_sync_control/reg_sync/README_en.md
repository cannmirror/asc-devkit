# reg_sync Sample

## Overview
This sample implements synchronization mechanisms for UB (Unified Buffer) read or write operations based on the Reg programming interface, supporting multiple scenarios selectable via environment variable.
    <table>
   	 	<tr>
 	 		<td>scenarioNum</td>
 	 		<td>Synchronization Scenario</td>
 	 	</tr>
 	 	<tr>
 	 		<td>1</td>
 	 		<td>Register Ordering Preservation</td>
 	 	</tr>
 	 	<tr>
 	 		<td>2</td>
 	 		<td>LocalMemBar (Write-Read Dependency)</td>
 	 	</tr>
 	 </table>


## Supported Products
- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── reg_sync
│   ├── scripts
│   │   ├── gen_data.py                // Input data and golden data generation script
│   ├── CMakeLists.txt                 // Build configuration file
│   ├── data_utils.h                   // Data read/write functions
│   ├── reg_sync.asc                   // AscendC operator implementation & sample invocation
│   └── README.md                      // Sample introduction
```

## Sample Description

### Scenario 1: Register Ordering Preservation

**Sample Function**: Performs in-place exp computation on input vector x, writing results back to the same address.

**Sample Specifications**:
<table>
<tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="3" align="center">AIV Sample</td></tr>
<tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
<tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">float</td></tr>
<tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[1, 1024]</td><td align="center">float</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="3" align="center">reg_sync</td></tr>
</table>

- Sample Implementation  
In RegSyncVf function:
1. LoadAlign and StoreAlign operate on the same address
2. Read and write to the same register: read with LoadAlign, then write with StoreAlign
3. Hardware automatically ensures StoreAlign waits for LoadAlign to complete; no LocalMemBar needed
- Invocation Implementation  
  Use kernel invocation operator `<<<>>>` to call the kernel function, launching 1 core.

### Scenario 2: LocalMemBar (Write-Read Dependency)

**Sample Function**: Computes the sum of absolute values of a vector: `sum = Σ|x[i]|`

**Sample Specifications**:
<table>
<tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="3" align="center">AIV Sample</td></tr>
<tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
<tr><td align="center">x</td><td align="center">[1, 8]</td><td align="center">float</td></tr>
<tr><td rowspan="1" align="center">Sample Output</td><td align="center">sum</td><td align="center">[1, 1]</td><td align="center">float</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="3" align="center">reg_sync</td></tr>
</table>

- Sample Implementation  
In UbSyncVf function:
1. Read input data from UB to RegTensor
2. Compute absolute value to get |x|
3. Write |x| to UB temporary buffer
4. Call LocalMemBar to wait for write completion (read-wait-write synchronization)
5. Read |x| from UB temporary buffer
6. Accumulate sum and output result
- Invocation Implementation  
  Use kernel invocation operator `<<<>>>` to call the kernel function, launching 1 core.

**LocalMemBar Necessity Explanation**:
Step 3 writes |x| to UB, and step 5 needs to read from the same UB address. There is a read-after-write (RAW) dependency; the read operation must wait for the write operation to complete. Without LocalMemBar, step 5 may read stale data that hasn't been updated. LocalMemBar ensures UB writes complete before executing read operations.


## Build and Run
Execute the following steps in the sample root directory to build and run the operator.
- Configure Environment Variables  
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
  - Default path, root user installed CANN package

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN package

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN package

    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Sample Execution

  ```bash
  SCENARIO=1
  mkdir -p build && cd build;                                                   # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j; # Build project (default npu mode)
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO                         # Generate test input data
  ./demo                                                                        # Execute the compiled program, run the sample
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Examples:

  ```bash
  SCENARIO=1
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes or scenarios, clean cmake cache by running `rm CMakeCache.txt` in the build directory and re-run cmake.

- Build Options Description

| Option　　　　　 | Available Values　　　　　　　　　| Description　　　　　　　　　　　　　　　　　　　　　　　|
| ----------------| -----------------------------| ---------------------------------------------------|
| `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation　　　　　　　|
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
| `SCENARIO_NUM` | `1`, `2`　　　　　　　　　　| Scenario number: 1=Register ordering preservation, 2=LocalMemBar　　　　　　　|

- Execution Result  
  The following execution result indicates successful precision comparison.

  ```bash
  test pass!
  ```