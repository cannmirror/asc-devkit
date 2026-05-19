# CrossCoreSetFlag and CrossCoreWaitFlag Cross-Core Synchronization Example

## Overview
This example first introduces the three synchronization modes supported by the cross-core synchronization interfaces CrossCoreSetFlag and CrossCoreWaitFlag (see table below), then demonstrates the specific usage of these three synchronization modes in two actual scenarios: pure Vector computation and Cube-Vector fusion computation.
<table border="1" align="center">
  <tr bgcolor="lightgray">
    <td>Synchronization Control Mode</td>
    <td align="center">Description</td>
  </tr>
  <tr>
    <td rowspan="2">mode 0</td>
    <td>For AIC scenarios, synchronizes all AIC cores. Instructions after CrossCoreWaitFlag will not execute until all AIC cores have executed CrossCoreSetFlag.</td>
  </tr>
  <tr>
    <td>For AIV scenarios, synchronizes all AIV cores. Instructions after CrossCoreWaitFlag will not execute until all AIV cores have executed CrossCoreSetFlag.</td>
  </tr>
  <tr>
    <td>mode 1</td>
    <td>Within a single AI Core, synchronization control between AIV cores. Instructions after CrossCoreWaitFlag will execute only when both AIV cores have executed CrossCoreSetFlag.</td>
  </tr>
  <tr>
    <td rowspan="2">mode 2</td>
    <td>After an AIC core executes CrossCoreSetFlag, instructions after CrossCoreWaitFlag on both AIVs will continue executing.</td>
  </tr>
  <tr>
    <td>After both AIVs execute CrossCoreSetFlag, instructions after CrossCoreWaitFlag on the AIC can execute.</td>
  </tr>
</table>

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── cross_core_set_wait_flag
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data matches golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   ├── cross_core_set_wait_flag.h   // Ascend C example implementation
│   └── cross_core_set_wait_flag.asc // Invocation example and result verification
```

## Example Description
<table border="1" style="text-align: center;">
  <tr>
    <td>SCENARIO_NUM Value</td>
    <td>Scenario</td>
    <td>Synchronization Mode Used</td>
  </tr>
  <tr>
    <td>0</td>
    <td>Pure Vector Computation Scenario (16 AIVs)</td>
    <td>mode 0 (AIV full-core synchronization)</td>
  </tr>
  <tr>
    <td>1</td>
    <td>Pure Vector Computation Scenario (2 AIVs)</td>
    <td>mode 1</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Cube-Vector Fusion Computation Scenario</td>
    <td>mode 2 (AIC waits for AIV), mode 2 (AIV waits for AIC), mode 0 (AIC full-core synchronization)</td>
  </tr>
</table>
This example controls execution branches via SCENARIO_NUM, where different values correspond to different scenarios and synchronization modes. As shown in the table above, when SCENARIO_NUM takes different values, it demonstrates the specific usage of three synchronization modes in pure Vector computation scenarios and Cube-Vector fusion computation scenarios.

### Computation Formula and Example Specifications

#### SCENARIO_NUM=0 (Pure Vector Computation Scenario, Mode 0)
- Computation Formula:  
  $$
  z = \sum_{i=0}^{15} (x \times i)
  $$
  - x is the input vector, all values are 1
  - i is the BlockIdx of each AIV (range 0-15)
  - z is the accumulated result of all AIV computations

- Example Specifications:
  <table border="1" align="center">
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center">CrossCoreSetFlagMode0</td></tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[32]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[32]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">cross_core_set_wait_flag_custom</td></tr>
  </table>

#### SCENARIO_NUM=1 (Pure Vector Computation Scenario, Mode 1)
- Computation Formula:  
  $$
  z = (x \times 2) + (x \times 3)
  $$
  - x is the input vector, all values are 1
  - Only AIVs with BlockIdx=2 and 3 participate in computation
  - z is the accumulated result of these two AIV computations

- Example Specifications:
  <table border="1" align="center">
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="4" align="center">CrossCoreSetFlagMode1</td></tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[32]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">z</td><td align="center">[32]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">cross_core_set_wait_flag_custom</td></tr>
  </table>

#### SCENARIO_NUM=2 (Cube-Vector Fusion Computation Scenario)
- Computation Formula:  
  $$
  C = \text{LeakyRelu}(Cast(A) \times Cast(B))
  $$
  - A is the left matrix with shape [M, K], data type uint8
  - B is the right matrix with shape [K, N], data type uint8
  - First convert A and B data types from uint8 to half
  - Then execute matrix multiplication: A × B
  - Finally execute LeakyRelu operation on the result
  - C is the final result with shape [M, N], data type float32

- Example Specifications:
  <table border="1" align="center">
  <tr><td rowspan="1" align="center">Example Type (OpType)</td><td colspan="5" align="center">CrossCoreSetFlagMode2</td></tr>
  <tr><td rowspan="3" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
  <tr><td align="center">a</td><td align="center">[32, 32]</td><td align="center">uint8</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td align="center">b</td><td align="center">[32, 64]</td><td align="center">uint8</td><td align="center">ND</td><td align="center">false</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">c</td><td align="center">[32, 64]</td><td align="center">float32</td><td align="center">ND</td><td align="center">-</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">mmad_custom</td></tr>
  </table>
## Example Implementation
### 1. Cube-Vector Fusion Computation Scenario
#### 1.1 Overall Logic
<p align="center">
  <img src="img/融合场景_示意图.png" width="100%">
   </p>
<p align="center">
Figure 1: Schematic diagram of overall computation logic for Cube-Vector fusion computation scenario
</p>

This example focuses on the fusion operator (configured with `__mix__(1,2)`) scenario, where each AI Core has 1 AIC and 2 AIVs. The logical core count is configured as numBlocks = 8, corresponding to 8 AICs and 16 AIVs. As shown in Figure 1, the overall computation logic is divided into three core phases: precision conversion phase, blocked matrix multiplication and atomic accumulation phase, and LeakyRelu operation and result write-back phase.
#### 1.2 Precision Conversion Phase (Mode 2, AIC waits for AIV within a single AI Core)
Since the left and right matrix data types on GM are uint8, which do not meet the mmad instruction's input data type requirements, the data on GM must first be moved to AIV for precision conversion before blocked matrix multiplication can be performed on AIC. Therefore,
within each AI Core, one AIC must wait for the other 2 AIVs in that AI Core to complete data precision conversion before starting the blocked matrix multiplication computation.
Specifically: The left matrix (A matrix) data in GM is split into 8 parts along the K axis and distributed to AIVs with even BlockIdx for uint8 to half precision conversion; the right matrix (B matrix) data in GM is split into 8 parts along the K axis and distributed to AIVs with odd BlockIdx for uint8 to half precision conversion. As shown in Figure 2, according to the above description, cross-core synchronization mode 2 is required.
The code segment corresponding to the above description is as follows:

        if (blockIdx % 2 == 0) {
            ...
            AscendC::Cast(castALocal, aLocal, AscendC::RoundMode::CAST_NONE, A_BLOCKS_LENGTH);
            ...
            AscendC::DataCopy(AVectorGM, castALocal, A_BLOCKS_LENGTH);
        } else {
            ...
            AscendC::Cast(castBLocal, bLocal, AscendC::RoundMode::CAST_NONE, B_BLOCKS_LENGTH);
            ...
            AscendC::DataCopy(BVectorGM, castBLocal, B_BLOCKS_LENGTH);
        }
        // Mode 2, within each AICore, one AIC waits for 2 AIVs
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG);

<p align="center">
  <img src="img/融合场景_精度转换阶段.png" width="100%">
   </p>
<p align="center">
Figure 2: Precision conversion phase, mode 2 schematic diagram
</p>

#### 1.3 Blocked Matrix Multiplication and Atomic Accumulation Phase (Mode 0, AIC full-core synchronization)
After each AIC executes blocked matrix multiplication computation, it enables the atomic accumulation mechanism, moves computation results to the same GM region, and accumulates the blocked matrix multiplication results from 8 AICs on GM to obtain the complete C matrix. To obtain the correct C matrix, it is necessary to wait for all 8 AICs to complete blocked matrix multiplication computation and move results to GM via FixPipe. As shown in Figure 3, according to the above description, cross-core synchronization mode 0 (AIC full-core synchronization) is required. The code segment corresponding to the above description is as follows:
$$
C = \sum_{i=1}^{8} A_i \cdot B_i
$$

        // Mode 2, within each AICore, AIC waits for 2 AIVs
        AscendC::CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);

        CopyIn(a1Local, b1Local);
        SplitA(a1Local, a2Local);
        SplitBTranspose(b1Local, b2Local);
        Compute(a2Local, b2Local, c1Local);
        CopyOut(c1Local);
        // Mode 0, 8 AICs in 8 AICores synchronize
        AscendC::CrossCoreSetFlag<0, PIPE_FIX>(SYNC_AIC_FLAG);  
        AscendC::CrossCoreWaitFlag(SYNC_AIC_FLAG);  

        // Mode 2, within each AICore, 2 AIVs wait for AIC
        AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);  

<p align="center">
  <img src="img/融合场景_分块矩阵乘与原子累加阶段.png" width="100%">
   </p>
<p align="center">
Figure 3: Blocked matrix multiplication and atomic accumulation phase, mode 0 schematic diagram
</p>

#### 1.4 LeakyRelu and Result Write-back Phase (Mode 2, AIV waits for AIC)
Within each AI Core, 2 AIVs must wait for the AIC to complete blocked matrix multiplication and atomic accumulation operations before performing LeakyRelu operation on C matrix blocks.
Specifically, the accumulated C matrix is split into 16 parts along the M axis and distributed to 16 AIVs for LeakyRelu operation. As shown in Figure 4, according to the above description, cross-core synchronization mode 2 (2 AIVs wait for 1 AIC within a single AI Core) is required. The code segment corresponding to the above description is as follows:

        // Mode 2, within each AICore, 2 AIVs wait for AIC
        AscendC::CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);

        // Perform LeakyRelu operation
        float alpha = 0.001;
        ...
        AscendC::LeakyRelu(reluCLocal, cLocal, alpha, C_AIV_BLOCKS_LENGTH);
        ...

<p align="center">
  <img src="img/融合场景_LeakyRelu运算与结果回写阶段.png" width="100%">
   </p>
<p align="center">
Figure 4: LeakyRelu operation and result write-back phase, mode 2 schematic diagram
</p>

### 2. Pure Vector Computation Scenario
#### 2.1 Comparison of Mode 0 and Mode 1
This example sets NUM\_BLOCKS to 8 (8 AI Cores), with an AIC:AIV ratio of 2 within each AI Core, meaning this example runs a total of 8 AICs and 16 AIVs, with AIV BlockIdx ranging from 0\~15.
As shown in Figure 5 below, the computation logic for mode 0 and mode 1 in this example is nearly identical, with the only difference being the number of AIVs participating in synchronization: for mode 0, all 16 AIVs participate in synchronization; for mode 1, only 2 AIVs (BlockIdx=2 and 3) in the second AI Core participate in synchronization. Therefore, the next section will describe the overall logic of mode 0 in detail, and mode 1 will not be described in detail.
<p align="center">
  <img src="img/纯aiv_模式1和模式0的区别.png" width="100%">
</p>
<p align="center">
Figure 5: Pure AIV scenario, mode 0 computation logic schematic diagram
</p>

#### 2.2 Overall Logic of Mode 0
The GM used in this example is divided into 2 blocks: one for storing input data (initialDataGm), and one for storing the accumulated results of all AIVs (atomicResultGm).
As shown in Figure 6 below, the overall logic of mode 0 consists of the following steps:
(1) Each AIV moves data (all 1s) from initialDataGm to UB, which is a PIPE\_MTE2 pipeline operation.

(2) Each AIV performs vector computation on UB data: multiplies by the BlockIdx corresponding to each core via Muls instruction, which is a PIPE\_V pipeline operation.

(3) The code segment corresponding to step 3 is as follows. atomicResultGm is used to store the accumulated results after all 16 AIVs have completed moving. By calling the CrossCoreSetFlag and CrossCoreWaitFlag interfaces, synchronization control is implemented: instructions after CrossCoreWaitFlag can only execute after all 16 AIVs have completed the PIPE_MTE3 move instruction.

         // Enable atomic accumulation for UB to GM move: data moved to atomicResult is accumulated with original value and overwrites the original value
        AscendC::SetAtomicAdd<float>(); 
        // DataCopy is a PIPE_MTE3 pipeline operation
        AscendC::DataCopy(atomicResultGm, xLocal, this->blockLength);   
        // When this AIV completes the preceding PIPE_MTE3 (DataCopy) pipeline operation, notify other AIV cores that this AIV has completed
        AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(0);  
        // Block this AIV from continuing to execute instructions until all other AIVs have completed PIPE_MTE3 pipeline operations, then unblock and continue execution.
        AscendC::CrossCoreWaitFlag(0); 

After the above synchronization completes, atomicResultGm now contains the accumulated value of vector computation results from 16 AIVs. At this point, data is moved from atomicResultGm to a specific AIV, and DumpTensor is used to print the data to verify whether the accumulated result meets expectations. Finally, the result from that AIV is moved out to atomicResultGm.
If the synchronization in the previous step is inserted incorrectly, the data moved from atomicResultGm to the AIV may be the accumulated value of vector computation results from only some AIVs, resulting in inaccurate printed and written results to atomicResultGm.

        if (AscendC::GetBlockIdx() == 0) {
            AscendC::DataCopy(yLocal, atomicResultGm, this->blockLength);   // PIPE_MTE2
            AscendC::printf("============== In PrintTensor Process AIV %d ==============", AscendC::GetBlockIdx());
            AscendC::DumpTensor(yLocal, AscendC::GetBlockIdx(), this->blockLength);
            AscendC::DataCopy(atomicResultGm, yLocal, this->blockLength);
            return;
        }
<p align="center">
  <img src="img/纯aiv_模式0示意图.png" width="100%">
</p>
<p align="center">
Figure 6: Pure AIV scenario, mode 0 computation logic schematic diagram
</p>

### 3. Notes
#### 3.1 Cube-Vector Fusion Computation Scenario
(1) In the Cube-Vector fusion computation scenario, the fusion operator (configured with `__mix__(1,2)`) requires using ASCEND_IS_AIV/ASCEND_IS_AIC to isolate AIV and AIC core code.
```
KernelMmad op;
if ASCEND_IS_AIC {
    op.InitAIC(A, B, c);
    op.ProcessAIC();
} 
if ASCEND_IS_AIV {
    op.InitAIV(a, b, A, B, c);
    op.ProcessAIV();
}
```
(2) GetBlockIdx (get the index of the current core) has different value ranges for AIC and AIV. Its values are related to the logical core count set by the operator and the ratio of AIC to AIV in an AI Core. In this example, NUM_BLOCKS=8 is set, and the ratio of AIC to AIV is 1:2, so the value ranges of GetBlockIdx for AIC and AIV are 0-7 and 0-15 respectively.

(3) The example uses the static tensor programming paradigm and requires manual insertion of intra-core synchronization. Additionally, in the static tensor programming mode, developers must manually call the InitSocState() interface to initialize global state registers.

#### 3.2 Pure Vector Computation Scenario
(1) When using CrossCoreSetFlag and CrossCoreWaitFlag cross-core synchronization interfaces, even in pure Vector computation scenarios, the kernel function cannot use the `__vector__` modifier.
In this example, the kernel function uses the `__mix__(1,2)` modifier, but in pure Vector scenarios, since only vector computation is executed, the ASCEND_IS_AIV macro must be used to ensure the program runs only on AIV cores, otherwise the program will hang.
```
if ASCEND_IS_AIV {
        op.Init(x, z, dataLength);
        op.Process();
}
```
 (2) Mode 1 requires that the 2 AIVs participating in synchronization must belong to the same AI Core, otherwise the program will hang. In this example, the two AIVs participating in synchronization have GetBlockIdx=2 and 3, both belonging to the 2nd AI Core (index starting from 1); if GetBlockIdx is changed to 3 and 4 (belonging to two different AI Cores), the program will hang.

 (3) The example uses the static tensor programming paradigm and requires manual insertion of intra-core synchronization. Additionally, in the static tensor programming mode, developers must manually call the InitSocState() interface to initialize global state registers.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure Environment Variables  
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, CANN package installed by root user
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, CANN package installed
    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Example Execution
  ```bash
  SCENARIO_NUM=0  # Set scenario number (values: 0, 1, 2)
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenarioNum $SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled executable program
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin  # Verify output correctness
  ```

  To use CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:
  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU simulation mode
  ```
  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory and re-running cmake.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `0` (default), `1`, `2` | Scenario number: 0 (Pure Vector mode 0), 1 (Pure Vector mode 1), 2 (Cube+Vector fusion) |

- Execution Result

  The following output indicates successful precision comparison.
  ```bash
  test pass!
  ```