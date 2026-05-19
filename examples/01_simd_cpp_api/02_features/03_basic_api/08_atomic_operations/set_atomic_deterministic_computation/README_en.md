# Deterministic Computation with Data Movement Accompanying Atomic Operations Sample

## Overview

This sample first introduces the necessity and specific implementation approach for deterministic computation in scenarios involving data movement with accompanying atomic operations. It then describes how to apply this approach in single-AIV core, multi-AIV core, and multi-AIC core scenarios.

## Supported Products

- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── set_atomic_determine_compute
│   ├── img                      // Image resources directory for README
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   ├── set_atomic_add_multi_aic.h    // Multi-AIC core synchronization implementation
│   ├── set_atomic_add_multi_aiv.h    // Multi-AIV core synchronization implementation
│   ├── set_atomic_add_single_aiv.h   // Single-AIV core synchronization implementation
│   ├── set_atomic_determine_compute.asc  // Ascend C sample implementation & invocation sample
```

## 1. Deterministic Computation

### 1.1 Necessity of Deterministic Computation

Floating-point addition does not satisfy the mathematical properties of commutativity and associativity. Different orders of addition can lead to different computational results, which can be verified with the following data:

```python
x = 1.0
y = 1e16
z = -1e16

# Different addition orders produce different results
left = (x + y) + z    # Add x and y first, then add z
right = x + (y + z)   # Add y and z first, then add x
```

Due to floating-point precision limitations, the results of `(1.0 + 1e16) + (-1e16)` and `1.0 + (1e16 + (-1e16))` are different:
- The former: `1.0 + 1e16` loses precision, resulting in a value close to `1e16`. Adding `-1e16` gives `0.0`.
- The latter: `1e16 + (-1e16)` exactly equals `0.0`. Adding `1.0` gives `1.0`.

Therefore, in parallel computing environments, synchronization mechanisms must be used to ensure a fixed order of addition, guaranteeing deterministic results. Deterministic computation refers to a computational process that always produces exactly the same output results under the same input conditions, regardless of the number of executions or the execution environment. Deterministic computation provides guarantees for system stability and experimental verifiability.

### 1.2 Overview of Deterministic Computation

To introduce the problem of non-deterministic computation in atomic operation scenarios, we construct the following common deterministic computation scenario: First, GM is initialized by moving a single set of floating-point data; then, an atomic accumulation operation is initiated; finally, multiple data movements are performed to accumulate multiple sets of floating-point data on GM. The specific pseudocode is as follows:

```
①Move data data0 to GM;    // Data movement, overwrites random values in GM, expected GM data is data0 
②SetAtomicAdd();         // Enable atomic accumulation, subsequent movements from UB/L0C/L1 to GM all perform atomic accumulation 
③Move data1 to GM;    // Data movement with accompanying atomic operation, expected GM data is data0 + data1 
④Move data2 to GM;    // Data movement with accompanying atomic operation, expected GM data is data0 + data1 + data2 
⑤Move data3 to GM;    // Data movement with accompanying atomic operation, expected GM data is data0 + data1 + data2 + data3
```

As shown in the figure below, the developer's expected result: The order of instruction dispatch strictly corresponds to the actual instruction execution order. No matter how many times this code is executed, the final GM data is always data0 + data1 + data2 + data3, achieving deterministic computation.

![Deterministic Computation Scenario](img/确定性计算场景，GM上数据变化过程.png)

Figure 1: Deterministic computation scenario, data change process on GM

However, in reality, if the developer does not intervene, the execution order of these instructions may change each time the program runs, ultimately causing the GM data to be inconsistent with the expected result. The following are two possible instruction execution orders and their corresponding execution flows.

#### 1.2.1 Non-deterministic Computation, Result 1

![Non-deterministic Computation Scenario 1](img/非确定性计算场景1，GM上数据变化过程.png)

Figure 2: Non-deterministic computation scenario 1, data change process on GM

As shown in the figure, the instruction execution flow in this scenario is as follows:
1. Initial state, GM data is: random value;
2. Move data0 to GM, GM data is initialized to: data0;
3. Execute SetAtomicAdd, enabling atomic accumulation for subsequent data movement instructions, GM data is: data0;
4. Three data movement instructions with accompanying atomic operations are executed out of order, with actual execution order being "move data2 → move data3 → move data1". Final GM data is: data0 + data2 + data3 + data1.

**Cause of non-deterministic computation 1**:
Data movement instructions with accompanying atomic operations are executed out of order. Since floating-point addition does not satisfy associativity, i.e., (a+b)+c != a+(b+c), the final GM data data0 + data2 + data3 + data1 deviates from the expected data0 + data1 + data2 + data3.

The prerequisites for data movement instructions with accompanying atomic operations being out of order to cause deviation in the final result are as follows:
- The atomic operation type is atomic accumulation (maximum and minimum operations satisfy associativity)
- The atomic operation data type is floating-point (integer addition satisfies associativity)
- There are 3 or more data movement instructions with accompanying atomic operations (floating-point addition satisfies commutativity)

#### 1.2.2 Non-deterministic Computation, Result 2

![Non-deterministic Computation Scenario 2](img/非确定性计算场景2，GM上数据变化过程.png)

Figure 3: Non-deterministic computation scenario 2, data change process on GM

As shown in the figure, the instruction execution flow in this scenario is as follows:
1. Initial state, GM data is: random value;
2. Execute SetAtomicAdd, enabling atomic accumulation for subsequent data movement instructions, GM data is: random value;
3. Execute two data movement instructions with accompanying atomic operations sequentially, in the order "move data1 → move data2". GM data is: random value + data1 + data2;
4. Move data0 to GM, the accumulated result on GM is overwritten by data0. GM data is: data0;
5. Finally, execute the data3 movement. Final GM data is: data0 + data3.

**Cause of non-deterministic computation 2**:
Out-of-order execution between ordinary data movement instructions before enabling atomic accumulation and data movement instructions with atomic operations enabled causes the data on GM that has already undergone atomic operations to be incorrectly overwritten by data0, resulting in non-deterministic computation results.

This type of out-of-order execution causing result deviation does not require any prerequisites. Developers do not need to distinguish between atomic operation types, atomic operation data types, or consider whether the number of data movement instructions with accompanying atomic operations reaches 3 or more.

### 1.3 Deterministic Computation Implementation Approach

Based on the two root causes of non-deterministic computation, the following describes the implementation approach for deterministic computation from the perspective of solving both issues. The core idea is to insert appropriate synchronization between instructions so that relevant instructions are executed in the expected determined order each time the program runs, ultimately ensuring that the output results are the same for every program execution. Specifically, this includes the following two aspects:

- Insert synchronization between data movement instructions before enabling atomic accumulation and instructions enabling atomic operations
  As shown in the pseudocode below, inserting synchronization between instructions ① and ② ensures that the initial value of GM meets expectations before starting atomic operations.
- Synchronization between multiple data movement instructions after enabling atomic accumulation
  Inserting synchronization between instructions ③ and ④, and between ④ and ⑤ ensures that the order of floating-point addition meets expectations.

Synchronization is not required between the instruction enabling atomic operations and subsequent data movement instructions.

#### 1.3.1 Intra-core Synchronization Implementation

```cpp
// The entire atomic accumulation is executed within the same core, controlling the execution order of 5 instructions as "①→②→③→④→⑤" 
①Move data data0 to GM;    // Data movement, overwrites random values in GM, expected GM data is data0 
Intra-core synchronization 
②SetAtomicAdd();         // Enable atomic accumulation, subsequent movements from UB/L0C/L1 to GM all perform atomic accumulation 
// No synchronization needed between instructions ② and ③ 
③Move data1 to GM;    // Data movement after enabling atomic accumulation, expected GM data is data0 + data1 
Intra-core synchronization 
④Move data2 to GM;    // Data movement after enabling atomic accumulation, expected GM data is data0 + data1 + data2 
Intra-core synchronization 
⑤Move data3 to GM;    // Data movement after enabling atomic accumulation, expected GM data is data0 + data1 + data2 + data3
```

The pipeline types for data movement instructions and instructions enabling atomic operations are shown in the table below. When the above instructions are executed within the same core, developers can insert [single-pipeline synchronization](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0271.html) or [multi-pipeline synchronization](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0270.html) as needed. Refer to the SCENARIO_NUM=1 branch in the sample.

<table border="1" style="text-align: left;">
  <tr>
    <th style="padding: 8px;">Instruction Name</th>
    <th style="padding: 8px;">Pipeline Type</th>
  </tr>
  <tr>
    <td style="padding: 8px;">DataCopy</td>
    <td style="padding: 8px;">PIPE_MTE3</td>
  </tr>
  <tr>
    <td style="padding: 8px;">Fixpipe</td>
    <td style="padding: 8px;">PIPE_FIX</td>
  </tr>
  <tr>
    <td style="padding: 8px;">SetAtomicAdd/SetAtomicMax/SetAtomicMin</td>
    <td style="padding: 8px;">PIPE_S</td>
  </tr>
</table>

#### 1.3.2 Inter-core Synchronization Implementation

As shown in the pseudocode below, when the above instructions are executed in different cores, the intra-core synchronization needs to be replaced with inter-core synchronization.

```cpp
// The entire atomic accumulation is executed in 4 different cores, controlling the execution order of 4 cores as "core 0→core 1→core 2→core 3" 
if (GetBlockIdx == 0) { 
   Move data data0 to GM; 
   Inter-core synchronization 
} else if (GetBlockIdx == 1) { 
   Inter-core synchronization     
   SetAtomicAdd();          
   Move data1 to GM; 
   Inter-core synchronization    
} else if (GetBlockIdx == 2) { 
   Inter-core synchronization 
   SetAtomicAdd();          
   Move data2 to GM;   
   Inter-core synchronization  
} else if (GetBlockIdx == 3) { 
   Inter-core synchronization 
   SetAtomicAdd();          
   Move data3 to GM;    
}
```

Since hardware synchronization interfaces for controlling execution order between different cores are not currently provided, inter-core synchronization in deterministic computation scenarios must be implemented through software methods. The software synchronization approaches differ for three scenarios: pure Vector samples, pure Cube samples, and Mix (containing both Vector and Cube computation) samples, as shown in the table below.

<table border="1" style="text-align: left;">
  <tr>
    <th style="padding: 8px;">Sample Type</th>
    <th style="padding: 8px;">Software Synchronization Approach</th>
    <th style="padding: 8px;">Description</th>
  </tr>
  <tr>
    <td style="padding: 8px;" rowspan="2">Pure Vector Sample</td>
    <td style="padding: 8px;">Approach 1: Multiple pairs of IBSet and IBWait interfaces can be combined to achieve synchronization between multiple AIVs. Refer to the SCENARIO_NUM=2 branch in the sample.</td>
    <td style="padding: 8px;">Approach 1 supports specifying partial AIVs to participate in synchronization and can control the execution order of each AIV.</td>
  </tr>
  <tr>
    <td style="padding: 8px;">Approach 2: Use the three interfaces InitDetermineComputeWorkspace, NotifyNextBlock, and WaitPreBlock together to ensure all AIV cores execute in ascending blockIdx order. Refer to the SCENARIO_NUM=4 branch in the sample (to be supported).</td>
    <td style="padding: 8px;">Approach 2 requires all AIVs to participate in synchronization, and the execution order is fixed to ascending blockIdx.</td>
  </tr>
  <tr>
    <td style="padding: 8px;">Pure Cube Sample</td>
    <td style="padding: 8px;">Inter-core synchronization is achieved through semaphores in GM. First, establish synchronization between a pair of cores, then extend to synchronization between multiple cores. Refer to the SCENARIO_NUM=3 branch in the sample.</td>
    <td style="padding: 8px;">When accessing GM through the Scalar unit, consider data consistency issues between multiple cores.<br>The "core" here can be either an AIV or an AIC.</td>
  </tr>
</table>

The figure below shows how to perform inter-core synchronization between two cores using semaphores in GM:

![Inter-core Software Synchronization Flowchart](img/一对核之间软件同步方案流程图.png)

Figure 4: Inter-core software synchronization flowchart between a pair of cores

- After the previous core completes data movement or enables atomic operations, it writes value 1 to a semaphore in inter-core shared GM through the Scalar unit, indicating that its task is complete. Intra-core synchronization also needs to be inserted in the previous core:
  - When there are multiple data movement instructions in the previous core, intra-core synchronization 1 needs to be inserted between them.
  - Before the Scalar unit writes data to GM, all previous data movement instructions must have completed execution, so intra-core synchronization 2 also needs to be inserted between them.
- Before executing data movement tasks, the current core continuously reads the semaphore value through the Scalar unit. If the semaphore does not equal 1, the current core enters a blocking wait state; when the semaphore equals 1 is detected, the current core unblocks and begins executing its own data movement or atomic operation. To ensure the current core does not execute data movement instructions before the semaphore equals 1, intra-core synchronization 3 needs to be inserted before the data movement instructions.

#### 1.3.3 Methods for Scalar Unit to Access Semaphores on GM

There are two methods for the Scalar unit to access semaphores on GM:

1. **Access via DCache**
   Use the GlobalTensor member functions GetValue and SetValue. In this case, developers need to manually call the [DataCacheCleanAndInvalid](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_0177.html) interface to ensure data consistency between multiple cores.

2. **Access without DCache**
   Use [WriteGmByPassDCache](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/ascendcopapi/atlasascendc_api_07_00089.html) and ReadGmByPassDCache. This method ensures data consistency between multiple cores without additional operations.

Performance difference between the two approaches: Bypassing DCache results in poorer performance, but if the amount of GM data read/written is small, consider using the method that bypasses DCache.

Inter-core synchronization approaches also need to be used in conjunction with intra-core synchronization. The functions of the three intra-core synchronizations are described below:

- **Intra-core synchronization 1 (optional)**: When there are multiple data movement instructions within a core, this synchronization ensures that each data movement operation executes in strict order.
- **Intra-core synchronization 2 (required)**: Wait for all tasks in the previous core to complete before allowing the Scalar unit to write 1 to the global memory semaphore.
- **Intra-core synchronization 3 (required)**: Wait for the Scalar unit to detect that the semaphore has been updated to 1 before the current core starts executing subsequent tasks.

## 2. Sample Description

### 2.1 Scenario Configuration Description

<table border="1" style="text-align: left;">
  <tr>
    <td>SCENARIO_NUM Value</td>
    <td>Business Scenario</td>
    <td>Kernel Function</td>
    <td>Synchronization Mode Used</td>
  </tr>
  <tr>
    <td>1</td>
    <td>Deterministic computation within a single AIV core</td>
    <td>set_atomic_add_single_aiv_custom</td>
    <td>PipeBarrier ensures order</td>
  </tr>
  <tr>
    <td>2</td>
    <td>Deterministic computation between multiple AIV cores</td>
    <td>set_atomic_add_multi_aiv_custom</td>
    <td>SetFlag/WaitFlag AIV inter-core synchronization</td>
  </tr>
  <tr>
    <td>3</td>
    <td>Deterministic computation between multiple AIC cores</td>
    <td>set_atomic_add_multi_aic_custom</td>
    <td>SetFlag/WaitFlag AIC inter-core synchronization</td>
  </tr>
</table>

### 2.2 Computation Formula and Sample Specifications

All three scenarios use the same computation formula:

$$
z = src0 + src2 + src3 + src1
$$

Where:
- `src0` is the GM initial value vector (all zeros)
- `src1`, `src2`, `src3` are three input vectors participating in atomic accumulation
- The accumulation order is fixed as: `src0` (initial) → +`src2` → +`src3` → +`src1`
- `z` is the final accumulation result

#### 2.2.1 SCENARIO_NUM=1 (Deterministic Computation within a Single AIV Core)
- **Computation Method**: Execute atomic accumulation operations sequentially in a fixed order within a single AIV core
- **Synchronization Mechanism**: Use PipeBarrier to ensure each atomic operation completes before executing the next one

- **Sample Specifications**:
  <table border="1">
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">SetAtomicAddSingleAiv</td></tr>
  <tr><td rowspan="5" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src0</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">src1</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">src2</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">src3</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Number of Cores (numBlocks)</td><td colspan="4" align="center">1</td></tr>
  </table>

#### 2.2.2 SCENARIO_NUM=2 and SCENARIO_NUM=3 (Deterministic Computation between Multiple Cores)
The core logic of these two scenarios is the same, with the difference being that the accumulated data is distributed across multiple AIV cores or AIC cores. Compared to the single-core scenario, the multi-core scenario requires synchronization between AIV cores or AIC cores. Therefore, the input has an additional sync_buf parameter for storing inter-core synchronization semaphores on GM, with an initial value that must be 0.

- **Sample Specifications**:
  <table border="1">
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">SetAtomicAddMultiCore</td></tr>
  <tr><td rowspan="6" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src0</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">src1</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">src2</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">src3</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td align="center">sync_buf</td><td align="center">[256]</td><td align="center">int32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">z</td><td align="center">[8]</td><td align="center">float32</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Number of Cores (numBlocks)</td><td colspan="4" align="center">4</td></tr>
  </table>

### 2.3 Notes

#### (1) When `SCENARIO_NUM=3`, the sample does not support Ascend 950PR/Ascend 950DT.
When `SCENARIO_NUM=1` and `SCENARIO_NUM=2`, the sample supports Ascend 950PR/Ascend 950DT.
When `SCENARIO_NUM=3`, `DataCopy` is called to move data from L1 to GM. However, on the Ascend 950PR/Ascend 950DT architecture, the DataCopy interface does not support the L1 Buffer -> GM path. Therefore, when `SCENARIO_NUM=3`, the sample does not support Ascend 950PR/Ascend 950DT. To support Ascend 950PR/Ascend 950DT, refer to the compatibility approach in the [Basic API Migration Guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_compatibility_10_00005.html).

#### (2) When `SCENARIO_NUM=2`, the sample does not support the static tensor programming style.
When SCENARIO_NUM=2, `IBSet` and `IBWait` are called for inter-core synchronization. However, the internal implementation of these two interfaces requires the TPipe framework for intra-core synchronization. Therefore, when SCENARIO_NUM=2, the sample does not support the static tensor programming style.

## Build and Run

Execute the following steps in the root directory of this sample to build and run it.
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
    
- Sample execution
  ```bash
  SCENARIO_NUM=1 # Default demonstrates deterministic computation within a single AIV core
  mkdir -p build && cd build;   # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=${SCENARIO_NUM};make -j;  # Build project
  python3 ../scripts/gen_data.py
  ./demo                        # Execute the compiled executable program to run the sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify if output results are correct, confirm algorithm logic is correct
  ```
  The following execution result indicates successful precision comparison:
  ```bash
  test pass!
  ```