# Matmul Basic API Best Practices Sample

## Overview

This sample is based on static Tensor programming paradigm, implementing high-performance matrix multiplication through L1/L0 double buffering, large packet transfer, fine-grained pipeline synchronization, UnitFlag, L2Cache, and other optimization methods. This sample is based on basic API implementation, using the same optimization methods as the high-level API version. The sample objective is to demonstrate tuning implementation details based on static Tensor programming paradigm.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_basic_api_high_performance
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script file
│   │   └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── mmad.asc                // Ascend C sample implementation
```

## Sample Description

  Matmul computation formula:
  $$
  C = A * B
  $$
  - A, B are source operands, A is the left matrix with shape [M, K]; B is the right matrix with shape [K, N]
  - C is the destination operand, storing the matrix multiplication result with shape [M, N]

- Sample Specification:



<table>
<tr><td rowspan="1" align="center">Sample Type(OpType)</td><td colspan="5" align="center">Matmul</td></tr>
<tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">isTrans</td></tr>
<tr><td align="center">A</td><td align="center">[M, K]</td><td align="center">half</td><td align="center">ND</td><td align="center">false</td></tr>
<tr><td align="center">B</td><td align="center">[K, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">true</td></tr>
<tr><td rowspan="1" align="center">Sample Output</td><td align="center">C</td><td align="center">[M, N]</td><td align="center">half</td><td align="center">ND</td><td align="center">-</td></tr>
<tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="5" align="center">mmad_custom</td></tr>
</table>

## Sample Implementation

### Performance Metric Description

| Metric | Description |
|------|------|
| Task Duration(μs) | Total execution time of the entire task, operator execution time should be based on this parameter |
| Block Num | Number of cores used (Block count) |
| aicore_time(μs) | Average execution time of AI Core |
| aic_mac_time(μs) | Execution time of Cube computation unit |
| aic_mac_ratio | Time ratio of Cube computation unit, reflecting computation unit utilization |
| aic_mte1_time(μs) | Execution time of MTE1 (L1 to L0A/L0B transfer) |
| aic_mte1_ratio | Time ratio of MTE1, reflecting L1 to L0 data transfer pressure |
| aic_mte2_time(μs) | Execution time of MTE2 (GM to L1 transfer) |
| aic_mte2_ratio | Time ratio of MTE2, reflecting GM to L1 data loading pressure |
| aic_fixpipe_time(μs) | Execution time of FixPipe (L0C to GM transfer) |
| aic_fixpipe_ratio | Time ratio of FixPipe, reflecting result write-back memory access pressure |


### Data Flow Path:
```
GM ──(MTE2, DataCopy)──> L1 ──(MTE1, LoadData)──> L0A/L0B ──(Cube, Mmad)──> L0C ──(Fixpipe)──> GM
       DataCopyInA/B             DataLoadA/B                   Compute             CopyOut
```

### Core Features

#### 1. L1/L0 Double Buffer Ping-Pong Layout

Both L1 and L0 use Ping-Pong double buffering, forming a three-stage pipeline for DataCopyIn (GM→L1), DataLoad (L1→L0), and Compute, with each stage processing data from different buffers without blocking each other.

```
Time ──────────────────────────────────────────────────────────────>

MTE2:  |─ A1 Ping ──|─ A1 Pong ──|─ A1 Ping ──| ...
MTE1:               |─ A2 Ping ──|─ A2 Pong ──|─ A2 Ping ──| ...
Cube:                            |─ Mmad ─────|─ Mmad ─────| ...
Fixpipe:                           |─ CopyOut ──|(unitflag)
```


**L1 Double Buffer Layout**: A1 occupies L1 front half (0~256KB), B1 occupies L1 back half (256~512KB), each further divided into Ping/Pong blocks:

```
L1 (512KB):
├── A1 Ping: [0, 128KB)
├── A1 Pong: [128KB, 256KB)
├── B1 Ping: [256KB, 384KB)
└── B1 Pong: [384KB, 512KB)
```

**L0 Double Buffer Layout**: A2/B2 each have independent 64KB space, each further divided into Ping/Pong:

```
L0A/L0B (64KB):
├── A2 Ping: [0, 16KB)
├── A2 Pong: [32KB, 48KB)
├── B2 Ping: [0, 32KB)
└── B2 Pong: [32KB, 64KB)
```

```cpp
// A1: L1 Ping/Pong
AscendC::LocalTensor<half> a1LocalPing(AscendC::TPosition::A1, 0, a1PingpongSize);
AscendC::LocalTensor<half> a1LocalPong(AscendC::TPosition::A1, a1PingpongSize * sizeof(half), a1PingpongSize);
// A2: L0 Ping/Pong
AscendC::LocalTensor<half> a2LocalPing(AscendC::TPosition::A2, 0, a2PingpongSize);
AscendC::LocalTensor<half> a2LocalPong(AscendC::TPosition::A2, L0_PINGPONG_BYTES, a2PingpongSize);
```

#### 2. Large Packet Transfer

Through `stepKa`/`stepKb` parameters, package multiple basic blocks into one DataCopyIn operation (called "large packet"), reducing MTE2 transfer count. For example, `stepKa=8` means transferring 8 baseM * baseK blocks from GM to L1 in one operation.


```cpp
// DataCopyInA: Transfer stepKa baseK blocks in one operation
AscendC::Nd2NzParams nd2nzParams;
nd2nzParams.nValue = curM;
nd2nzParams.dValue = baseK * stepKa;  // Large packet contains stepKa baseM * baseK
```

#### 3. Fine-Grained Pipeline Synchronization

Use four types of hardware event flags to achieve precise pipeline synchronization, divided into forward synchronization (data ready notification) and reverse synchronization (buffer release notification):

| Event Type | Direction | Purpose | flag Number |
|---------|------|------|----------|
| MTE2_MTE1 | Forward | L1 data ready notification, DataCopyIn notifies DataLoad can read | 0/1: A1 Ping/Pong; 2/3: B1 Ping/Pong |
| MTE1_MTE2 | Reverse | L1 buffer release notification, DataLoad notifies DataCopyIn can write | Same as above |
| MTE1_M | Forward | L0 data ready notification, DataLoad notifies Compute can compute | mte1DBFlag (0/1 alternating) |
| M_MTE1 | Reverse | L0 buffer release notification, Compute notifies DataLoad can write | mte1DBFlag (0/1 alternating) |

**Reverse Synchronization Needs Pre-setting**: Since reverse synchronization is "consumer SetFlag → producer WaitFlag", must preset SetFlag before first use, otherwise first WaitFlag will deadlock:

```cpp
// Initialization: Pre-set reverse synchronization flags, prevent first WaitFlag deadlock
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);  // A1 Ping writable
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);  // A1 Pong writable
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);  // B1 Ping writable
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);  // B1 Pong writable
AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(0);     // L0 Ping writable
AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(1);     // L0 Pong writable
```

**Large Packet Granularity Forward Synchronization**: DataLoad reads data from L1 large packet by base block in K direction, only needs to wait for data ready when reading the first base block of large packet, subsequent base blocks are in the same large packet, no need to wait repeatedly:

```cpp
// Forward synchronization: Only large packet first element needs to wait for data ready
if (kOffsetInChunkA == 0) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(a1ReadIdx);
}
// Reverse synchronization: Only notify DataCopyIn can overwrite after large packet last element consumed
if ((kOffsetInChunkA + 1) == stepKa) {
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(a1ReadIdx);
}
```

#### 4. LoadData3D Replaces LoadData2D——Reduce Instruction Queue Occupancy

On Atlas A2/A3 architecture, this sample uses `LoadData3DParamsV2` (namely LoadData3D) instead of `LoadData2DParams` (namely LoadData2D) to complete L1→L0 data transfer. This is a key instruction queue optimization.

**Problem Background**: MTE1 instruction queue depth is 32. When using LoadData2D, due to limited single LoadData2D instruction transfer granularity, transferring a baseM×baseK slice requires emitting multiple LoadData2D instructions in a for loop. For example when baseM=128, baseK=64, need to emit at least `baseK/16 = 4` LoadData2D instructions.

> **Note**: Atlas A5 chip provides `LoadData2DParamsV2` interface, single instruction can complete transfer, no need for LoadData3D. Therefore this sample uses `__NPU_ARCH__` conditional compilation to distinguish LoadData implementation for two architectures.

#### 5. Multi-Core Parallel Splitting

Evenly split matrix to multi-core parallel computation in M/N direction, 4×6 splitting strategy (M direction 4 blocks, N direction 6 blocks, total 24 cores) satisfies 512B address alignment, and reduces same address access conflict:

```cpp
constexpr uint32_t mIter = AscendC::DivCeil(M, singleCoreM);
uint32_t mIterIdx = AscendC::GetBlockIdx() % mIter;
uint32_t nIterIdx = AscendC::GetBlockIdx() / mIter;
```

#### 6. Constant Tiling

All Tiling parameters (baseM/baseK/baseN, singleCoreM/K/N, stepKa/stepKb) are determined at compile time through template parameters, no Scalar dynamic computation needed at runtime, reducing Scalar overhead:

```cpp
template <uint32_t M, uint32_t K, uint32_t N, uint32_t baseM, uint32_t baseK, uint32_t baseN,
          uint32_t singleCoreM, uint32_t singleCoreK, uint32_t singleCoreN,
          uint32_t stepKa, uint32_t stepKb>
class KernelMmad { ... };
```

#### 7. UnitFlag Optimization

After enabling UnitFlag, MMAD and FIXPIPE achieve fine-grained (512B) pipeline parallelism, instead of instruction-level synchronization. Whenever Cube completes 512B data result computation, FIXPIPE immediately transfers out that data, Cube computation and result write-back pipeline overlap:

```cpp
mmadParams.unitFlag = (kBlockIdx != kLoopCount - 1) ? 2 : 3;  // Enable UnitFlag
```

- `unitFlag = 2`: Intermediate K block, MMAD computation result does not write back immediately, but waits for next 512B completion then pipeline transfer out
- `unitFlag = 3`: Last K block, notifies FIXPIPE to write back all remaining results to GM

#### 8. DataCopyIn Prioritizes B Matrix Transfer

When `stepKa > stepKb`, B matrix needs to switch to next L1 buffer (Pong) every `stepKb` baseK, while A matrix needs `stepKa` baseK to switch. Therefore B data demand is more urgent. If transfer A first, MTE2 pipeline is occupied by A, B transfer has to wait until A completes before starting, causing B data not ready when needed.

This sample when triggering DataCopyIn after Compute, **transfers B first then A**, prioritizing more urgent B data:

```
k=0:  Compute → DataCopyIn(B1 Pong) → DataCopyIn(A1 Pong)
                 ↑ MTE2 transfers B first        ↑ Then transfers A
k=stepKb:       Needs B1 Pong → Ready ✓ (B has stepKb rounds time to transfer)
k=stepKa:       Needs A1 Pong → Ready ✓ (A has stepKa rounds time to transfer, more ample)
```

#### 9. L2Cache Optimization

L2Cache is AI Core shared external cache, pure read bandwidth is approximately 3 to 4 times GM. If data cannot hit L2Cache, need to access GM, bandwidth utilization efficiency is lower, causing MTE2 to become performance bottleneck.

L2Cache splitting specific implementation is consistent with Case 6 in [High-level API Matmul Sample](../matmul_high_performance/README.md), core idea is both splitting A matrix M-axis, making B matrix stay in L2Cache across rounds. This sample implements double outer loop through `ProcessL2Cache()` method, schedules by rounds according to `outerMIdx`, within each round 24 cores parallelly compute their own sub-blocks:

```cpp
// ProcessL2Cache: Split by rounds in M direction, each round 24 cores cover mIterPerRound M sub-blocks
constexpr uint32_t mIterPerRound = AscendC::DivCeil(M, singleCoreM * 2);
constexpr uint32_t outerMLoopCount = AscendC::DivCeil(mIterTotal, mIterPerRound);

for (uint32_t outerMIdx = 0; outerMIdx < outerMLoopCount; outerMIdx++) {
    uint32_t mIterIdx = AscendC::GetBlockIdx() % mIterPerRound + outerMIdx * mIterPerRound;
    uint32_t nIterIdx = AscendC::GetBlockIdx() / mIterPerRound;
    if (mIterIdx >= mIterTotal || nIterIdx >= nIterTotal) continue;
    InitComputeParamsL2Cache(mIterIdx, nIterIdx);
    ProcessLoop(...);
}
```

#### 10. K Direction Main Loop Complete Flow

Using stepKa=8, stepKb=4 as example, showing detailed execution flow of K direction loop within a complete (mBlockIdx, nBlockIdx) sub-block:

```
Pre-processing:
  SetFlag(MTE1_MTE2, 0/1/2/3)  // Pre-set reverse synchronization: L1 Ping/Pong both writable
  SetFlag(M_MTE1, 0/1)         // Pre-set reverse synchronization: L0 Ping/Pong both writable
  DataCopyIn(A1 Ping, k=0)     // Transfer first A large packet to Ping
  DataCopyIn(B1 Ping, k=0)     // Transfer first B large packet to Ping
  SetFlag(MTE2_MTE1, 0/2)      // Notify A1/B1 Ping data ready

K-loop kBlockIdx = 0, 1, ..., kLoopCount-1:
  ┌─ a1ReadIdx = (kBlockIdx / stepKa) % 2        // Currently read L1 A Ping/Pong
  │  b1ReadIdx = (kBlockIdx / stepKb) % 2        // Currently read L1 B Ping/Pong
  │  kOffsetInChunkA = kBlockIdx % stepKa        // Current baseK offset within A large packet
  │  kOffsetInChunkB = kBlockIdx % stepKb        // Current baseK offset within B large packet
  │
  │  WaitFlag(M_MTE1, mte1DBFlag)                // Wait for previous round Compute release L0
  │  if (kOffsetInChunkA == 0)
  │      WaitFlag(MTE2_MTE1, a1ReadIdx)          // Wait L1 A large packet data ready (only first element)
  │  if (kOffsetInChunkB == 0)
  │      WaitFlag(MTE2_MTE1, b1ReadIdx + 2)      // Wait L1 B large packet data ready (only first element)
  │
  │  DataLoadA(A1 → A2)                          // L1 → L0
  │  DataLoadB(B1 → B2)                          // L1 → L0
  │
  │  if (kOffsetInChunkA + 1 == stepKa)
  │      SetFlag(MTE1_MTE2, a1ReadIdx)           // A large packet last element: notify L1 A writable
  │  if (kOffsetInChunkB + 1 == stepKb)
  │      SetFlag(MTE1_MTE2, b1ReadIdx + 2)       // B large packet last element: notify L1 B writable
  │
  │  Compute(Mmad)                               // Cube computation (M instruction)
  │  SetFlag(M_MTE1, mte1DBFlag)                 // Notify L0 can overwrite
  │  mte1DBFlag ^= 1                             // Switch L0 Ping/Pong
  │
  │  // DataCopyIn after Compute, B first then A
  │  if (B large packet last element && has B data):
  │      WaitFlag(MTE1_MTE2, b1WriteIdx + 2)     // Wait L1 B buffer writable
  │      DataCopyInB(next B large packet)        // GM → L1 (MTE2 instruction)
  │      SetFlag(MTE2_MTE1, b1WriteIdx + 2)      // Notify L1 B data ready
  │  if (A large packet last element && has A data):
  │      WaitFlag(MTE1_MTE2, a1WriteIdx)         // Wait L1 A buffer writable
  │      DataCopyInA(next A large packet)        // GM → L1 (MTE2 instruction)
  │      SetFlag(MTE2_MTE1, a1WriteIdx)          // Notify L1 A data ready
  └─
```

**Timing Diagram** (stepKa=8, stepKb=4):

```
Prefetch: DataCopyIn(A1 Ping) + DataCopyIn(B1 Ping)                     ← Before K-loop

k=0:  WaitFlag(A1Ping, B1Ping) → DataLoad → Compute → DataCopyIn(B1Pong) → DataCopyIn(A1Pong)
k=1:  DataLoad → Compute
k=2:  DataLoad → Compute
k=3:  DataLoad(release B1Ping) → Compute → DataCopyIn(transfer B1Ping)

k=4:  WaitFlag(B1Pong) → DataLoad(B1Pong) → Compute
k=5:  DataLoad → Compute
k=6:  DataLoad → Compute
k=7:  DataLoad(release A1Ping B1Pong) → Compute → DataCopyIn(B1Pong) → DataCopyIn(A1Ping)
k=8:  WaitFlag(A1Pong ready ✓) → WaitFlag(B1Ping ready ✓) → DataLoad → Compute 
...
```


### Performance Data Analysis

#### Atlas A2 Training Series Chip Performance Data
- Scenario 1: Not enable L2Cache splitting, singleCoreM=2048, singleCoreN=1536, 24 cores one round full coverage
- Scenario 2: Enable L2Cache splitting, singleCoreM=1024, singleCoreN=1536, 24 cores split into 2 rounds computation

| Scenario | Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| Scenario 1 | 4121.16 | 24 | 3670.7 | 3081.664 | 0.84 | 337.343 | 0.092 | 2538.348 | 0.692 | 3552.248 | 0.968 | 160.405 | 0.044 |
| Scenario 2 | 4081.64 | 24 | 3636.85 | 3082.158 | 0.847 | 345.139 | 0.095 | 2553.064 | 0.702 | 3487.068 | 0.959 | 161.812 | 0.044 |

Excluding startup overhead, has achieved 84.7% of this chip's peak computing power.

After enabling L2Cache splitting, aic_mte2_time reduced from 3552.248μs to 3487.068μs, reduced by 1.84%. Current splitting strategy is simple, users can further optimize L2Cache splitting strategy to improve MTE2 bandwidth.


#### Ascend 950PR Chip Performance Data

- Scenario 1: Not enable L2Cache splitting, singleCoreM=2048, singleCoreN=1024, 32 cores one round full coverage
- Scenario 2: Enable L2Cache splitting, singleCoreM=1024, singleCoreN=1024, 32 cores split into 2 rounds computation

| Scenario | Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| Scenario 1 | 2572.047 | 32 | 2571.44 | 2564.813 | 0.997 | 144.604 | 0.056 | 828.001 | 0.322 | 1874.267 | 0.729 | 221.997 | 0.086 |
| Scenario 2 | 2574.492 | 32 | 2573.39 | 2564.147 | 0.996 | 104.845 | 0.041 | 819.207 | 0.318 | 1892.742 | 0.736 | 223.129 | 0.087 |

Has achieved 99.7% of this chip's peak computing power.

After enabling L2Cache splitting, there is no obvious effect on Ascend 950PR chip. Reason: L2Cache optimization goal is to alleviate MTE2 bound, but current bottleneck is Cube computation rather than data transfer, so reducing MTE2 duration cannot improve overall performance; additionally L2Cache splitting divides computation into 2 rounds scheduling, introducing extra Scalar overhead and scheduling overhead, causing Scenario 2 Task Duration slightly higher than Scenario 1. Meanwhile, Scenario 2 aic_mte2_time (1892.742μs) is slightly higher than Scenario 1 (1874.267μs), because when sample is in Cube bound, MTE2 pipeline is blocked by Cube computation, profiler collected aic_mte2_time includes pipeline wait time rather than pure data transfer time. L2Cache optimization although reduced actual data access latency, but is masked by Cube computation bottleneck, cannot reflect in MTE2 metric.


### Theoretical Performance Analysis

#### Cube Computation Performance Analysis

**Atlas A2 Training Series Chip**: Sample parameters M=N=K=8192, baseM=128, baseN=256, baseK=64, computation chip main frequency is 1.85GHz, processing 16×16×16 multiply-add operations per cycle.

$$cube\_time = \frac{M \times N \times K}{16 \times 16 \times 16 \times core\_num \times cube\_freq} = \frac{8192 \times 8192 \times 8192}{16 \times 16 \times 16 \times 24 \times 1850} = 3022.92\mu s$$

Cube computation duration error:

$$Error = \frac{aic\_mac\_time - cube\_time}{cube\_time} = \frac{3082.158 - 3022.92}{3022.92} = 1.95\%$$


**Ascend 950PR Chip**: Sample parameters M=N=K=8192, baseM=256, baseN=256, baseK=64, processor main frequency is 1.65GHz, processing 16×16×16 multiply-add operations per cycle.

$$cube\_time = \frac{M \times N \times K}{16 \times 16 \times 16 \times core\_num \times cube\_freq} = \frac{8192 \times 8192 \times 8192}{16 \times 16 \times 16 \times 32 \times 1650} = 2542.00\mu s$$

Cube computation duration error:

$$Error = \frac{aic\_mac\_time - cube\_time}{cube\_time} = \frac{2564.813 - 2542.00}{2542.00} = 0.90\%$$


#### MTE2 Bandwidth Analysis

**Total Data Read**:

Atlas A2 Training Series Chip (baseM=128, baseN=256):

$$Total data read = \left(\frac{N}{baseN} \times M \times K + \frac{M}{baseM} \times K \times N\right) \times sizeof(half) = (32 \times 8192 \times 8192 + 64 \times 8192 \times 8192) \times 2B = 12GB$$

Ascend 950PR Chip (baseM=256, baseN=256):

$$Total data read = \left(\frac{N}{baseN} \times M \times K + \frac{M}{baseM} \times K \times N\right) \times sizeof(half) = (32 \times 8192 \times 8192 + 32 \times 8192 \times 8192) \times 2B = 8GB$$

**MTE2 Theoretical Duration**:

Atlas A2 Training Series Chip: L2Cache peak bandwidth approximately 5TB/s, HBM bandwidth approximately 1.8TB/s. First read from HBM, subsequent read from L2Cache.

$$First time data read from HBM total = M \times K \times sizeof(half) + K \times N \times sizeof(half) = 256MB$$

$$MTE2 theoretical duration = \frac{HBM read data total}{1.8TB/s} + \frac{L2Cache read data total}{5TB/s}$$

MTE2 duration error:

$$MTE2 duration error = \frac{3487.068 - 2672.44}{2672.44} = 30.48\%$$

Current MTE2 duration differs significantly from theoretical value because actual chip L2Cache size is 192MB, current L2Cache splitting strategy is simple; on the other hand, when MTE2 transfer scenario is ND2NZ (GM data Layout is ND, transfer to L1 needs ND→NZ format conversion), L2Cache bandwidth decreases. Users can further optimize L2Cache splitting strategy to improve MTE2 bandwidth.

Ascend 950PR Chip: L2Cache peak bandwidth approximately 5TB/s, HBM bandwidth approximately 1.6TB/s.

$$MTE2 theoretical duration = \frac{HBM read data total}{1.6TB/s} + \frac{L2Cache read data total}{5TB/s}$$

MTE2 duration error:

$$MTE2 duration error = \frac{1874.267 - 1832.10}{1832.10} = 2.30\%$$

Compared to Atlas A2 Training Series Chip, Ascend 950PR Chip data transfer is more efficient, MTE2 bandwidth utilization is higher.



## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

- Configure Environment Variables
  Select the corresponding environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in the current environment.
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

- Sample Execution
  ```bash
  SCENARIO=1                                                           # Select execution scenario (1 for not enabling L2Cache splitting, 2 for enabling L2Cache splitting)
  mkdir -p build && cd build;                                          # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;  # Build project (default npu mode)
  python3 ../scripts/gen_data.py                                       # Generate test input data
  ./demo                                                               # Execute compiled executable program
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin
  ```

  When using CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;   # NPU simulation mode
  ```

  > **Note:** Before switching build mode, need to clear cmake cache. Can execute `rm CMakeCache.txt` in build directory and then re-run cmake.

- Build Option Description

  | Parameter | Description | Available Values | Default Value |
  |------|------|---------|--------|
  | `SCENARIO_NUM` | `1` / `2` | 1: Not enable L2Cache splitting; 2: Enable L2Cache splitting | `1` |
  | `CMAKE_ASC_RUN_MODE` | Run mode | `npu`, `sim` | `npu` |
  | `CMAKE_ASC_ARCHITECTURES` | NPU hardware architecture | `dav-2201`, `dav-3510` | `dav-2201` |

  The execution result shown below indicates the accuracy comparison succeeded.
  ```bash
  test pass!
  ```

## Performance Analysis

Use the `msprof` tool to obtain detailed performance data:

```bash
msprof ./demo   # Analyze sample performance
```

A PROF_ prefixed folder will be generated in the current directory, with `mindstudio_profiler_output` directory storing Host and various Device performance data summary. Performance data analysis is recommended to view files in this directory:
```bash
PROF_xxxx_XXXXXX
├── device_{id}
└── host
└── mindstudio_profiler_log
└── mindstudio_profiler_output    # Store Host and various Device performance data summary
    ├── msprof_*.json
    ├── xx_*.csv
    └── README.txt
```