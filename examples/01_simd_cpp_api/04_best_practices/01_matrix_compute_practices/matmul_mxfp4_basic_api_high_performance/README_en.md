# MxFP4 Matmul Basic API High Performance Sample

## Overview

This sample demonstrates how to implement a high-performance MxFP4 Matmul kernel using Ascend C basic API and static Tensor programming, with various optimization techniques including L1/L0 double buffering, large packet transfer, and fine-grained pipeline synchronization.

## Supported Products

- Ascend 950PR / Ascend 950DT

## Directory Structure

```text
├── matmul_mxfp4_basic_api_high_performance
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script file
│   │   └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   └── mmad_mx.asc             // Ascend C basic API sample implementation
```

## MxMatmul Introduction

Regular Matmul has only two matrix inputs:

$$
C = A * B
$$

MxMatmul introduces two additional scale inputs. A and B are MxFP4 low-bit data, while scaleA and scaleB are scaling factors. During computation, matrix data and corresponding scales participate in the calculation together:

$$
C = (\text{scaleA} \otimes A) \times (\text{scaleB} \otimes B)
$$

In MxMatmul, every 32 elements in the K direction share one scale. That means the number of scales is much smaller than the number of matrix elements, but it broadcasts along the K direction to the corresponding group of MxFP4 data, as shown in the figure:

  <img src="figure/MxMatmul.png"  width="80%">

## Sample Specification

This sample implements MxFP4 Matmul with a fixed shape of `8192 x 8192 x 8192`, with output data type `bfloat16_t`.

### Input and Output

| Input/Output | Logical Shape | Data Type | Data Layout Type | Description |
|------|------|----------|------|------|
| A | `[M, K]` | `fp4x2_e1m2_t` | ND | Left matrix, each byte packs 2 fp4 elements |
| scaleA | `[M, scaleK]` | `fp8_e8m0_t` | ND | Scaling factor for A matrix, every 32 elements in K direction of A matrix share one scale |
| B | `[N, K]` | `fp4x2_e1m2_t` | ND | Right matrix, input to kernel in `[N, K]` format |
| scaleB | `[N, scaleK]` | `fp8_e8m0_t` | ND | Scaling factor for B matrix, every 32 elements in K direction of B matrix share one scale |
| C | `[M, N]` | `bfloat16_t` | ND | Output matrix |

Where:
- Two `fp4x2_e1m2_t` are packed and stored in one byte, so when the data type is `fp4`, K must be even
- `scaleK = align_even(ceil(K / 32))`, meaning round up first then align to multiples of 2. Due to hardware constraints requiring scale data to satisfy 2Byte continuous alignment in K direction, scaleK must be even
- Due to hardware constraints, when the scaleB matrix is input as `[scaleK, N]`, it requires 2Byte continuity in K direction, so it is recommended to input ScaleB in `[N, scaleK]` format

#### Four-Path Input Explanation

Compared with regular Matmul, MxMatmul has additional scale transfer, load, and computation paths in the kernel. A/B and scaleA/scaleB must enter the computation pipeline in the same K block rhythm, otherwise the Cube side cannot obtain matching scaling information.

- The data layout format of the four-path inputs is shown in the figure below:

  <img src="figure/formatOfMx.png">

- The transfer and computation of the four-path inputs is shown in the figure below:

  <img src="figure/InputOfMxMatmul.png">

### Key Parameters

| Parameter | Value | Description |
|------|----|------|
| `M` | `8192` | Matrix M dimension size |
| `N` | `8192` | Matrix N dimension size |
| `K` | `8192` | Matrix K dimension size |
| `baseM` | `256` | Cube computation base block M dimension size |
| `baseK` | `256` | Cube computation base block K dimension size |
| `baseN` | `256` | Cube computation base block N dimension size |
| `singleCoreM` | `2048` | Single-core M direction computation range |
| `singleCoreN` | `1024` | Single-core N direction computation range |
| `singleCoreK` | `8192` | Single-core K direction computation range |
| `stepKa` | `2` | In GM->L1 transfer, large packet transfer step for A in K direction |
| `stepKb` | `2` | In GM->L1 transfer, large packet transfer step for B in K direction |
| `scaleFactorKa` | `4` | In GM->L1 transfer, scaleA transfer ratio relative to A in K direction |
| `scaleFactorKb` | `4` | In GM->L1 transfer, scaleB transfer ratio relative to B in K direction |
| `Block Num` | `32` | Number of cores used |

> **Constraint**: This sample does not support the scenario where there are tail blocks in the K direction, meaning `K` must be divisible by `baseK`.

## Sample Implementation

The entire kernel process: after multi-core splitting, each core is responsible for a singleCoreM * singleCoreN sub-matrix. It loops in the M/N direction, transfers A/B and scale into L1, then loops in the K direction, loads to L0A/L0B, completes Cube accumulation computation, and finally writes the results back to GM.

### Data Flow Path

The overall data flow is as follows:

```text
Physical Address                               Pipeline
  GM
  |  DataCopyInA / DataCopyInAScale   MTE2
  |  DataCopyInB / DataCopyInBScale   MTE2
  v
  L1
  |  DataLoadA(with MX scaleA)        MTE1
  |  DataLoadB(with MX scaleB)        MTE1
  v
  L0A / L0B / L0A_MX / L0B_MX
  |
  |  Mmad                              M
  v
  L0C
  |
  |  Fixpipe: float -> bfloat16       FIX
  v
  GM
```

#### 1. Multi-Core Splitting: Split Large Matrix into 32 Sub-tasks

The sample uses 32 cores in parallel. `M=8192` is split into 4 parts by `singleCoreM=2048`, `N=8192` is split into 8 parts by `singleCoreN=1024`, forming exactly `4 x 8 = 32` sub-matrices.

```text
M direction: 8192 / 2048 = 4 blocks
N direction: 8192 / 1024 = 8 blocks

Total blocks = 4 * 8 = 32
```

Each core only processes its own `2048 x 1024` output region. Within the core, it loops in the K direction by `baseK=256` blocks to complete accumulation.

The corresponding core index calculation is as follows:

```cpp
constexpr uint32_t mIter = AscendC::DivCeil(M, singleCoreM);
uint32_t mIterIdx = AscendC::GetBlockIdx() % mIter;
uint32_t nIterIdx = AscendC::GetBlockIdx() / mIter;

uint64_t gmOffsetA = mIterIdx * singleCoreM * K;
uint64_t gmOffsetB = nIterIdx * K * singleCoreN;
uint64_t gmOffsetC = mIterIdx * singleCoreM * N + nIterIdx * singleCoreN;
```

#### 2. GM to L1: Matrix and Scale Enter Pipeline Together

MxMatmul has four input paths: A, B, scaleA, and scaleB. Therefore, the GM to L1 transfer is also divided into four types:

| Data | Transfer Function | GM Source Data | L1 Target |
|------|----------|-----------|---------|
| A | `DataCopyInA` | `A [M, K]` | `A1` |
| B | `DataCopyInB` | `B [N, K]` | `B1` |
| scaleA | `DataCopyInAScale` | `scaleA [M, scaleK]` | `scaleA1` |
| scaleB | `DataCopyInBScale` | `scaleB [N, scaleK]` | `scaleB1` |

Through `stepKa=2` and `stepKb=2`, transfer `stepKa * baseM * baseK` A matrix data and `stepKb * baseN * baseK` B matrix data in one operation, reducing the number of transfer instructions and improving GM to L1 transfer efficiency.

The scale transfer granularity is controlled by `scaleFactorKa=4` and `scaleFactorKb=4`, transferring `scaleFactorKa * stepKa * baseM * baseSK` scaleA matrix data and `scaleFactorKb * stepKb * baseN * baseSK` scaleB matrix data in one operation, where `baseSK = baseK / 32`. Its purpose is to let scaleA/scaleB cover a larger range in the K direction compared to A/B in one transfer, reducing the MTE2 pressure from repeated scale transfers.

For the A side example, `DataCopyInA` uses `Nd2NzParams` to transfer `stepKa` K direction base blocks in one operation, each base block is `baseM * baseK`:

```cpp
constexpr uint32_t packedStepK = AscendC::DivCeil(baseK * stepKa, 2);
AscendC::Nd2NzParams nd2nzA1Params;
nd2nzA1Params.ndNum = 1;
nd2nzA1Params.nValue = curM;
nd2nzA1Params.dValue = packedStepK;
nd2nzA1Params.srcDValue = PACKED_K;
nd2nzA1Params.dstNzC0Stride = baseM;
nd2nzA1Params.dstNzNStride = 1;
AscendC::DataCopy(a1Local, aGM[kChunkIdx * baseK + mBlockIdx * K * baseM], nd2nzA1Params);
```

The scale side transfers in b16 format. Since `Mmad` requires K direction continuity when reading data with minimum fractal, scale data must satisfy 2Byte continuity in K direction, so b16 type transfer is used to ensure correct data layout:

```cpp
constexpr uint32_t stepScaleK = AscendC::DivCeil(baseK * stepKa * scaleFactorKa, SCALE_CEIL_NUMBER);
AscendC::Dn2NzParams dn2nzParams;
dn2nzParams.dValue = curM;
dn2nzParams.nValue = stepScaleK / 2;
dn2nzParams.srcDValue = SCALE_K / 2;
dn2nzParams.dstNzC0Stride = stepScaleK / 2;

auto asLocalB16 = as1Local.ReinterpretCast<half>();
AscendC::DataCopy(asLocalB16, asGMB16, dn2nzParams);
```

#### 3. L1 to L0: LoadData Enters MX Computation Path

Regular Matmul only needs to load A/B; MxMatmul's `LoadData` also needs to bring the corresponding scale, transferring one base block at a time.

```text
A1 block + scaleA1 block -> L0A MX data
B1 block + scaleB1 block -> L0B MX data
```

When A/B and scale enter L0, the Cube computation unit can complete scaled matrix multiplication accumulation according to MX semantics.

The key code is `LoadData` receiving both matrix LocalTensor and scale LocalTensor:

```cpp
uint32_t srcAddr = kOffsetInChunkA * baseK * baseM;
uint32_t scaleSrcAddr = (kOffsetInScaleChunkA * baseK / SCALE_CEIL_NUMBER) * CUBE_BLOCK;

AscendC::LoadData(a2Local,
                  a1Local[srcAddr],
                  as1Local[scaleSrcAddr],
                  loadDataParams,
                  loadMxDataParams);
```

#### 4. L0 Double Buffering: Overlap Loading and Computation

The sample uses Ping-Pong double buffering at both L1 and L0. The core goal is to overlap transfer, loading, and computation as much as possible, reducing waiting.

The main buffer layout of this sample is as follows:

| Level | Buffer | Content | Purpose |
|------|--------|------|------|
| L1 | `A1 Ping/Pong` | A large packet data | Store A data from GM->L1 |
| L1 | `B1 Ping/Pong` | B large packet data | Store B data from GM->L1 |
| L1 | `scaleA1 Ping/Pong` | A corresponding scale data | Store A-side MXScale |
| L1 | `scaleB1 Ping/Pong` | B corresponding scale data | Store B-side MXScale |
| L0A | `A2 Ping/Pong` | Current K block A data | Cube `Mmad` left operand |
| L0B | `B2 Ping/Pong` | Current K block B data | Cube `Mmad` right operand |
| L0C | `cLocal` | float accumulation result | `Mmad` output, for `Fixpipe` write-back |

> **Note**: `L0A_MX` and `L0B_MX` are used to store scale data, their addresses have a fixed relationship with `L0A`/`L0B`, no need for user manual allocation.

The corresponding LocalTensor creation is as follows:

```cpp
AscendC::LocalTensor<fp4x2_e1m2_t> a1LocalPing(AscendC::TPosition::A1, 0, a1BufSize);
AscendC::LocalTensor<fp4x2_e1m2_t> a1LocalPong(AscendC::TPosition::A1, a1BufSize, a1BufSize);
AscendC::LocalTensor<fp8_e8m0_t> as1LocalPing(AscendC::TPosition::A1, 2 * a1BufSize, as1BufSize);
AscendC::LocalTensor<fp8_e8m0_t> as1LocalPong(AscendC::TPosition::A1, 2 * a1BufSize + as1BufSize, as1BufSize);

AscendC::LocalTensor<fp4x2_e1m2_t> a2LocalPing(AscendC::TPosition::A2, 0, a2PingpongSize);
AscendC::LocalTensor<fp4x2_e1m2_t> a2LocalPong(AscendC::TPosition::A2, L0_PINGPONG_BYTES, a2PingpongSize);
```

A simplified pipeline rhythm is as follows:

```text
time     |---------------------------------------------------------------------------->

GM->L1   | A/B/scaleA/scaleB Ping | A/B/scaleA/scaleB Pong | A/B/scaleA/scaleB Ping |
L1->L0                            | L0 Ping load --|       | L0 Pong load --|    
Cube                                               | Mmad Ping ---|   | Mmad Pong ---|
Fixpipe                                                           | fixpipe C ---
```

The horizontal line length in the figure only expresses that different stages may have different durations, and does not represent measured proportions; actual duration should be based on `msprof` collection results.

In the K loop, when the current K block enters Cube computation, the next batch of A/B/scaleA/scaleB can initiate transfer in advance. Ping and Pong are used alternately, and producers and consumers use event synchronization to confirm whether the buffer is writable and whether the data is readable.

#### 5. Cube Computation: Accumulate by K Block

Each output sub-matrix will first loop along the K direction in the innermost layer. Each round processes a `baseK=256` K block, then loops along the M/N direction:

```text
for nBlock in N blocks:
  for nBlock in M blocks:
    for kBlock in K blocks:
        LoadData(A block, scaleA block)
        LoadData(B block, scaleB block)
        Mmad accumulate
```

`Mmad` input comes from L0A/L0B, output accumulates to L0C. The first K block initializes the accumulation result, subsequent K blocks continue accumulating until the complete K direction computation finishes.

Key parameters in the computation phase include `m/n/k` sizes and whether to initialize the C matrix:

```cpp
AscendC::MmadParams mmadParams;
mmadParams.m = curM;
mmadParams.n = curN;
mmadParams.k = baseK;
mmadParams.cmatrixInitVal = (kBlockIdx == 0);
AscendC::Mmad(cLocal, a2Local, b2Local, mmadParams);
```

Where `cmatrixInitVal` controls the first K block to initialize the accumulation result, subsequent K blocks continue accumulating on existing L0C data.

#### 6. Fixpipe Write-back: Convert from float to bfloat16 Output

The Cube side accumulation result is stored in L0C with float data type. After computation finishes, `Fixpipe` converts the result to `bfloat16_t` and writes it back to GM.

```text
L0C float result
      |
      | Fixpipe, F32 -> BF16
      v
GM C [M, N]
```

During write-back, use `quantPre` to complete F32 to BF16 conversion:

```cpp
AscendC::FixpipeParamsArch3510<AscendC::CO2Layout::ROW_MAJOR> fixpipeParams;
fixpipeParams.nSize = curN;
fixpipeParams.mSize = curM;
fixpipeParams.srcStride = curMAlign;
fixpipeParams.dstStride = N;
fixpipeParams.quantPre = QuantMode_t::F322BF16;
AscendC::Fixpipe(cGM[mBlockIdx * baseM * N + nBlockIdx * baseN], cLocal, fixpipeParams);
```

### Event Synchronization: Parallelize the Pipeline

When using static Tensor programming, synchronization is key to performance and correctness. This sample mainly uses four types of events, with eventID controlled from `EVENT_ID0` to `EVENT_ID3`:

| Event | Direction | Purpose | flag Number |
|------|------|------|----------|
| `MTE2_MTE1` | GM->L1 notifies L1->L0 | After DataCopyIn completes, notify DataLoad that L1 data can be read | `EVENT_ID0/1`: A+B Data Ping/Pong; `EVENT_ID2/3`: As+Bs Scale Ping/Pong |
| `MTE1_MTE2` | L1->L0 notifies GM->L1 | After DataLoad consumes L1 data, notify DataCopyIn that it can overwrite that buffer | Same as above |
| `MTE1_M` | L1->L0 notifies Cube | After LoadData completes, notify Mmad that computation can start | `EVENT_ID0/1`: L0 Ping/Pong |
| `M_MTE1` | Cube notifies L1->L0 | After Mmad consumes L0 buffer, notify next LoadData that it can write | `EVENT_ID0/1`: L0 Ping/Pong |

A/B data have the same lifecycle, so they are bound to the same group of events by Ping/Pong phase; scaleA/scaleB have the same lifecycle, but scale chunks are usually larger than data chunks, so they use another group of Ping/Pong events separately. Code enforces compile-time constraints that `stepKa == stepKb`, and `stepKa * scaleFactorKa == stepKb * scaleFactorKb`.

Reverse synchronization needs to be preset before entering the main loop, otherwise the first wait for a writable buffer will block:

```cpp
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
```

Large packet granularity synchronization only waits for data ready at the first element of the large packet, and releases the buffer at the last element:

```cpp
if (kOffsetInDataChunk == 0) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(GetDataEventId(dataReadIdx));
}
if (((kOffsetInDataChunk + 1) == dataChunkStep) || (kBlockIdx + 1 == kLoopCount)) {
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(GetDataEventId(dataReadIdx));
}
if (kOffsetInScaleChunk == 0) {
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(GetScaleEventId(scaleReadIdx));
}
if (((kOffsetInScaleChunk + 1) == scaleChunkStep) || (kBlockIdx + 1 == kLoopCount)) {
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(GetScaleEventId(scaleReadIdx));
}
```

From the pipeline perspective, these events connect three phases:

```text
DataCopyIn(GM->L1)
        |
        | MTE2_MTE1
        v
LoadData(L1->L0)
        |
        | MTE1_M
        v
Mmad(Cube)
        |
        | M_MTE1
        v
next LoadData
```

Reverse events are also important. For example, `MTE1_MTE2` indicates that a Ping/Pong buffer in L1 has been consumed and can be overwritten by the next GM->L1 transfer. Without these reverse synchronizations, the pipeline can easily have problems like overwriting unconsumed data or waiting for incomplete data.

## Performance Optimization Summary

This sample's performance optimization revolves around three things:

- **Multi-core splitting**: Split the `8192 x 8192` output matrix to 32 cores for parallel computation.
- **Large packet transfer**: Transfer A/B by multiple K blocks, transfer scale by larger K coverage range, reducing MTE2 instruction and data transfer pressure.
- **Double buffer pipeline**: Both L1 and L0 use Ping-Pong, letting DataCopyIn, LoadData, Mmad, and Fixpipe overlap execution as much as possible.

## Theoretical Performance Comparison

Performance metric description table:
| Metric | Description |
|------|------|
| `Task Duration(μs)` | Total execution time of the entire task, operator end-to-end execution time should be based on this parameter |
| `Block Num` | Number of cores used, which is the number of blocks the kernel launches |
| `aicore_time(μs)` | Average execution time of AI Core |
| `aic_mac_time(μs)` | Cube computation unit execution time, mainly corresponding to `Mmad` phase |
| `aic_mac_ratio` | Cube computation unit time ratio, reflecting computation unit utilization |
| `aic_scalar_time(μs)` | Scalar instruction execution time, reflecting loop scheduling, address calculation, parameter configuration, and so on overhead |
| `aic_scalar_ratio` | Scalar time ratio |
| `aic_mte1_time(μs)` | MTE1 execution time, mainly corresponding to L1 to L0A/L0B `LoadData` |
| `aic_mte1_ratio` | MTE1 time ratio, reflecting L1 to L0 data transfer pressure |
| `aic_mte2_time(μs)` | MTE2 execution time, mainly corresponding to GM to L1 `DataCopyIn` |
| `aic_mte2_ratio` | MTE2 time ratio, reflecting GM to L1 data loading pressure |
| `aic_fixpipe_time(μs)` | Fixpipe execution time, mainly corresponding to L0C to GM result write-back |
| `aic_fixpipe_ratio` | Fixpipe time ratio, reflecting result write-back memory access pressure |

Ascend 950PR chip performance data:

| Case version | Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| Basic API MxMatmul | 681.056 | 32 | 679.99 | 640.16 | 0.941 | 62.824 | 0.092 | 312.728 | 0.46 | 588.736 | 0.866 | 32.422 | 0.048 |

This sample has reached `94.1%` of theoretical peak performance (the `aic_mac_ratio` in the table).

### Cube Computation Performance Analysis

This sample's performance data was obtained on Ascend 950PR, with a main frequency of 1.65GHz. For MX-FP4 data type, it processes 16×64×16 multiply-add operations per cycle.

Cube theoretical computation time $T_{cube}$ is:

$$T_{cube} = \frac{M \times N \times K}{16 \times 64 \times 16 \times 1.65 \times 10^9 \times \text{core_num}} = \frac{8192 \times 8192 \times 8192}{4096 \times 1.65 \times 10^9 \times 32} = 635.5 μs$$

From the table, `aic_mac_time` is `640.16 μs`. Relative to the theoretical value `635.5 μs`, the error $E_{cube}$ is:

$$E_{cube} = \frac{T_{actual} - T_{cube}}{T_{cube}} = \frac{640.16 - 635.5}{635.5} = 0.73 \%$$

### MTE2 Bandwidth Analysis

**Data Reuse Principle**:

In matrix multiplication, each element $C_{i,j}$ of output matrix C requires the i-th row of A and the j-th column of B to participate in computation. During block computation, the same input data block is reused by multiple output blocks:

- A matrix is split into `M/baseM` row blocks in the M direction, each A row block participates in computation of `N/baseN` output blocks in the N direction
- B matrix is split into `N/baseN` column blocks in the N direction, each B column block participates in computation of `M/baseM` output blocks in the M direction

MxFP4 Matmul also includes two scale inputs, with every 32 elements in the K direction sharing one scale, so `scaleK = K/32`:

- scaleA shape `[M, scaleK]`, split into `M/baseM` row blocks in the M direction, each scaleA row block participates in computation of `N/baseN` output blocks in the N direction
- scaleB shape `[scaleK, N]`, split into `N/baseN` column blocks in the N direction, each scaleB column block participates in computation of `M/baseM` output blocks in the M direction

Due to limited L1/L2Cache capacity that cannot cache all input data, the same data block is transferred multiple times from HBM to L2Cache/L1, causing repeated data transfer.

**Total Data Read**:

MxFP4 Matmul input contains four data paths: A, B, scaleA, scaleB. A/B matrices use `fp4x2_e1m2_t`, with 2 fp4 elements packed into 1 byte, single element `sizeof = 0.5B`; scaleA/scaleB use `fp8_e8m0_t`, single element `sizeof = 1B`.

This sample parameters `M=N=K=8192`, `scaleK=256`, block parameters `baseM=baseN=256`.

Total data read $D_{total}$ is:

$$D_{total} = \frac{N}{baseN} \times M \times K \times 0.5B + \frac{M}{baseM} \times K \times N \times 0.5B + \frac{N}{baseN} \times M \times scaleK \times 1B + \frac{M}{baseM} \times scaleK \times N \times 1B$$

$$= (32 \times 8192 \times 8192 \times 0.5 + 32 \times 8192 \times 8192 \times 0.5 + 32 \times 8192 \times 256 \times 1 + 32 \times 256 \times 8192 \times 1) B$$

$$= (1GB + 1GB + 64MB + 64MB) = 2.125GB$$

**MTE2 Theoretical Duration**:

Ascend 950PR chip: L2Cache peak bandwidth is approximately 5TB/s, HBM (corresponding to GM) bandwidth is approximately 1.6TB/s. Ideally, first access retrieves data from HBM and caches to L2Cache, subsequent accesses read directly from L2Cache.

> **Unit Note**: Bandwidth unit uses decimal system, 1 TB/s = 10^12 B/s.

Data first read from HBM $D_{HBM}$ is:

$$D_{HBM} = M \times K \times 0.5B + K \times N \times 0.5B + M \times scaleK \times 1B + N \times scaleK \times 1B = 32MB + 32MB + 2MB + 2MB = 68MB$$

Data read from L2Cache $D_{L2Cache}$ is:

$$D_{L2Cache} = D_{total} - D_{HBM} = 2.125GB - 68MB \approx 2.057GB$$

MTE2 theoretical duration $T_{MTE2}$ is:

$$T_{MTE2} = \frac{D_{HBM}}{1.6TB/s} + \frac{D_{L2Cache}}{5TB/s} = \frac{68MB}{1.6TB/s} + \frac{2.057GB}{5TB/s} \approx 42.5μs + 411.4μs = 453.9μs$$

MTE2 duration error $E_{MTE2}$ is:

$$E_{MTE2} = \frac{T_{actual} - T_{MTE2}}{T_{MTE2}} = \frac{588.736μs - 453.9μs}{453.9μs} = 29.7\%$$

L2Cache size in Ascend 950PR is 128MB, which cannot cache all input data, and some data will experience L2Cache miss during transfer, requiring retrieval from HBM. Users can further optimize L2Cache splitting strategy to improve MTE2 bandwidth.

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

> **Note**: The `en_dtypes` library used in this sample requires version `0.0.4`. Installation command:

```bash
pip3 install en_dtypes==0.0.4
```

### Configure Environment Variables

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

### Sample Execution

```bash
mkdir -p build && cd build;  # Create and enter build directory
cmake .. -DCMAKE_ASC_RUN_MODE=npu -DCMAKE_ASC_ARCHITECTURES=dav-3510; make -j;
python3 ../scripts/gen_data.py
./demo
python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin
```

When using NPU simulation mode, set `-DCMAKE_ASC_RUN_MODE=sim`.

```bash
cmake .. -DCMAKE_ASC_RUN_MODE=npu -DCMAKE_ASC_ARCHITECTURES=dav-3510; make -j; # NPU mode
cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510; make -j; # NPU simulation mode
```

Build option description:

| Parameter | Available Values | Description |
|------|--------|------|
| `CMAKE_ASC_RUN_MODE` | `npu` / `sim` | Run mode: NPU run, NPU simulation |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | Target SoC architecture |

> **Note**: Before switching `CMAKE_ASC_RUN_MODE` / `CMAKE_ASC_ARCHITECTURES`, you need to clear the CMake cache. You can execute `rm CMakeCache.txt` in the build directory and then re-run cmake.

The execution result shown below indicates the accuracy comparison succeeded.

```bash
test pass!
```

### Performance Analysis

Use the `msprof` tool to obtain detailed performance data:

```bash
msprof ./demo
```

A PROF_ prefixed folder will be generated in the current directory, with `mindstudio_profiler_output` directory storing performance data summary. Performance data analysis is recommended to view files in this directory:

```bash
PROF_xxxx_XXXXXX
├── device_{id}
├── host
├── mindstudio_profiler_log
└── mindstudio_profiler_output
    ├── msprof_*.json
    ├── xx_*.csv
    └── README.txt
```

View specific performance analysis results:

```bash
cat ./PROF_*/mindstudio_profiler_output/op_summary_*.csv
```