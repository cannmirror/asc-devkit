# MxFP4 Matmul Performance Tuning Sample

## Overview

This sample uses MxFP4 matrix multiplication as an example to introduce MxMatmul performance tuning methods based on Ascend C `Matmul` high-level API. The sample contains two scenarios (Case 1-2), both using constant tiling, using template constant `MatmulApiStaticTiling` (static tiling) instead of runtime tiling copy and computation.

**Optimization Path**:
- Case 1: Multi-core MDL constant tiling (scaleA/B transfers synchronously with A/B)
- Case 2: Multi-core MDL constant tiling (in **GM→L1 transfer**, scaleA/B transfers multiple times relative to A/B, `mxTypePara`)

## Supported Products

- Ascend 950PR / Ascend 950DT

## Directory Structure

```
├── matmul_mxfp4_high_performance
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script file
│   │   └── verify_result.py    // Golden value comparison file
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data read/write functions
│   ├── matmul_mx.asc           // Ascend C sample implementation (contains 2 optimization cases)
│   └── matmul_mx.h             // Sample header file (static Tiling template and kernel implementation)
```

## Sample Description

### Sample Function

The sample implements MxFP4 matrix multiplication with a fixed shape of 8192×8192 (with scale quantization coefficient input).

### MxMatmul Introduction

MxMatmul (Matrix Multiply with Scale) is a **matrix multiplication with quantization scaling coefficients**, an extension capability of Ascend C Matmul API in MX (Mixed-Precision) quantization scenarios. Compared with basic Matmul, the core difference of MxMatmul lies in the introduction of **scale input**.

#### Computation Formula

$$
C = (\text{scaleA} \otimes A) \times (\text{scaleB} \otimes B)
$$

Where $\otimes$ represents broadcast multiplication, when left/right matrix multiplies with left/right quantization coefficient matrix, every 32 elements in K direction share one quantization factor.

#### Parameter Description

| Input | Name | Shape | Data Type | Data Layout Type | Description |
|------|------|------|----------|--------------|------|
| A | Left matrix | [8192, 8192] | `fp4x2_e1m2_t` | `ND` | MX FP4 left matrix |
| scaleA | Left quantization coefficient matrix | [8192, 256] | `fp8_e8m0_t` | `ND` | A matrix scaling factor matrix, every 32 elements in A matrix K direction share one scaling factor |
| B | Right matrix | [8192, 8192] | `fp4x2_e1m2_t` | `ND` | MX FP4 right matrix |
| scaleB | Right quantization coefficient matrix | [256, 8192] | `fp8_e8m0_t` | `ND` | B matrix scaling factor matrix, every 32 elements in B matrix K direction share one scaling factor |
| C | Output | [8192, 8192] | `bfloat16_t` | `ND` | Computation result |

  <img src="figure/MxMatmul.png">

#### Four-Path Input Explanation

- In the sample `sK = ceil(K / 64) * 2`, when `K=8192`, `sK=256`
- Therefore `scaleA` shape is `[M, sK] = [8192, 256]`, `scaleB` shape is `[sK, N] = [256, 8192]`
- `scale` ND needs special explanation: `scaleA` writes in conventional row-major `[M, sK]`; `scaleB` disk write order is equivalent to `[sK/2, N, 2]`, meaning first continuous `2 Byte` in K direction, then advances along N direction. The `ND` layout of four-path inputs is shown in the figure below:
  <img src="figure/NDformat.png">

- The transfer of four-path inputs is shown in the figure below:

  <img src="figure/InputOfMxMatmul.png">

## Sample Implementation

### Implementation Key Points

This sample unifies tiling parameters in `matmul_mx.h` for compile-time determination, passing to `MatmulImpl` through template constant `CONSTANT_CFG`:

```cpp
constexpr static auto CONSTANT_CFG = GetMxConstantCFG<aType, bType, cType, EnableScaleCache>();
AscendC::Matmul<aType, bType, cType, cType, CONSTANT_CFG,
                    AscendC::MatmulCallBackFunc<nullptr, nullptr, nullptr>,
                    AscendC::Impl::Detail::MatmulWithScalePolicy>
    matmulObj;
REGIST_MATMUL_OBJ(pipe, GetSysWorkSpacePtr(), matmulObj, (TCubeTiling*)nullptr);
```

Explanation:
- Kernel side does not do `TCubeTiling` runtime copy/computation.
- `SCENARIO_NUM` only determines template instantiation: `case1 -> MatmulKernel<false>`, `case2 -> MatmulKernel<true>`.

### Case1 and Case2 Difference

Both scenarios use constant tiling, with consistent L1 parameters: `depthA1/depthB1=4`, `stepKa/stepKb=2`, `stepM/stepN=1`, `dbL0A/dbL0B=2`.
The only difference is `mxTypePara`:

| Scenario | `mxTypePara` | Semantic |
|------|--------------|------|
| Case 1 (`SCENARIO_NUM=1`) | `CASE1_MX_TYPE_PARA = 0x01010101` | scaleA/B transfers synchronously with A/B |
| Case 2 (`SCENARIO_NUM=2`) | `CASE2_MX_TYPE_PARA = 0x01010404` | scaleA/B transfers more relative to A/B in K direction |

`mxTypePara` definition:

- In MxMatmul, you can set mxTypePara to control the loading ratio of Scale matrix and matrices A, B in L1.
- **MX Scale Scaling Factor**: `scaleFactorKa=4` means scaleA data loading ratio in K direction is 4 times of A matrix; `scaleFactorKb=4` means scaleB data loading ratio in K direction is 4 times of B matrix

    - **mxTypePara**: Composite parameter, used in MxMatmul scenarios, represents the multiple of scaleA/scaleB loaded into L1 relative to A/B matrices loaded into L1:
      - **bit [0:6]** `scaleFactorKa`: scaleA and A matrix K direction loaded data amount ratio coefficient, range [1, 127]
      - **bit [8:14]** `scaleFactorKb`: scaleB and B matrix K direction loaded data amount ratio coefficient, range [1, 127]
      - **bit [16:22]** `scaleFactorM`: scaleA and A matrix M direction loaded data amount ratio coefficient, range [1, 127]
      - **bit [24:30]** `scaleFactorN`: scaleB and B matrix N direction loaded data amount ratio coefficient, range [1, 127]
    - Usage constraints:
      - Only when Ka direction fully loaded (`baseK * stepKa * scaleFactorKa >= singleCoreK`), can set `scaleFactorM > 1`
      - Only when Kb direction fully loaded (`baseK * stepKb * scaleFactorKb >= singleCoreK`), can set `scaleFactorN > 1`
      - scaleA, scaleB loaded data amount in M, N, K directions cannot exceed actual size
      - This parameter only takes effect in MDL mode

### Parameter Setting and Transfer Data Amount Calculation

The following statistics transfer path is **GM->L1**, calculated based on current fixed parameters:

| Parameter | Value |
|------|----|
| `M` | `8192` |
| `N` | `8192` |
| `K` | `8192` |
| `singleCoreM` | `2048` |
| `singleCoreN` | `1024` |
| `singleCoreK` | `8192` |
| `baseM` | `256` |
| `baseN` | `256` |
| `baseK` | `256` |
| `stepKa` | `2` |
| `stepKb` | `2` |
| `scaleFactorKa (case1)` | `1` |
| `scaleFactorKb (case1)` | `1` |
| `scaleFactorKa (case2)` | `4` |
| `scaleFactorKb (case2)` | `4` |
| `Data Type` | A/B: `fp4x2` (`0.5 Byte/elem`) |
| `Data Type` | scale: `fp8` (`1 Byte/elem`) |

**Explanation**:

A/B base block size: `baseM * baseK * 0.5 = 256 * 256 * 0.5 = 32,768 B = 32 KB`, scaleA/scaleB base block size: `256 * (256/32) * 1 = 2,048 B = 2 KB`.

case1 single GM→L1 transfer amount:

- A: `stepM * stepKa = 1 * 2 = 2` base blocks, byte amount `2 * 32 = 64 KB`
- B: `stepN * stepKb = 1 * 2 = 2` base blocks, byte amount `2 * 32 = 64 KB`
- scaleA: `stepM * stepKa * scaleFactorKa = 1 * 2 * 1 = 2` base blocks, `4 KB`
- scaleB: `stepN * stepKb * scaleFactorKb = 1 * 2 * 1 = 2` base blocks, `4 KB`
- **Total: `64 + 64 + 4 + 4 = 136 KB`**

> **Note**: `dbL0A/dbL0B=2` means L1→L0 uses double buffer (while L0 computes current portion, next portion is ready from L1), so **L1 needs to simultaneously accommodate `136 × 2 = 272 KB`** data, but GM→L1 each MTE2 transfer amount is still `136 KB`.

case2 single GM→L1 transfer amount:
- A/B same as case1: each `64 KB` (total `128 KB`)
- scaleA: `1 * 2 * 4 = 8` base blocks, `16 KB`
- scaleB: `1 * 2 * 4 = 8` base blocks, `16 KB`
- **Total MTE2 each time: `64 + 64 + 16 + 16 = 160 KB`**

> Similarly, due to `dbL0A/dbL0B=2`, **L1 resident total is `160 × 2 = 320 KB`**.

Case1/Case2 difference in scale side mainly reflects in "single transfer granularity and transfer count":

- Case1: Each small amount transfer, scale transfer count approximately `16` times (`8192/512`)
- Case2: Each large amount transfer, scale transfer count approximately `4` times (`8192/2048`)
- In this shape, both have the same scale theoretical total byte amount, but Case2 has fewer batches, larger reuse window, more beneficial for reducing MTE2 duration.



## Performance Comparison Summary

### Ascend 950PR Chip Performance Data
| Case version | Task Duration(μs) | Block Num | aicore_time(μs) | aic_mac_time(μs) | aic_mac_ratio | aic_scalar_time(μs) | aic_scalar_ratio | aic_mte1_time(μs) | aic_mte1_ratio | aic_mte2_time(μs) | aic_mte2_ratio | aic_fixpipe_time(μs) | aic_fixpipe_ratio |
|------|------------------|-----------|----------------|-----------------|---------------|-------------------|-----------------|------------------|----------------|------------------|----------------|--------------------|-------------------|
| Case 1 | 750.219 | 32 | 749.13 | 660.15 | 0.881 | 258.354 | 0.345 | 437.64 | 0.584 | 753.906 | 0.982 | 33.257 | 0.044 |
| Case 2 | 693.283 | 32 | 692.34 | 641.444 | 0.926 | 241.563 | 0.349 | 428.914 | 0.62 | 612.536 | 0.885 | 33.965 | 0.049 |

Case 2 has reached `92.6%` of theoretical peak performance (the `aic_mac_ratio` in the table).

### Case 2 Gain (Relative to Case 1)

Both scenarios use constant tiling + template constantization. Case 2 gain relative to Case 1 mainly comes from `mxTypePara` enabling scale multi-transfer capability.

- End-to-end latency: `750.219 -> 693.283 μs`, reduced `56.936 μs`, gain `7.59%`.
- MTE2 absolute duration: `753.906 -> 612.536 μs`, reduced `141.370 μs`, gain `18.75%`.
- MTE2 ratio: `0.982 -> 0.885`, decreased `9.7%`.
- MAC ratio: `0.881 -> 0.926`, improved `4.5%`.

**Tuning Tips**:
> MX Matmul key difference lies in `scale` and A/B transfer can be decoupled; when `aic_mte2_ratio` is high, prioritize adjusting `scale` transfer ratio through `mxTypePara` to improve L1 reuse, reduce repeated GM->L1 transfer.


### Theoretical Performance Comparison
This sample performance data was obtained on Ascend 950PR, with processor main frequency of 1.65GHz. For MX-FP4 data type, it processes 16×64×16 multiply-add operations per cycle. Cube theoretical computation time is
$$
T_{\text{theory}} = \frac{M \times N \times K}{16 \times 64 \times 16 \times 1.65 \times 10^9 \times \text{core_num}} = \frac{8192 \times 8192 \times 8192}{4096 \times 1.65 \times 10^9 \times 32} = 635.5 μs
$$
Case 1/Case 2 `aic_mac_time` are `660.150 μs` / `641.444 μs` respectively, relative to theoretical value `635.5 μs`:
- Case 1 error: `(660.150 - 635.5) / 635.5 = 3.88%`
- Case 2 error: `(641.444 - 635.5) / 635.5 = 0.94%`

## Build and Run

- Build and Execute

Execute the following steps in the sample root directory to build and run the sample:

> **Note**: The `en_dtypes` library used in this sample requires version `0.0.4`. Installation command:

  ```bash
  pip3 install en_dtypes==0.0.4
  ```

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
  SCENARIO_NUM=2
  mkdir -p build && cd build;  # Create and enter build directory
  cmake .. -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=npu -DCMAKE_ASC_ARCHITECTURES=dav-3510; make -j;
  python3 ../scripts/gen_data.py
  ./demo
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin
  ```

  When using NPU simulation mode, set `-DCMAKE_ASC_RUN_MODE=sim`
  ```bash
  cmake .. -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=npu -DCMAKE_ASC_ARCHITECTURES=dav-3510; make -j; # npu mode
  cmake .. -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510; make -j; # npu simulation mode
  ```

  Build option description:

  | Parameter | Available Values | Description |
  |------|--------|------|
  | `SCENARIO_NUM` | `1` / `2` | 1: Constant tiling + scale synchronous transfer; 2: Constant tiling + scale multi-transfer |
  | `CMAKE_ASC_RUN_MODE` | `npu` (default) / `sim` | Run mode: NPU run, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | Target SoC architecture (this sample only supports 3510) |

  > **Note:** Before switching `CMAKE_ASC_RUN_MODE` / `CMAKE_ASC_ARCHITECTURES` / `SCENARIO_NUM`, need to clear cmake cache. Can execute `rm CMakeCache.txt` in build directory and then re-run cmake.
  

  The execution result shown below indicates the accuracy comparison succeeded.
  ```bash
  test pass!
  ```

## Performance Analysis

Use the `msprof` tool to obtain detailed performance data:

```bash
msprof ./demo   # Analyze performance
```

A PROF_ prefixed folder will be generated in the current directory, with `mindstudio_profiler_output` directory storing Host and various Device performance data summary. Performance data analysis is recommended to view files in this directory

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
View specific performance analysis results:
```
# View Task Duration and various data
cat ./PROF_*/mindstudio_profiler_output/op_summary_*.csv
```