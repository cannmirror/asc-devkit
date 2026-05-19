# MatrixTranspose Sample

## Overview

This sample uses matrix transposition as an example to introduce memory access coalescing optimization strategies in Ascend C SIMD and SIMT hybrid programming scenarios. The sample contains 2 kernel versions, starting from direct index transposition, then adjusting data write-back patterns through UB relay to make GM read/write closer to continuous access, thereby demonstrating global memory access optimization methods for matrix transposition in hybrid programming.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```text
├── matrix_transpose
│   ├── CMakeLists.txt         // Compilation project file
│   ├── matrix_transpose.asc   // Matrix transposition sample implementation
│   └── README.md
```

## Sample Description

- Computation Formula:

  $$
  output(x, y) = input(y, x)
  $$

  - input is the input matrix with shape [H,W] and data type float
  - output is the output matrix with shape [W,H] and data type float

- Sample Specifications:
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">MatrixTranspose</td></tr>
  <tr><td rowspan="2" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">[1024,1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">output</td><td align="center">[1024,1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">transpose_naive_kernel / transpose_coalesced_kernel</td></tr>
  </table>

## Sample Implementation

### Case Implementation Description

This sample implements different memory access strategies through two independent kernels, each kernel corresponding to a specific Case version.

| Case   | Implementation Characteristics                                           | Kernel Function Used               | Optimization Feature              |
| ------ | -------------------------------------------------- | -------------------------- | --------------------- |
| Case 0 | Direct output coordinate calculation based on transpose formula, GM continuous read, non-continuous write   | transpose_naive_kernel     | Direct index transposition version      |
| Case 1 | Use UB to temporarily store tile and swap read/write directions, GM read/write closer to continuous access | transpose_coalesced_kernel | UB relay + global memory access coalescing |

#### Thread Block Layout

This sample uses SIMD and SIMT hybrid programming, with kernel functions calling `__simt_vf__` functions internally through `asc_vf_call`. To facilitate comparison of performance differences between versions, this sample only supports square matrices, and matrix width and height need to be integer multiples of 32. All Cases use exactly the same thread block layout:

- This sample flattens 2D tile coordinates to 1D `blockIdx.x`, and flattens 32×32 elements within a tile to 1D `threadIdx.x`.
- Each tile size is 32×32, one block processes one tile. Number of blocks is `bn = grid_x * grid_y`, where `grid_x = matrix_width / TILE_DIM`, `grid_y = matrix_height / TILE_DIM`.

Based on the above partitioning, `blockIdx.x` represents the flattened 1D tile number. By integer division and modulo of `grid_width`, you can restore the 2D tile coordinates `(block_row, block_col)` that the current block is responsible for. `threadIdx.x` represents the flattened 1D element number within the tile. By integer division and modulo of `TILE_DIM`, you can restore the local coordinates `(tile_row, tile_col)` of the element that the thread processes within the tile.

```cpp
int block_row = blockIdx.x / grid_width;
int block_col = blockIdx.x % grid_width;

int tile_row = threadIdx.x / TILE_DIM;
int tile_col = threadIdx.x % TILE_DIM;
```

The figure below shows a more intuitive mapping schematic. The left side shows a 1024×1024 matrix divided into 32×32 tiles. The right side shows one tile that one block needs to process, with size 32×32.

<img src="./figure/blockMapping.png" width="60%">

Taking the green element in the figure as an example, you can directly use the built-in variables in the kernel to derive its input coordinates in GM.

- Tile coordinates are (2,1), corresponding to:
  - block_row = 2
  - block_col = 1
  - blockIdx.x = block_row × grid_width + block_col = 2 × 32 + 1 = 65
- Element's local coordinates within the tile are (2,29), corresponding to:
  - threadIdx.x = tile_row × TILE_DIM + tile_col = 2 × 32 + 29 = 93

Combining the index calculation in the code:

```cpp
int input_row = block_row * TILE_DIM + tile_row;
int input_col = block_col * TILE_DIM + tile_col;
int index_in = input_col + width * input_row;
```

Substituting the variable values in this example:

- block_row = 65 / 32 = 2
- block_col = 65 % 32 = 1
- tile_row = 93 / 32 = 2
- tile_col = 93 % 32 = 29
- input_row = block_row × TILE_DIM + tile_row = 2 × 32 + 2 = 66
- input_col = block_col × TILE_DIM + tile_col = 1 × 32 + 29 = 61

Therefore, the input element coordinates processed by this thread in GM are:

- input[input_row,input_col] = input[66,61]

If continuing to substitute the 1D address formula:

- index_in = input_col + width × input_row = 61 + 1024 × 66

### Performance Metrics Description

| Metric                | Description                                                                                      |
| ------------------- | ----------------------------------------------------------------------------------------- |
| Task Duration(us)   | Task total latency, including time to schedule to accelerator, execution time on accelerator, and response end time                  |
| aiv_time(us)        | Task theoretical execution time on AI Vector Core, unit is us                                            |
| aiv_total_cycles    | After this Task is assigned to each AI Vector Core computation unit, the total execution cycles on each AI Vector Core computation unit |
| aiv_vec_time(us)    | vec type instruction (vector computation instruction) latency, unit is us                                               |
| aiv_vec_ratio       | Ratio of vec type instruction (vector computation instruction) cycles in total cycles                           |
| aiv_scalar_time(us) | scalar type instruction (scalar computation instruction) latency, unit is us                                            |
| aiv_scalar_ratio    | Ratio of scalar type instruction (scalar computation instruction) cycles in total cycles                        |

Except for Task Duration, all other metrics in this example show the average of performance metrics across all blocks.

### Case 0: Direct Index Transposition Version

**Sample Objective**: Implement basic matrix transposition functionality as the latency comparison baseline for subsequent optimized versions

**Core Implementation**:

- Each block processes one 32×32 tile
- Each SIMT thread processes 1 element within the tile
- Thread first reads input element from GM by original coordinates, then calculates the output position after transposition for this element, and writes data directly to the transposed GM address
- GM read direction is continuous, GM write-back direction is not continuous

The figure below shows the data flow of Case 0, with red highlighting elements processed by one Warp when reading GM and writing GM. For threads in the same Warp, they read a row of elements in the tile from GM input and write back to a column of the tile in GM output. When reading GM input, element addresses accessed by adjacent threads are continuous, which is continuous read. When writing back to output, adjacent threads are scattered to different rows of the output matrix, which is non-continuous write. Therefore, the core issue of this version is that the write-back address after transposition is no longer continuous, which usually significantly affects overall throughput.

<img src="./figure/case0.png" width="60%">

**Key Code**:

```cpp
int index_in = input_col + width * input_row;
int index_out = input_row + height * input_col;

output[index_out] = input[index_in];
```

**Performance Data**:

| Task Duration(us) | aiv_time(us) | aiv_total_cycles | aiv_vec_time(us) | aiv_vec_ratio | aiv_scalar_time(us) | aiv_scalar_ratio |
| :---------------: | :----------: | :--------------: | :--------------: | :-----------: | :-----------------: | :--------------: |
|      70.304      |    4.257    |     7024.463     |      4.115      |     0.967     |        0.131        |      0.030      |

**Analysis**:

- Case 0's Task Duration is 70.304us, as the direct index transposition version, it serves as the comparison baseline for subsequent optimized versions
- This version's GM read is still continuous read, but GM write-back is cross-row, non-continuous write. Write requests from the same Warp are difficult to efficiently merge, so overall latency is mainly limited by the write-back memory access pattern

---

### Case 1: UB Relay + Global Memory Access Coalescing Transposition Version

**Optimization Objective**: Adjust transposition write-back pattern through UB relay to make GM read/write closer to continuous access, reduce end-to-end latency

**Core Implementation**:

- Each thread reads 1 element from tile in GM, one Warp reads one row of elements from a tile
- Write elements to tile in UB by original coordinates, one Warp writes one read row to one row of tile in UB
- After synchronization, each thread fetches data from UB, one Warp reads one column of elements from tile in UB
- Write fetched values back to transposed position in GM, one Warp writes one read column of elements from UB to one row of tile in GM output

The figure below shows the data flow of Case 1, with red and yellow highlighted elements showing elements processed by one Warp's threads when reading GM and writing GM. When reading GM input, the entire tile is moved to UB according to GM layout. When writing GM output, one Warp's threads read one column of elements from UB and write back to their corresponding transposed positions.

<img src="./figure/case1.png" width="60%">

Unlike Case 0, in Case 0 threads "directly write input elements to transposed GM positions", so adjacent threads are scattered to different rows of the output matrix. In Case 1, threads first put data into UB, transferring the original non-continuous global write access to non-continuous read within UB. Therefore, the core benefit of this version is: although it adds one UB read/write and one synchronization, it gains a "read continuous, write also continuous" access pattern on the GM side, and overall latency is usually significantly lower than Case 0.

**Key Code**:

```cpp
int tile_row = threadIdx.x / TILE_DIM;
int tile_col = threadIdx.x % TILE_DIM;

tile[tile_row][tile_col] = input[index_in];
asc_syncthreads();

int block_row = blockIdx.x / grid_width;
int block_col = blockIdx.x % grid_width;

int output_row = block_col * TILE_DIM + tile_row;
int output_col = block_row * TILE_DIM + tile_col;
int index_out = output_col + output_row * height;

output[index_out] = tile[tile_col][tile_row];
```

**Optimization Methods**:

- Use UB as tile relay area, transfer non-continuous GM write in Case 0 to UB-side access
- Adjust output tile block coordinates to make write-back to GM closer to row-continuous write for the same Warp
- Use `asc_syncthreads()` to ensure the entire tile is written to UB before executing transposition-direction read

**Performance Data**:

| Task Duration(us) | aiv_time(us) | aiv_total_cycles | aiv_vec_time(us) | aiv_vec_ratio | aiv_scalar_time(us) | aiv_scalar_ratio |
| :---------------: | :----------: | :--------------: | :--------------: | :-----------: | :-----------------: | :--------------: |
|      44.440      |    2.441    |     4027.577     |      2.295      |     0.941     |        0.134        |      0.054      |

**Analysis**:

- Compared to Case 0's direct index transposition version, Case 1's Task Duration decreased from 70.304us to 44.440us, a latency reduction of about 36.8%
- Calculated by Task Duration, Case 1's overall performance is about 1.58x that of Case 0, indicating that after improving GM write-back continuity through UB relay, end-to-end latency has significant improvement
- Case 1 still requires additional UB read/write and synchronization, so the optimized Task Duration did not drop to the ideal level of pure continuous memory access; this latency is the necessary overhead brought by UB relay in exchange for GM memory access coalescing

---

## Performance Comparison Summary

### Ascend 950PR Performance Data

**Overall Optimization Effect**:

- Through memory access coalescing optimization from Case 0 to Case 1, sample Task Duration decreased from 70.304us to 44.440us, a latency reduction of about 36.8%
- Case 1 achieves about 1.58x performance improvement relative to Case 0, indicating that after improving GM write-back continuity through UB relay, end-to-end latency has significant benefit

| Case version | Task Duration(us) | End-to-end Latency Relative to Case 0 | Optimization Point                           |
| ------------ | ----------------- | -------------------- | -------------------------------- |
| Case 0       | 70.304            | **1x**         | Direct index transposition, GM continuous read, non-continuous write |
| Case 1       | 44.440            | **1.58x**      | UB relay, global memory access coalescing             |

## Tuning Suggestions

1. **Prioritize GM Memory Access Continuity**: Matrix transposition has very small computation volume, end-to-end latency is mainly affected by read/write memory access patterns.
2. **Use UB Relay to Improve Write-back Pattern**: When direct transposition causes GM non-continuous write, non-continuous access can be transferred to UB side in exchange for continuous read/write on GM side.
3. **Note Synchronization Overhead**: UB relay requires synchronization to ensure tile data integrity. When optimizing, need to simultaneously consider the balance between memory access benefit and synchronization, UB read/write overhead.

## Compilation and Execution

Execute the following steps in the root directory of this sample to compile and run the sample.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment.

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
  SCENARIO_NUM=0                       # Select execution scenario, options 0-1
  mkdir -p build && cd build;          # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM ..;make -j;  # Compile project
  ./demo                               # Execute sample
  ```

- Compilation Options Description

  | Option             | Available Values      | Description              |
  | ---------------- | ----------- | ----------------- |
  | `SCENARIO_NUM` | `0`-`1` | Sample type, default is 0 |

  The execution result is as follows, indicating that the accuracy comparison passed.


  ```text
  [Success] Case accuracy is verification passed.
  ```

## Performance Analysis

Use the `msprof` tool to obtain detailed performance data:

```bash
msprof op ./demo   # Analyze case performance
```

A folder named "OPPROF_{timestamp}_XXX" will be generated in the default directory. The performance data folder structure is as follows:

```text
├──dump                       # Raw performance data, users do not need to focus on
├──ArithmeticUtilization.csv  # cube/vector instruction cycle ratio
├──L2Cache.csv                # L2 Cache hit rate
├──Memory.csv                 # UB, L1 and main memory read/write bandwidth rate
├──MemoryL0.csv               # L0A, L0B, and L0C read/write bandwidth rate
├──MemoryUB.csv               # Vector and Scalar to UB read/write bandwidth rate
├──OpBasicInfo.csv            # Operator basic information
├──PipeUtilization.csv        # Collection of computation unit and transfer unit latency and ratio
├──ResourceConflictRatio.csv  # Ratio of bank group, bank conflict and resource conflict on UB in all instructions
└──visualize_data.bin         # MindStudio Insight presentation file
```