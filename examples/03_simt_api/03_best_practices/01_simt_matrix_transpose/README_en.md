# MatrixTranspose Performance Tuning Example

## Overview

This example uses matrix transpose to demonstrate memory access optimization strategies in Ascend C SIMT programming. The example includes one matrix copy baseline version and three progressively optimized transpose kernel versions, starting from direct index transpose and gradually introducing UB staging, global memory access coalescing, and padding to reduce UB Bank conflicts, demonstrating the tuning path for matrix transpose in SIMT programming.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```text
├── 01_simt_matrix_transpose
│   ├── CMakeLists.txt         // Build project file
│   ├── matrix_transpose.asc   // SIMT matrix transpose example implementation
│   └── README.md
```

## Example Description

- Calculation formula:

  $$
  output(x, y) = input(y, x)
  $$

  - input is the input matrix with shape [H,W] and data type float
  - output is the output matrix with shape [W,H] and data type float
- Example specification:

  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center">MatrixTranspose</td></tr>
  <tr><td rowspan="2" align="center">Example Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">[1024,1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Example Output</td><td align="center">output</td><td align="center">[1024,1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">copy_kernel / transpose_naive_kernel / transpose_coalesced_kernel / transpose_avoid_bank_conflicts_kernel</td></tr>
  </table>

## Example Implementation

### Case Implementation Description

This example implements different memory access strategies through 4 independent kernels, each corresponding to a specific Case version.

| Case   | Implementation Characteristics                                           | Kernel Used                          | Optimization Features                                |
| ------ | -------------------------------------------------- | ------------------------------------- | --------------------------------------- |
| Case 0 | Read and write by same coordinates, no transpose, GM sequential read and sequential write       | copy_kernel                           | Matrix copy version (baseline)                    |
| Case 1 | Directly calculate output coordinates according to transpose formula                       | transpose_naive_kernel                | Direct index transpose version                        |
| Case 2 | Stage tile in UB and swap read/write directions, GM read/write closer to sequential access | transpose_coalesced_kernel            | UB staging + global memory access coalescing for transpose           |
| Case 3 | Add padding in UB tile to reduce bank conflicts                 | transpose_avoid_bank_conflicts_kernel | UB staging + global memory access coalescing + avoid UB Bank conflicts |

#### Thread Block Layout

To facilitate comparison of performance differences between versions, this example only supports square matrices, and the matrix width and height must be multiples of 32. All Cases use the same thread block layout:

- For 2D regular computations like matrix transpose, you can decompose a large matrix into several local tile sub-blocks, so that each block only processes one local region, facilitating index calculation and thread cooperation.
- Each tile size is 32x32, and one block processes one tile. The grid configuration is `(matrix_width/32, matrix_height/32, 1)`.
- The block configuration is fixed at `(32, 32, 1)`, with a total of 32x32=1024 threads per block, and each thread processes only 1 element within the tile.
- The `threadIdx.x` direction has 32 threads, corresponding to the column index within the tile; the `threadIdx.y` direction has 32 threads, corresponding to the row index within the tile.

Based on the above tiling method, `blockIdx` locates the tile that the current block is responsible for, and `threadIdx` locates the element that the current thread processes within the tile. By adding the tile coordinates and local coordinates within the tile, you can get the global coordinates `x_index` and `y_index` of the element in the original matrix; then expand according to row-major layout to get the linear index `index` in GM.

```cpp
int x_index = blockIdx.x * TILE_DIM + threadIdx.x;
int y_index = blockIdx.y * TILE_DIM + threadIdx.y;
int index = x_index + width * y_index;
```

The figure below shows a more intuitive mapping diagram. The left side shows a 1024x1024 matrix divided into 32x32 tiles. The right side shows one tile that one block needs to process, with a size of 32x32.

<img src="./figure/blockMapping.png" width="60%">

Taking the green element in the figure as an example, you can directly use the built-in variables in the kernel to derive its input coordinates in GM.

- The tile coordinates of the green element in the figure are (2,1), corresponding to:
  - blockIdx.y = 2
  - blockIdx.x = 1
- The local coordinates of the green element within the tile are (2,29), corresponding to:
  - threadIdx.y = 2
  - threadIdx.x = 29

Substituting the variable values in this example:

- x_index = blockIdx.x x TILE_DIM + threadIdx.x = 1 x 32 + 29 = 61
- y_index = blockIdx.y x TILE_DIM + threadIdx.y = 2 x 32 + 2 = 66

Therefore, the input element coordinates that this thread processes in GM are:

- input[y_index,x_index] = input[66,61]

If you continue to substitute into the 1D address formula:

- index = x_index + width x y_index = 61 + 1024 x 66

### Performance Metric Description

| Metric                | Description                                                                                      |
| ------------------- | ----------------------------------------------------------------------------------------- |
| Task Duration(us)   | Overall Task duration, including time to schedule to accelerator, execution time on accelerator, and response completion time                  |
| aiv_time(us)        | Theoretical execution time of Task on AI Vector Core, in us                                            |
| aiv_total_cycles    | Total execution cycles on each AI Vector Core compute unit after the Task is assigned to each AI Vector Core compute unit |
| aiv_vec_time(us)    | vec type instruction (vector computation instruction) duration, in us                                               |
| aiv_vec_ratio       | Ratio of vec type instruction (vector computation instruction) cycles in total cycle number                           |
| aiv_scalar_time(us) | scalar type instruction (scalar computation instruction) duration, in us                                            |
| aiv_scalar_ratio    | Ratio of scalar type instruction (scalar computation instruction) cycles in total cycle number                        |

Except for Task Duration, all other metrics in this example show the average value of performance metrics across all blocks.

### Case 0: Matrix Copy Version

**Example Objective**: Establish a time baseline for the matrix copy scenario to provide a performance reference for subsequent transpose versions.

**Core Implementation**:

- Each block processes one 32x32 tile
- Each thread processes 1 element within the tile
- The block locates the current tile according to `blockIdx`, and the thread locates one element within the tile according to `threadIdx`
- The thread reads `input[index]` from GM and directly writes back to `output[index]`

Throughout the process, there is no coordinate exchange and no UB participation. This version does not involve matrix transpose. Adjacent threads in the same Warp have consistent GM read/write directions and sequential memory access patterns, so it can serve as the time baseline for subsequent transpose versions.

**Key Code**:

```cpp
int x_index = blockIdx.x * TILE_DIM + threadIdx.x;
int y_index = blockIdx.y * TILE_DIM + threadIdx.y;
int index = x_index + width * y_index;

output[index] = input[index];
```

**Performance Data**:

| Task Duration(us) | aiv_time(us) | aiv_total_cycles | aiv_vec_time(us) | aiv_vec_ratio | aiv_scalar_time(us) | aiv_scalar_ratio |
| :---------------: | :----------: | :--------------: | :--------------: | :-----------: | :-----------------: | :--------------: |
|      24.777      |    1.054    |     1739.820     |      0.889      |     0.847     |        0.153        |      0.141      |

**Analysis**:

- Case 0 has a Task Duration of 24.777us, serving as the time baseline for sequential GM read/write scenarios
- Subsequent transpose versions need to complete coordinate exchange while approaching this baseline

---

### Case 1: Direct Index Transpose Version

**Optimization Objective**: Implement the most direct matrix transpose function and observe the time change caused by directly writing back to the transposed address.

**Core Implementation**:

- Each block processes one 32x32 tile
- Each thread processes 1 element within the tile
- The thread first reads the input element from GM according to the original coordinates, then calculates the output position after transposing this element, and writes the data directly to the transposed GM address
- GM read direction is sequential, GM write direction is non-sequential

The figure below shows the data flow of Case 1, with red marking the elements processed by one Warp when reading GM and writing GM. Threads of the same Warp will read a row of elements from the GM input tile and write back to a column of the GM output tile. When reading GM input, adjacent threads access consecutive element addresses, which is sequential read. When writing back to output, adjacent threads are scattered to different rows of the output matrix, which is non-sequential write. Therefore, the core issue of this version is that the transposed write-back address is no longer sequential, which typically significantly affects overall throughput.

<img src="./figure/case1.png" width="60%">

**Key Code**:

```cpp
int index_in = x_index + width * y_index;
int index_out = y_index + height * x_index;

output[index_out] = input[index_in];
```

**Performance Data**:

| Task Duration(us) | aiv_time(us) | aiv_total_cycles | aiv_vec_time(us) | aiv_vec_ratio | aiv_scalar_time(us) | aiv_scalar_ratio |
| :---------------: | :----------: | :--------------: | :--------------: | :-----------: | :-----------------: | :--------------: |
|      60.477      |    3.516    |     5801.925     |      3.357      |     0.955     |        0.147        |      0.041      |

**Analysis**:

- Compared with the Case 0 copy baseline, Task Duration increased from 24.777us to 60.477us, approximately 2.44 times the copy version
- Direct index transpose itself has very small computation, but the transposed GM write-back becomes cross-row, non-sequential access, so the end-to-end time is significantly higher than the copy baseline
- In this version, GM read is still sequential read, but GM write-back address is non-sequential, and write requests from the same Warp are difficult to efficiently coalesce, which is the main reason for the increase in Task Duration

---

### Case 2: UB Staging + Global Memory Access Coalescing Transpose Version

**Optimization Objective**: Adjust transpose write-back method through UB staging to make GM read/write closer to sequential access and reduce end-to-end time.

**Core Implementation**:

- Each thread reads 1 element from the tile in GM, one Warp will read one row of elements from a tile
- Write elements to the tile in UB according to original coordinates, one Warp will write one read row to one row in the UB tile
- After synchronization, each thread fetches data from UB, one Warp will read one column of elements from the UB tile
- Write the fetched values back to the transposed position in GM, one Warp will write one column of elements read from UB to one row in the GM output tile

The figure below shows the data flow of Case 2, with red and yellow marked elements showing the elements processed by threads of one Warp when reading GM and writing GM. When reading GM input, the entire tile is moved to UB according to GM layout. When writing GM output, threads of one Warp will read one column of elements from UB and write back to their corresponding transposed positions.

<img src="./figure/case2.png" width="60%">

Unlike Case 1, in Case 1 threads "directly write input elements to the transposed GM position", so adjacent threads are scattered to different rows of the output matrix. In Case 2, threads first put data into UB, transferring the originally non-sequential global write access to non-sequential read within UB. Therefore, the core benefit of this version is: although one UB read/write and one synchronization are added, it achieves "sequential read and sequential write" access pattern on the GM side, and the overall time is usually significantly lower than Case 1.

**Key Code**:

```cpp
tile[threadIdx.y][threadIdx.x] = input[index_in];
asc_syncthreads();

x_index = blockIdx.y * TILE_DIM + threadIdx.x;
y_index = blockIdx.x * TILE_DIM + threadIdx.y;
int index_out = x_index + y_index * height;

output[index_out] = tile[threadIdx.x][threadIdx.y];
```

**Optimization Methods**:

- Use UB as tile staging area, transfer the non-sequential GM write in Case 1 to UB-side access
- Swap output tile block coordinates to make the same Warp closer to sequential row-wise write when writing back to GM
- Use `asc_syncthreads()` to ensure the entire tile is written to UB before executing transpose-direction read

**Performance Data**:

| Task Duration(us) | aiv_time(us) | aiv_total_cycles | aiv_vec_time(us) | aiv_vec_ratio | aiv_scalar_time(us) | aiv_scalar_ratio |
| :---------------: | :----------: | :--------------: | :--------------: | :-----------: | :-----------------: | :--------------: |
|      35.945      |    1.814    |     2993.315     |      1.646      |     0.910     |        0.156        |      0.083      |

**Analysis**:

- Compared with the naive transpose of Case 1, Task Duration decreased from 60.477us to 35.945us, a time reduction of about 40.6%, overall performance improvement of about 1.68 times
- Case 2 transfers the non-sequential GM write in Case 1 to UB-side access through UB staging, making both GM read and write closer to sequential access, so Task Duration is significantly reduced
- Compared with the Case 0 copy baseline, Case 2's Task Duration is still about 45.1% higher. This gap mainly comes from extra UB read/write, synchronization, and transpose-direction UB access overhead

---

### Case 3: UB Staging + Global Memory Access Coalescing + Avoid UB Bank Conflict Transpose Version

**Optimization Objective**: On the basis of the global memory access coalescing version, reduce bank conflicts in the transpose read phase through UB padding.

**Core Implementation**:

- GM to UB phase is exactly the same as Case 2
- The difference is only in UB layout, changed from 32x32 to 32x33
- After synchronization, writing back from UB to GM is exactly the same as Case 2
- This version does not change the algorithm path, nor does it change block and thread tiling, it only adjusts the physical layout in UB

Taking the UB division rule of Ascend 950PR/Ascend 950DT as an example, the following explains how Bank Conflict is generated in this example, and the theoretical conflict intensity difference between Case 2 and Case 3.

The bank division in UB is shown in the figure below. The total UB size is 256KB, which can be viewed as two rows, each row 128KB. The first 128KB corresponds to bank0 to bank7, and the second 128KB corresponds to bank8 to bank15. bank0 and bank8 belong to the same group, bank1 and bank9 belong to the same group, and so on.

<img src="./figure/bank结构示意图.png">

For SIMT, the most critical issue is whether concurrent threads of the same Warp under the same UB access instruction will concentrate on accessing a few banks.

- **Read-write conflict**: A read operation and a write operation simultaneously attempt to access the same bank.
- **Write-write conflict**: Multiple write operations simultaneously attempt to access the same bank group.
- **Read-read conflict**: Two read operations simultaneously attempt to access the same bank, or more than two read operations simultaneously attempt to access the same bank group.

Since the tile in this example is very small:

- `32x32x4B = 4096B`
- `32x33x4B = 4224B`

Much smaller than 128KB, so one tile usually only falls in the first 128KB area. In this analysis, you can approximate it as only using `bank0~bank7`.

In Case 2, the layout of the first 10 rows of the UB tile array according to row-major storage is shown below. For display convenience, only the first 10 rows of elements are shown here, and the rest follow the same pattern. Each row of 32 float data will be stored across exactly 4 banks, with the first element of each row marked in blue. In Case 2, threads of one Warp will read one column of elements from the tile and write back to GM output. When accessing UB, 32 threads will concentrate on accessing two banks, meaning one bank will have 16 threads accessing simultaneously, generating a large number of read conflicts.

<img src="./figure/case2bank.png">

In Case 3, one column of padding is added to the tile array in UB, changing from 32 elements per row to 33 elements per row. Its layout in UB is shown below. Since each row has 33 elements stored across 5 banks, elements in the same column are staggered and distributed in different banks. In Case 3, when accessing UB, accesses from 32 threads will be distributed across 8 banks, meaning one bank will have 4 threads accessing simultaneously, greatly reducing the scale of conflicts.

<img src="./figure/case3bank.png">

**Key Code**:

```cpp
__ubuf__ float tile[TILE_DIM][TILE_DIM + 1];

tile[threadIdx.y][threadIdx.x] = input[index_in];
asc_syncthreads();

x_index = blockIdx.y * TILE_DIM + threadIdx.x;
y_index = blockIdx.x * TILE_DIM + threadIdx.y;
int index_out = x_index + y_index * height;

output[index_out] = tile[threadIdx.x][threadIdx.y];
```

**Optimization Methods**:

- Add +1 padding to UB tile, making each row stride from 32 floats to 33 floats
- Reduce the probability of concentrated access to a few banks by the same Warp in the transpose read phase by changing the bank distribution of elements in the same column in UB

**Performance Data**:

| Task Duration(us) | aiv_time(us) | aiv_total_cycles | aiv_vec_time(us) | aiv_vec_ratio | aiv_scalar_time(us) | aiv_scalar_ratio |
| :---------------: | :----------: | :--------------: | :--------------: | :-----------: | :-----------------: | :--------------: |
|      26.725      |    1.224    |     2018.943     |      1.059      |     0.869     |        0.152        |      0.121      |

**Analysis**:

- Compared with transpose_coalesced_kernel of Case 2, Task Duration decreased from 35.945us to 26.725us, a time reduction of about 25.7%, overall performance improvement of about 1.35 times
- Case 3 reduces bank conflicts in the UB transpose read phase through padding on the basis of Case 2, so end-to-end Task Duration continues to decrease
- Compared with naive transpose of Case 1, Case 3's Task Duration decreased by about 55.8%, overall performance improvement of about 2.26 times, indicating that the two optimization steps of "GM memory access coalescing + UB bank conflict reduction" have cumulative effect on end-to-end time
- Compared with the Case 0 copy baseline, Case 3's Task Duration is only about 7.9% higher, already approaching the sequential GM read/write baseline level. The remaining gap mainly comes from UB staging, synchronization, and small amount of bank conflicts that still exist during UB read/write

---

## Performance Comparison Summary

### Ascend 950PR Performance Data

**Overall Optimization Effect**:

- From Case 1 direct index transpose to Case 3 fully optimized version, Task Duration decreased from 60.477us to 26.725us, a time reduction of about 55.8%, overall performance improvement of about 2.26 times
- Case 3 is only about 7.9% higher than the Case 0 copy baseline, indicating that through GM memory access coalescing and UB Bank conflict optimization, matrix transpose has approached the sequential GM read/write baseline

| Case version | Task Duration(us) | Task Duration relative to Case 0 | Optimization Points                                |
| ------------ | ----------------- | ----------------------- | ------------------------------------- |
| Case 0       | 24.777            | **1x**            | Matrix copy baseline, GM sequential read, sequential write        |
| Case 1       | 60.477            | **2.44x time**     | Direct index transpose, GM sequential read, non-sequential write      |
| Case 2       | 35.945            | **1.45x time**     | UB staging, global memory access coalescing                  |
| Case 3       | 26.725            | **1.08x time**     | UB staging, global memory access coalescing, avoid UB Bank conflicts |

## Tuning Recommendations

1. **First establish a copy baseline**: When analyzing memory access type operator performance, it is recommended to first measure the copy scenario time, then compare actual operator performance with it.
2. **Prioritize GM memory access sequentiality**: Matrix transpose has very small computation, and end-to-end time is mainly affected by read/write memory access patterns.
3. **Use UB staging to improve write-back pattern**: When direct transpose causes GM non-sequential write, you can transfer non-sequential access to the UB side in exchange for sequential read/write on the GM side.
4. **Continue to analyze UB Bank conflicts**: After GM memory access coalescing, bank conflicts in the UB transpose read phase may become the next layer bottleneck, and you can adjust UB physical layout through padding and other methods.

## Build and Run

Execute the following steps in the root directory of this example to build and run the example.

- Configure environment variables
  Select the corresponding command to configure environment variables based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on the current environment.

  - Default path, CANN software package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installation

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example execution

  ```bash
  SCENARIO_NUM=3                       # Select execution scenario, options 0-3
  mkdir -p build && cd build;          # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO_NUM ..;make -j;  # Build the project
  ./matrix_transpose                   # Run the example
  ```

- Build option description

  | Option             | Available Values      | Description              |
  | ---------------- | ----------- | ----------------- |
  | `SCENARIO_NUM` | `0`-`3` | Example type, default is 3 |

  The following output indicates successful accuracy verification.


  ```text
  [Success] Case accuracy is verification passed.
  ```

## Performance Analysis

Use the `msprof` tool to obtain detailed performance data:

```bash
msprof op ./matrix_transpose   # Analyze case performance
```

After the command completes, a folder named "OPPROF_{timestamp}_XXX" will be generated in the default directory. The performance data folder structure is shown below:

```text
├──dump                       # Raw performance data, users do not need to focus on
├──ArithmeticUtilization.csv  # cube/vector instruction cycle ratio
├──L2Cache.csv                # L2 Cache hit rate
├──Memory.csv                 # UB, L1 and main memory read/write bandwidth rate
├──MemoryL0.csv               # L0A, L0B, and L0C read/write bandwidth rate
├──MemoryUB.csv               # Vector and Scalar to UB read/write bandwidth rate
├──OpBasicInfo.csv            # Operator basic information
├──PipeUtilization.csv        # Compute unit and transfer unit duration and ratio
├──ResourceConflictRatio.csv  # Bank group, bank conflict and resource conflict ratio in UB among all instructions
└──visualize_data.bin         # MindStudio Insight presentation file
```