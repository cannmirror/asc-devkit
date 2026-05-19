# Matmul and LeakyRelu Fusion Computation Sample

## Overview

This sample implements Matmul and LeakyRelu fusion computation based on static Tensor programming mode, demonstrating the programming pattern of collaborative computation between Cube unit and Vector unit. The hardware Cube:Vector core ratio is 1:2. The Cube core completes matrix multiplication computation, and the Vector core completes LeakyRelu activation computation.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── matmul_leakyrelu_basic_api
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script file
│   │   └── verify_result.py    // Golden data comparison file
│   ├── CMakeLists.txt          // Compilation project file
│   ├── data_utils.h            // Data read/write functions
│   └── matmul_leakyrelu_basic_api.asc  // Ascend C sample implementation & invocation sample
```

## Sample Description

- Sample Function:
  Implements Matmul and LeakyRelu fusion computation. The computation formula is as follows:

  Matmul computation:
  $$
  C = A \times B
  $$

  LeakyRelu computation:
  $$
  C = \begin{cases}
  C & \text{if } C \geq 0 \\
  C \times 0.001 & \text{if } C < 0
  \end{cases}
  $$

  Where A is the left matrix with shape [M, K]; B is the right matrix with shape [K, N]; C is the output matrix with shape [M, N].

- Sample Specifications:
  This sample has parameters M = 512, K = 512, N = 1024, using 4 Cube cores and 8 Vector cores for computation. The input specifications are shown in the table below:

  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center">Matmul+LeakyRelu Fusion</td></tr>
  <tr><td rowspan="3" align="center">Sample Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">A (left matrix)</td><td align="center">[512, 512]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">B (right matrix)</td><td align="center">[512, 1024]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Sample Output</td><td align="center">C</td><td align="center">[512, 1024]</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">mmad_vec_custom</td></tr>
  </table>

  **Core Partitioning Logic**

  The output matrix C is partitioned along M and N directions, divided into 4 blocks in total. Each block is completed by 1 Cube core for Matmul computation, and each Cube core's computation result is completed by 2 Vector cores working together for LeakyRelu computation.

  Core partitioning parameters description:
  - Number of partitions in M direction: M / singleCoreM = 512 / 256 = 2
  - Number of partitions in N direction: N / singleCoreN = 1024 / 512 = 2
  - Total Cube cores: 2 × 2 = 4
  - Total Vector cores: 4 × 2 = 8 (Cube:Vector = 1:2)
  - Each Cube core is responsible for singleCoreM*singleCoreN Matmul computation, producing baseM*baseN results each time
  - Each Vector core is responsible for singleCoreM/2*singleCoreN computation, producing baseM/2*baseN results each time

- Sample Implementation:
  - Overall Computation Flow:

    The overall sample flow is as follows (showing Cube core and Vector core collaborative computation):

**Cube Core: Matmul Matrix Multiplication**
```
GM(A:ND,half) -> L1(A:Nz,half) -> L0A(A:Nz,half) -
             │                  │                 │
          DataCopy            LoadData            │
          ND->Nz              Nz->Nz              │
                                                  │--->L0C(Nz,float) -> GM(C:ND,half)
                                                  │  │                  │
                                                  │ Mmad             Fixpipe
                                                  │ C=A×B       Nz->ND, float32->half
                                                  │
GM(B:ND,half) -> L1(B:Nz,half) -> L0B(B:Zn,half) -
             │                  │
          DataCopy            LoadData
          ND->Nz              Nz->Zn(transpose)
```

  > **Note**: L0A fractal format differs across products:
  > - Ascend 950PR/Ascend 950DT products: L0A fractal format is Nz
  > - Atlas A2/A3 series products: L0A fractal format is Zz

**Inter-core Synchronization**
```
Cube core produces computation result -> CrossCoreSetFlag -> Vector core waits (CrossCoreWaitFlag)
```

**Vector Core (1 cube core's result is passed to 2 vector cores for computation): LeakyRelu Activation**
```
GM(C:ND,half) -> UB(VECCALC,half) -> UB(VECCALC,half) -> GM(C:ND,half)
                │                   │                   │
            DataCopyPad          LeakyRelu          DataCopyPad
      MTE2 transfer (baseM/2×baseN)    VEC computation    MTE3 writeout

```

**Flow Details**:

1. **Cube Core Computation Phase**:
    - **GM → L1**: Use `DataCopy` to move matrices A and B from GM to L1, completing ND to Nz format conversion
    - **L1 → L0A/L0B**: Use `LoadData` to move data to L0A and L0B, matrix B needs transpose (Nz→Zn)
    - **L0A/L0B → L0C**: Use `Mmad` to execute matrix multiplication and accumulation, accumulating all data blocks along the K-axis direction
    - **L0C → GM**: Use `Fixpipe` to move results out to GM, completing Nz to ND format conversion and float32 to half type conversion

2. **Inter-core Synchronization**:
    - After a Cube core completes a baseM×baseN Matmul result, it sets a flag via `CrossCoreSetFlag`
    - Vector cores wait for the flag via `CrossCoreWaitFlag` to ensure LeakyRelu starts only after Matmul computation completes

3. **Vector Core Computation Phase**:
    - **GM → UB**: Use `DataCopyPad` to move Matmul results to UB, each Vector core processes baseM/2×baseN data
    - **UB Computation**: Use `LeakyRelu` to execute activation computation, negative values are multiplied by 0.001
    - **UB → GM**: Use `DataCopyPad` to write results back to GM, completing fusion computation

4. **Core Ratio**:
    - Cube:Vector core ratio is 1:2, each Cube core produces baseM×baseN results
    - 2 Vector cores each process baseM/2×baseN data, jointly completing activation computation for one Matmul block

- Cube and Vector Collaboration Mechanism:
  - **Core Ratio**: Cube:Vector core ratio is 1:2, each Cube core's computation result is completed by 2 Vector cores working together for LeakyRelu computation
  - **Inter-core Synchronization**: After a Cube core completes Matmul computation, it notifies Vector cores to start computation via CrossCoreSetFlag; Vector cores wait for Cube core completion via CrossCoreWaitFlag
  - **Data Partitioning**: Each Vector core processes baseM/2 × baseN sized data, two Vector cores jointly process one baseM × baseN Matmul result block

  - Constraints:
    1. baseM/baseK/baseN satisfy 16-byte alignment
    2. baseM/baseK/baseN are divisible by singleCoreM/singleCoreK/singleCoreN
    3. singleCoreM/singleCoreK/singleCoreN are divisible by M/K/N, non-integer partitioning scenarios are not supported
    4. Vector core count is 2 times the Cube core count

  - Invocation Implementation
    The kernel call operator `__mix__(1, 2)` implements collaborative invocation of Cube and Vector cores, where parameter (1, 2) indicates a Cube:Vector core ratio of 1:2.

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
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;                      # Compile project (default npu mode)
  python3 ../scripts/gen_data.py                                            # Generate test input data
  ./demo                                                                    # Execute compiled program, run sample
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # Verify output results are correct, confirm algorithm logic is correct
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  Example:
  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note**: Before switching compilation modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then run cmake again.

- Compilation Options Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating that the accuracy comparison passed.
  ```bash
  test pass!
  ```