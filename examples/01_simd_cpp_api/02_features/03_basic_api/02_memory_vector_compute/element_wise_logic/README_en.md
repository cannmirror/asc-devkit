# Element-wise Logic Operations Example

## Overview

This sample demonstrates bitwise logic operations using And, Ors, ShiftLeft, and ShiftRight interfaces. The And interface performs bitwise AND operation on two source operands. The Ors interface performs OR operation between each element in a vector and a scalar. The ShiftLeft interface performs left shift operation on source operand (tensor form). The ShiftRight interface performs right shift operation on source operand (scalar form). The sample supports switching between different scenarios through compile parameters, helping developers understand the usage and implementation differences of these interfaces.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── element_wise_logic
│   ├── scripts
│   │   ├── gen_data.py         // Script to generate input data and golden data
│   │   └── verify_result.py    // Script to verify output data against golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── element_wise_logic.asc  // Ascend C sample implementation & invocation example
```

## Scenario Description

This sample switches between different scenarios through the compile parameter `SCENARIO_NUM`:

<table border="2">
<caption>Table 1: Scenario Configuration Reference</caption>
<tr><th>scenarioNum</th><th>Interface</th><th>Input Shape</th><th>Output Shape</th><th>Data Type</th><th>Description</th></tr>
<tr><td>1</td><td>And</td><td>[1, 512], [1, 512]</td><td>[1, 512]</td><td>uint16</td><td>Bitwise AND operation on two source operands</td></tr>
<tr><td>2</td><td>Ors</td><td>[1, 512], [1, 512]</td><td>[1, 512]</td><td>uint16</td><td>Scalar before, src0[0] as scalar with src1 vector for OR operation (dav-3510 only)</td></tr>
<tr><td>3</td><td>ShiftLeft</td><td>[1, 512], [1, 512]</td><td>[1, 512]</td><td>uint16</td><td>Left shift operation, shift amount specified by tensor (dav-3510 only)</td></tr>
<tr><td>4</td><td>ShiftRight</td><td>[1, 512]</td><td>[1, 512]</td><td>uint16</td><td>Right shift operation, shift amount specified by constant SHIFT_BITS=2</td></tr>
</table>

**Scenario 1: And Bitwise AND Operation**
- Input shape: src0=[1, 512], src1=[1, 512]
- Output shape: dst=[1, 512]
- Data type: uint16
- Parameter: count=512
- Implementation:

    ```cpp
    AscendC::And(dstLocal, src0Local, src1Local, COUNT);
    ```

- Description: Performs bitwise AND operation on each element in src0 and src1, result stored in dst
- Example:
  - Input src0: [1 2 3 ... 512]
  - Input src1: [512 511 510 ... 1]
  - Output dst: [1 0 3 ... 0]

**Scenario 2: Ors Vector with Scalar OR Operation (Scalar Before)** ---- Only supported on Ascend 950PR/Ascend 950DT**
- Input shape: src0=[1, 512] (src0Local[0] taken as scalar), src1=[1, 512] (vector)
- Output shape: dst=[1, 512]
- Data type: uint16
- Parameter: count=512
- Implementation:

    ```cpp
    static constexpr AscendC::BinaryConfig config = { 0 };
    AscendC::Ors<AscendC::BinaryDefaultType, true, config>(dstLocal, src0Local[0], src1Local, COUNT);
    ```

- Description: Scalar before, src0Local[0] as scalar (left operand), src1Local as vector (right operand), performs bitwise OR operation on each element in src1 with src0Local[0], result stored in dst
- Example:
  - Input src0[0]: 1
  - Input src1: [1 2 3 ... 512]
  - Output dst: [1 3 3 5 5 .. 513]

**Scenario 3: ShiftLeft Left Shift Operation (Tensor Form)** ---- Only supported on Ascend 950PR/Ascend 950DT**
- Input shape: src0=[1, 512] (data to shift, uint16), src1=[1, 512] (left shift amount, int16)
- Output shape: dst=[1, 512]
- Data type: uint16
- Parameter: count=512
- Implementation:

    ```cpp
    AscendC::ShiftLeft(dstLocal, src0Local, src1Local, COUNT);
    ```

- Description: Left shifts each element in src0 by the corresponding amount in src1, result stored in dst. src1 contains shift amounts, negative values not supported
- Example:
  - Input src0: [1 2 3 ... 512]
  - Input src1: [2 2 2 ... 2] (shift amounts)
  - Output dst: [4 8 12 ... 2048]

**Scenario 4: ShiftRight Right Shift Operation (Scalar Form)**
- Input shape: src0=[1, 512]
- Output shape: dst=[1, 512]
- Data type: uint16
- Parameters: count=512, SHIFT_BITS=2
- Implementation:

    ```cpp
    AscendC::ShiftRight(dstLocal, src0Local, SHIFT_BITS, COUNT);
    ```

- Description: Right shifts each element in src0 by SHIFT_BITS bits, result stored in dst. Unsigned data types perform logical right shift, signed data types perform arithmetic right shift
- Example:
  - Input src0: [1 2 3 4 5 ... 512]
  - Output dst: [0 0 0 1 1 1 1 2 2 2 ... 128]

## Build and Run

Execute the following steps in the sample root directory to build and run the sample.

- Configure Environment Variables  
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, root user installed CANN package

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN package

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Custom path install_path, installed CANN package

    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- Sample Execution

  ```bash
  SCENARIO_NUM=1  # Set scenario number
  mkdir -p build && cd build;      # Create and enter build directory
  cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # Build project
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # Generate test input data
  ./demo                           # Execute the compiled executable to run the sample
  python3 ../scripts/verify_result.py ./output/output.bin ./output/golden.bin  # Verify output results
  ```

  For CPU debug or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake .. -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # CPU debug mode
  cmake .. -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. Execute `rm CMakeCache.txt` in the build directory and run cmake again.

- Build Options

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 for Atlas A2/A3 series, dav-3510 for Ascend 950PR/Ascend 950DT. Note: Scenarios 2 and 3 only support dav-3510, will auto-switch during build |
  | `SCENARIO_NUM` | `1` (default), `2`, `3`, `4` | Scenario number: 1 (And bitwise AND), 2 (Ors vector scalar OR), 3 (ShiftLeft tensor form), 4 (ShiftRight scalar form) |

- Execution Result

  The following output indicates successful accuracy comparison.

  ```bash
  test pass!
  ```