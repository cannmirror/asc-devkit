# cast_tensor_api Sample

## Overview

This sample uses the **Tensor API** `Transform<Inst::Cast, Trait>` interface to implement data type conversion.

**Main changes:**

| Item | Tensor API |
|---|---|
| Data shape | 16x16 layout matrix |
| Copy in | `MakeCopy(CopyGM2UB{})` -> `Copy(atom, xUb, xGm)` |
| Cast compute | `Transform<Inst::Cast, Trait>(yUbTensor, xUbTensor)` |
| Copy out | `MakeCopy(CopyUB2GM{})` -> `Copy(atom, yGm, yUb)` |
| Memory allocation | Calculate offsets and use `MakeMemPtr<Location::UB, T>(offset)` |

**Supported scenarios:**

| SCENARIO_NUM | Conversion | RoundMode | SatMode | IndexPos |
|---|---|---|---|---|
| 1 | `half` -> `int32_t` | RD (floor) | NoSat | - |
| 2 | `float` -> `int16_t` | RN (round) | Sat | - |
| 3 | `int8_t` -> `int32_t` | RD (floor) | NoSat | PartP0 |
| 4 | `int32_t` -> `uint8_t` | RD (floor) | Sat | PartP0 |
| 5 | `bfloat16_t` -> `float` | - | - | Even |
| 6 | `float` -> `bfloat16_t` | RN (round) | NoSat | Even |

## Build and Run

- Run the sample

  Run the following commands in this sample directory.

  ```bash
  SCENARIO_NUM=1                                                                 # Run scenario 1.
  mkdir -p build && cd build                                                     # Create and enter the build directory.
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j # Build the project. NPU mode is used by default.
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM                      # Generate test input data.
  ./demo                                                                         # Run the generated executable.
  ```

  To use CPU debug mode or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim`.

  Examples:

  ```bash
  cmake -DSCENARIO_NUM=1 -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j # CPU debug mode.
  cmake -DSCENARIO_NUM=4 -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j # NPU simulation mode.
  ```

  > **Note:** Before switching the build mode, clear the CMake cache. You can run `rm CMakeCache.txt` in the `build` directory and then run `cmake` again.

Expected output:

```text
test pass!
```
