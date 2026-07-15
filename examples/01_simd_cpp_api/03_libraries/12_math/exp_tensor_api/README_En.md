# Exp Tensor API Sample

## Overview

This sample uses the Tensor API `Transform<Inst::Exp>` interface to compute the natural exponential function ($e^x$) element by element.

The sample uses the Tensor API `Transform` template interface for vector computation, `MakeCopy` / `Copy` for data movement, and `MakeTensor` / `MakeMemPtr` / `MakeFrameLayout` to construct Tensor objects.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```plain
├── exp_tensor_api
│   ├── scripts
│   │   └── gen_data.py         // Script for generating input data and golden data
│   ├── CMakeLists.txt          // Build project file
│   ├── data_utils.h            // Data input and output helper functions
│   └── exp_tensor_api.asc      // Ascend C operator implementation and invocation sample
```

## Sample Description

- Sample function:

  Computes the natural exponential function element by element. The formula is:

  $$
  dstLocal_i = Exp(srcLocal_i) = e^{srcLocal_i}
  $$

- Sample specification:

  <table>
  <caption>Table 1: Sample Input and Output Specification</caption>
  <tr><td rowspan="1" align="center">OpType</td><td colspan="4" align="center"> exp_tensor_api </td></tr>

  <tr><td rowspan="3" align="center">Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[64, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[64, 128]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel name</td><td colspan="4" align="center">exp_tensor_api_custom</td></tr>
  </table>

- Sample implementation:

  This sample implements `exp_tensor_api` with a fixed shape: input `src[1, 8192]` and output `dst[1, 8192]`.

  - Kernel implementation

    The sample uses the Tensor API `Transform<Inst::Exp>(dstUb, srcUb)` interface to compute the natural exponential function.

    Main process:

    1. Use `MakeTensor` + `MakeMemPtr` + `MakeFrameLayout` to create Tensor objects in GM and UB.
    2. Use `MakeCopy(CopyGM2UB{})` to construct a CopyAtom, and use `Copy` to move data from GM to UB.
    3. Use `SetFlag` / `WaitFlag` for MTE2-to-V pipeline synchronization.
    4. Use `Transform<Inst::Exp>` to execute the natural exponential computation.
    5. Use `SetFlag` / `WaitFlag` for V-to-MTE3 pipeline synchronization.
    6. Use `MakeCopy(CopyUB2GM{})` to construct a CopyAtom, and use `Copy` to move the result from UB back to GM.

  - Invocation implementation

    The kernel function is launched with the kernel invocation syntax `<<<>>>`.

## Core Interfaces

### Transform

```cpp
template<typename CalcFunc, typename TraitType = Std::ignore_t, typename... Args>
__aicore__ inline void Transform(const Args&... args)
```

For Exp unary computation, call it as follows:

```cpp
Transform<Inst::Exp>(dstTensor, srcTensor)
```

### MakeCopy / Copy

```cpp
// Construct CopyAtom.
auto copyInAtom = MakeCopy(CopyGM2UB{});
auto copyOutAtom = MakeCopy(CopyUB2GM{});

// Execute data movement.
Copy(copyInAtom, dstTensor, srcTensor);
```

### MakeTensor

```cpp
auto tensor = MakeTensor(
    MakeMemPtr<Location::GM>(ptr),
    MakeFrameLayout<NDLayoutPtn>(_1{}, AscendC::Std::Int<8192>{}));
```

## Build and Run

Run the following steps in the sample root directory to build and execute the operator.

- Configure environment variables.

  Configure environment variables according to the CANN development package installation method in [Quick Start](../../../../../docs/quick_start.md#prepare&install).

  ```bash
  source ${install_path}/cann/set_env.sh
  ```

- Run the sample.

  Run the following commands in this sample directory.

  ```bash
  mkdir -p build && cd build;      # Create and enter the build directory.
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build the project. The default mode is npu.
  python3 ../scripts/gen_data.py   # Generate test input data.
  ./demo                           # Run the generated executable.
  ```

  To use CPU debug mode or NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim`.

  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching the build mode, clear the CMake cache. You can run `rm CMakeCache.txt` in the `build` directory and then run `cmake` again.

- Build options

  | Option | Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, or NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU architecture. `dav-3510` corresponds to Ascend 950PR/Ascend 950DT, Atlas A3 training series, and Atlas A3 inference series products. |

- Execution result

  The following output indicates that the precision comparison succeeds.

  ```bash
  test pass!
  ```
