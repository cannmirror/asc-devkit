# Acosh Sample

## Overview

This sample calculates the inverse hyperbolic cosine function using the Acosh high-level API.

> **Related Samples:** In addition to the `Acosh` interface used in this sample, Ascend C also provides the following trigonometric function-related high-level API interfaces. Except for sincos, the implementation is essentially the same. You can simply replace the interface name to call them:
>
> - **acos**: Inverse cosine function.
> - **asin**: Inverse sine function.
> - **asinh**: Inverse hyperbolic sine function.
> - **atanh**: Inverse hyperbolic tangent function.
> - **cos**: Cosine function.
> - **cosh**: Hyperbolic cosine function.
> - **sinh**: Hyperbolic sine function.
> - **tan**: Tangent function.
> - **sincos**: Sine and cosine function, calculates sine and cosine separately. Two output tensors are required when calling.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```plain
├── acosh
│   ├── CMakeLists.txt     // Build project file
│   └── acosh.asc          // Ascend C sample implementation & call sample
```

## Sample Description

- Sample Function:
  Computes the inverse hyperbolic cosine function element-wise. The calculation formula is as follows:
  $$dstTensor_i = Acosh(srcTensor_i)$$
  $$Acosh(x)=\begin{cases}Nan, & x < 1 \\ \ln(x+\sqrt{x^{2}-1}), & x > 1\end{cases}$$
- Sample Specifications:
  <table>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center"> acosh </td></tr>

  <tr><td rowspan="3" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[1, 16]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Sample Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 16]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">acosh_custom</td></tr>
  </table>

- Sample Implementation:
  This sample implements acosh_custom with a fixed shape of input src[1, 16] and output dst[1, 16].

  - Kernel Implementation

    Uses the Acosh high-level API interface to complete the inverse hyperbolic cosine calculation.

  - Tiling Implementation

    The host side uses GetAcoshMaxMinTmpSize to obtain the maximum and minimum temporary space required for the Acosh interface calculation.

  - Call Implementation
    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the sample.

- Configure Environment Variables
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package on your current environment.
  - Default path, CANN software package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN software package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN software package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  ./demo                           # Execute the generated executable program to run the sample
  ```

  For CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Examples:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2/A3 series, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating that the accuracy comparison passed.

  ```bash
  test pass!
  ```