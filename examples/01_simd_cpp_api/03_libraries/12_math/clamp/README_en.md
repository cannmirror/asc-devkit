# Clamp Sample

## Overview

This sample implements the functionality of truncating values in the input (except nan values) to the interval [min, max] using the Clamp high-level API.
When min is greater than max, all values except nan are replaced with max. Both min and max can be scalars or tensors.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```plain
├── clamp
│   ├── scripts
│   │   └── gen_data.py   // Input data and golden data generation script
│   ├── CMakeLists.txt    // Build project file
│   ├── data_utils.h      // Data read and write functions
│   └── clamp.asc         // Ascend C sample implementation & call sample
```

## Sample Description

- Sample Function:
  Replaces values in the input that are greater than max and not NaN with max, values less than min and not NaN with min, and keeps values less than or equal to max and greater than or equal to min unchanged as output. When min is greater than max, all non-NaN values are replaced with max. min and max can be scalars or tensors.

  The calculation formula is as follows:

  $$
  dst_i = Clamp(src_i, min_i, max_i)
  $$

  $$
  Clamp(src_i, min_i, max_i) =
  \begin{cases}
  min_i, & src_i < min_i \\
  src_i, & min_i \le src_i \le max_i \\
  max_i, & src_i > max_i \\
  \end{cases}
  $$

- Sample Specifications:
  <table>
  <caption>Table 1: Sample Specifications</caption>
  <tr><td rowspan="1" align="center">Sample Type (OpType)</td><td colspan="4" align="center"> clamp </td></tr>

  <tr><td rowspan="5" align="center">Sample Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">src</td><td align="center">[1, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">src_min</td><td align="center">[1, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">src_max</td><td align="center">[1, 128]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Sample Output</td></tr>
  <tr><td align="center">dst</td><td align="center">[1, 128]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">clamp_custom</td></tr>
  </table>

- Scenario Description:
  <table>
  <caption>Table 2: scalarType Parameter Description</caption>
  <tr><td align="center">scalarType</td><td align="center">min Type</td><td align="center">max Type</td><td align="center">Description</td></tr>
  <tr><td align="center">1</td><td align="center">Tensor</td><td align="center">Tensor</td><td align="center">Both min and max are tensors</td></tr>
  <tr><td align="center">2</td><td align="center">Tensor</td><td align="center">Scalar</td><td align="center">min is a tensor, max is a scalar</td></tr>
  <tr><td align="center">3</td><td align="center">Scalar</td><td align="center">Tensor</td><td align="center">min is a scalar, max is a tensor</td></tr>
  <tr><td align="center">4</td><td align="center">Scalar</td><td align="center">Scalar</td><td align="center">Both min and max are scalars</td></tr>
  </table>

- Sample Implementation:

  This sample implements clamp_custom with a shape of input src[128], src_min[128], src_max[128] and output dst[128], supporting 4 scenario combinations where min and max are tensors or scalars.

  - Kernel Implementation

    Uses the Clamp high-level API interface to complete the Clamp calculation and obtain the final result, which is then moved to external storage.

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
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the generated executable program to run the sample
  ```

  For CPU debugging or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debugging mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Option Description

  | Option | Available Values | Description |
  |--------|------------------|-------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU execution, CPU debugging, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating that the accuracy comparison passed.

  ```bash
  test pass!
  ```