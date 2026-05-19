# IsFinite Example

## Overview

This example implements element-wise checking of whether input floating-point numbers are neither NAN nor INF using the IsFinite high-level API. The output result is a floating-point number or boolean value.

> **API Note:** In addition to the `IsFinite` interface used in this example, Ascend C provides the following condition-checking related high-level API interfaces. You can replace the interface name as needed:
>
> - **IsInf**: Checks whether the input is infinity.
> - **IsNaN**: Checks whether the input is Not a Number.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```plain
├── isfinite
│   ├── scripts
│   │   └── gen_data.py         // Input data and ground truth data generation script
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read and write functions
│   └── isfinite.asc            // Ascend C operator implementation & invocation example
```

## Example Description

- Example Function:
  Performs element-wise checking of whether input floating-point numbers are neither NAN nor INF, with output as floating-point number or boolean value. For input data that is neither NAN nor INF, when the output is floating-point type, the result at the corresponding position is 1 of that floating-point type, otherwise 0; when the output is bool type, the result at the corresponding position is true, otherwise false.
  The calculation formula is as follows:
  $$dst_i = IsFinite(src_i)$$

  When the input is floating-point type:
  $$
  IsFinite(x) = 
  \begin{cases}
  0.0, & x = \pm\inf \text{ or } x = \text{nan} \\
  1.0, & x \ne \pm\inf \text{ and } x \ne \text{nan}
  \end{cases}
  $$

  When the output is bool type:
  $$
  IsFinite(x) =
  \begin{cases}
  false, & x = \pm\inf \text{ or } x = \text{nan} \\
  true, & x \ne \pm\inf \text{ and } x \ne \text{nan}
  \end{cases}
  $$
- Example Specification:
  <table>
  <tr><td rowspan="1" align="center">Example Type(OpType)</td><td colspan="4" align="center"> isfinite </td></tr>

  <tr><td rowspan="3" align="center">Example Input</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">[1, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">Example Output</td></tr>
  <tr><td align="center">y</td><td align="center">[1, 1024]</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">isfinite_custom</td></tr>
  </table>

- Example Implementation:
  This example implements the isfinite_custom example with a fixed shape of input x[1, 1024] and output y[1, 1024].

  - Kernel Implementation
    The calculation logic is: The vector calculation interface provided by Ascend C operates on elements that are all LocalTensor. Input data needs to be moved to on-chip storage first, then the IsFinite high-level API interface is used to complete the isfinite calculation to get the final result, which is then moved to external storage.

  - Invocation Implementation
    Uses the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the root directory of this example to build and run the operator.

- Configure Environment Variables
  Select the corresponding command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit on your current environment.
  - Default path, CANN package installed by root user

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, CANN package installed by non-root user

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, CANN package installed

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Example Execution

  ```bash
  mkdir -p build && cd build;      # Create and enter the build directory
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;    # Build the project, default npu mode
  python3 ../scripts/gen_data.py   # Generate test input data
  ./demo                           # Execute the compiled executable to run the example
  ```

  When using CPU debug or NPU simulation mode, add the `-DCMAKE_ASC_RUN_MODE=cpu` or `-DCMAKE_ASC_RUN_MODE=sim` parameter.

  For example:

  ```bash
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # CPU debug mode
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, you need to clear the cmake cache. You can execute `rm CMakeCache.txt` in the build directory and then run cmake again.

- Build Options Description

  | Option | Available Values | Description |
  |------|--------|------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `cpu`, `sim` | Run mode: NPU run, CPU debug, NPU simulation |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-3510` (default) | NPU architecture: dav-3510 corresponds to Ascend 950PR/Ascend 950DT |

- Execution Result

  The execution result is as follows, indicating successful precision comparison.

  ```bash
  test pass!
  ```