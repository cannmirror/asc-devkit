# Add Operator Sample Using C_API (RegBase Scenario)

## Summary

This sample uses C_API interfaces to implement the Add operator sample, based on synchronous data movement and computation interfaces.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Directory Structure

```
├── c_api_simd_add
│   ├── CMakeLists.txt          // Build project file
│   ├── c_api_add.asc          // Ascend C operator implementation & call sample
│   └── README.md
```

## Operator Description

- Operator Function:  
  The Add operator implements the function of adding two values and returning the result. The corresponding mathematical expression is:  

  ```
  z = x + y
  ```

- Operator Specification:
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">2048*8</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">2048*8</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">z</td><td align="center">2048*8</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- Operator Implementation:

  - Kernel Implementation  

    C_API input data needs to be moved to on-chip storage first, then loaded into Reg vector computation registers, then computation interfaces are used to complete the addition of two input parameters to obtain the computation result, which is moved to Local Memory, then moved to external storage.

    The Add operator implementation flow consists of 3 steps:

    First step: Move input x and y from Global Memory to Local Memory, stored in xLocal and yLocal respectively.

    Second step: Load data from Unified Buffer to registers reg_src0 and reg_src1, use `asc_add` to execute addition operation on register data, store the result in register reg_dst, and move the result back to zLocal after computation.

    Third step: Move output data from zLocal to output z on Global Memory.

  - Mask Control in Vector Computation

    In vector computation instructions, mask controls which channels of the vector register participate in computation. This sample uses the asc_update_mask_b32 function to set a 32-bit vector mask, controlling the validity of each channel in the 256-bit vector register.

  - Call Implementation  
    Use the kernel call operator <<<>>> to invoke the kernel function.

## Build and Run

Execute the following steps in the sample root directory to build and run the operator.

- Configure Environment Variables
  Select the appropriate command to configure environment variables based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on your current environment.
  - Default path, root user installed CANN package

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - Default path, non-root user installed CANN package

    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - Specified path install_path, installed CANN package

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- Sample Execution

  ```bash
  mkdir -p build && cd build;   # Create and enter build directory
  cmake ..;make -j;             # Build project
  # Execute the following in the build directory
  ./c_api_add_example           # Execute sample
  ```

  The following output indicates successful accuracy verification.

  ```bash
  [Success] Case accuracy is verification passed.
  ```