# HelloWorld Operator Direct Call Sample

## Summary

This sample demonstrates the basic flow of verifying operator kernel function execution on the NPU side using the <<<>>> kernel call operator, with printf output inside the kernel function.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── hello_world_npu
│   ├── CMakeLists.txt      // Build project file
│   └── hello_world.asc     // Ascend C operator implementation & call sample
```

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
  ./demo                        # Execute sample
  ```
  The following output indicates successful execution.
  ```bash
  [Block (0/8)]: Hello World!!!
  [Block (1/8)]: Hello World!!!
  [Block (2/8)]: Hello World!!!
  [Block (3/8)]: Hello World!!!
  [Block (4/8)]: Hello World!!!
  [Block (5/8)]: Hello World!!!
  [Block (6/8)]: Hello World!!!
  [Block (7/8)]: Hello World!!!
  ```