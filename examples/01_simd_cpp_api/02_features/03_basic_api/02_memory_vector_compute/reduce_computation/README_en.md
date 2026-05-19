# ReduceComputation Example

## Overview

This example implements reduction computation using ReduceMax/ReduceMin/ReduceSum interfaces, supporting the following 6 scenarios:

    <table>
  	 	 	<tr>
  	 	 		<td>scenarioNum</td>
  	 	 		<td>Scenario</td>
          <td>Description</td>
  	 	 	</tr>
  	 	 	<tr>
  	 	 		<td>1</td>
  	 	 		<td>ReduceMax first n elements computation</td>
          <td>Find the maximum value and its corresponding index from the first n elements of the input tensor</td>
  	 	 	</tr>
  	 	 	<tr>
  	 	 		<td>2</td>
  	 	 		<td>ReduceMax high-dimensional tiling computation</td>
          <td>Find the maximum value and its corresponding index from all input elements, using mask to control elements participating in computation within each iteration</td>
  	 	 	</tr>
        <tr>
  	 	 		<td>3</td>
  	 	 		<td>ReduceMin first n elements computation</td>
          <td>Find the minimum value and its corresponding index from the first n elements of the input tensor</td>
  	 	 	</tr>
        <tr>
  	 	 		<td>4</td>
  	 	 		<td>ReduceMin high-dimensional tiling computation</td>
          <td>Find the minimum value and its corresponding index from all input elements, using mask to control elements participating in computation within each iteration</td>
  	 	 	</tr>
        <tr>
  	 	 		<td>5</td>
  	 	 		<td>ReduceSum first n elements computation</td>
          <td>Sum the first n elements of the input tensor</td>
  	 	 	</tr>
        <tr>
  	 	 		<td>6</td>
  	 	 		<td>ReduceSum high-dimensional tiling computation</td>
          <td>Sum all input elements, using mask to control elements participating in computation within each iteration</td>
  	 	 	</tr>
  	 	 </table>

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── reduce_computation
│   ├── scripts
│   │   ├── gen_data.py         // Input data and golden data generation script
│   │   └── verify_result.py    // Verification script for comparing output data with golden data
│   ├── CMakeLists.txt          // Build configuration file
│   ├── data_utils.h            // Data read/write functions
│   └── reduce_computation.asc  // Ascend C implementation & invocation example
```

## Example Description

- Example functionality:  
  This example implements reduction computation using ReduceMax/ReduceMin/ReduceSum interfaces, including first n elements computation interface and tensor high-dimensional tiling computation interface. For interface documentation, refer to [ReduceMax](../../../../../../docs/api/context/ReduceMax.md)/[ReduceMin](../../../../../../docs/api/context/ReduceMin.md)/[ReduceSum](../../../../../../docs/api/context/ReduceSum.md).

- Example specifications:  
  Input and output specifications for different scenarios are shown in the table below:

  <table border="2" align="center">
  <tr>
    <th align="center">scenarioNum</th>
    <th align="center">Example Scenario</th>
    <th align="center">Input name</th>
    <th align="center">Input shape</th>
    <th align="center">Input data type</th>
    <th align="center">Output name</th>
    <th align="center">Output shape</th>
    <th align="center">Output data type</th>
  </tr>
  <tr>
    <td align="center">1</td>
    <td align="center">ReduceMax first n elements computation</td>
    <td align="center">x</td>
    <td align="center">[1, 288]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">2</td>
    <td align="center">ReduceMax high-dimensional tiling computation</td>
    <td align="center">x</td>
    <td align="center">[1, 512]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">3</td>
    <td align="center">ReduceMin first n elements computation</td>
    <td align="center">x</td>
    <td align="center">[1, 288]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">4</td>
    <td align="center">ReduceMin high-dimensional tiling computation</td>
    <td align="center">x</td>
    <td align="center">[1, 512]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">5</td>
    <td align="center">ReduceSum first n elements computation</td>
    <td align="center">x</td>
    <td align="center">[1, 288]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  <tr>
    <td align="center">6</td>
    <td align="center">ReduceSum high-dimensional tiling computation</td>
    <td align="center">x</td>
    <td align="center">[8320]</td>
    <td align="center">half</td>
    <td align="center">y</td>
    <td align="center">[1, 16]</td>
    <td align="center">half</td>
  </tr>
  </table>

- Example implementation:

  - Kernel implementation
    - Calls DataCopy basic API to move data from GM (Global Memory) to UB (Unified Buffer), and moves the reduction computation results back to GM (Global Memory).
    - Calls ReduceMax/ReduceMin/ReduceSum interfaces to complete reduction computation.
    - In the ReduceSum first n elements computation scenario, calls [GetReduceRepeatSumSpr](../../../../../../docs/api/context/GetReduceRepeatSumSpr(ISASI).md) to obtain computation results.

- Invocation implementation  
  Uses kernel caller `<<<>>>` to invoke the kernel function.

## Build and Run

Execute the following steps in the example root directory to build and run the example.

- Configure environment variables  
  Select the appropriate command based on the [installation method](../../../../../../docs/en/quick_start.md#prepare&install) of the CANN development toolkit in your current environment.
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

- Example execution
  ```bash
  SCENARIO=1
  mkdir -p build && cd build;                                               # Create and enter build directory
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j;    # Build project, default npu mode
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO                     # Generate test input data
  ./demo                                                                    # Execute the compiled program to run the example
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO output/output.bin output/golden.bin   # Verify output correctness, confirm algorithm logic is correct
  ```

  When using NPU simulation mode, add `-DCMAKE_ASC_RUN_MODE=sim` parameter.
  
  Example:
  ```bash
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..;make -j; # NPU simulation mode
  ```

  > **Note:** Before switching build modes, clean the cmake cache by running `rm CMakeCache.txt` in the build directory, then re-run cmake.

- Build options description

  | Option             | Values                         | Description                                                                                                                             |
  | -------------------| -------------------------------| ----------------------------------------------------------------------------------------------------------------------------------------|
  | `CMAKE_ASC_RUN_MODE` | `npu` (default), `sim` | Run mode: NPU run, NPU simulation                                                                                                        |
  | `CMAKE_ASC_ARCHITECTURES` | `dav-2201` (default), `dav-3510` | NPU architecture: dav-2201 corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products, dav-3510 corresponds to Ascend 950PR/Ascend 950DT |
  | `SCENARIO_NUM` | `1`, `2`, `3`, `4`, `5`, `6`         | Scenario number: 1=ReduceMax first n elements computation, 2=ReduceMax high-dimensional tiling computation, 3=ReduceMin first n elements computation, 4=ReduceMin high-dimensional tiling computation, 5=ReduceSum first n elements computation, 6=ReduceSum high-dimensional tiling computation |

- Execution result

  The following result indicates successful accuracy comparison:
  ```bash
  test pass!
  ```