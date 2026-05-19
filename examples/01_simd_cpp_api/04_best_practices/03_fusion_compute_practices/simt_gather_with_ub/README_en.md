# SIMD and SIMT Hybrid Programming Using UB to Improve Memory Access Efficiency

## Overview

This sample implements a gather operator using SIMD and SIMT hybrid programming, retrieving 65536 data elements at specified indices from a one-dimensional vector of length 8192. It pre-transfers input data to UB, demonstrating performance optimization using UB to improve discrete memory access efficiency in SIMD and SIMT hybrid programming mode.

## Supported Products

- Ascend 950PR/Ascend 950DT

## Supported CANN Software Versions
- \>= CANN 9.0.0-beta.2

## Directory Structure

```
├── 15_simt_gather_with_ub
│   ├── CMakeLists.txt         # cmake build file
│   ├── gather_v1.asc          # Ascend C gather operator sample with direct GM access
│   ├── gather_v2.asc          # Ascend C gather operator sample using UB
|   └── README.md
```

## Operator Description

- Operator Functionality:  
  The gather operator implements the function of retrieving 65536 data elements at specified indices from a one-dimensional vector of length 8192. The calculation formula for the i-th data element in operator output is:
  
  ```
  output[i] = input[index[i]]
  ```

- Operator Specification:  
  <table>
  <tr><td rowspan="1" align="center">Operator Type(OpType)</td><td colspan="4" align="center">gather</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">Operator Input</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">input</td><td align="center">8192</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">index</td><td align="center">65536</td><td align="center">uint32_t</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">Operator Output</td><td align="center">output</td><td align="center">65536</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">Kernel Function Name</td><td colspan="4" align="center">gather_kernel</td></tr>
  </table>

- Thread Hierarchy Structure:
  * Number of thread blocks: 64
  * Number of threads per thread block: 1024

- Operator Implementation:  
  simt_gather is responsible for retrieving data at specified indices from input. First, calculate the index of data the thread should process, then store the data to Global Memory through assignment operation.
  In the v1 version, the SIMT kernel function reads data directly from Global Memory; in the v2 version, input data is pre-transferred to UB, and the SIMT kernel function reads input from UB to complete the gather operation.
  ```
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  ...
  uint32_t gather_idx = index[idx];
  ...
  output[idx] = input[gather_idx];
  ```

- Performance Benefit:  
  Using the msprof tool to collect on-board performance data, the v1 version operator average runtime is approximately 4.56us, the v2 version operator average runtime is approximately 3.57us, with performance improvement of approximately 21.71%.

## Build and Run

Execute the following steps in the root directory of this sample to build and run the operator.
- Configure environment variables  
  Select the appropriate environment variable configuration command based on the [installation method](../../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit package in the current environment.
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
    
- Sample execution  
  v1 version execution method:
  ```bash
  mkdir -p build && cd build;              # Create and enter build directory
  cmake -DGATHER_VERSION=v1 ..; make -j;   # Build project
  ./gather                                 # Execute sample
  ```
  v2 version execution method:
  ```bash
  mkdir -p build && cd build;              # Create and enter build directory
  cmake -DGATHER_VERSION=v2 ..; make -j;   # Build project
  ./gather                                 # Execute sample
  ```
  Execution result shown below indicates accuracy comparison passed.
  ```
  [Success] Case accuracy is verification passed.
  ```

## Performance Tuning

The operator tuning tool supports on-board tuning and simulation tuning modes, which can obtain operator performance data in actual hardware environment and simulation environment respectively, used for identifying performance bottlenecks and optimizing operator implementation.

### On-board Tuning

Based on the compiled executable file, collect operator performance data directly on NPU hardware, including operator basic information and memory load analysis.

**Operation Steps**

**1. Execute tuning command**

Run the operator tuning tool based on the compiled gather file.
```bash
msprof op ./gather
```

**2. View performance data**

A folder with OPPPROF_ prefix will be generated in the current directory. The directory structure and file descriptions are as follows:
```bash 
OPPROF_202xxxxx_XXXXXX
├── dump                             # Raw performance data (no need to focus on)
├── OpBasicInfo.csv                  # Operator basic data
├── ArithmeticUtilization.csv        # Cube and vector type instruction cycle ratio data
├──  ResourceConflictRatio.csv       # Resource conflict ratio data
├── ... (enabled aic-metrics)
└──  visualize_data.bin              # Operator visualization file (can be loaded via MindStudio Insight to visually view operator performance)
```
You can open the `visualize_data.bin` file through MindStudio Insight tool to visually view performance data.

### Simulation Tuning

In scenarios without NPU hardware environment, obtain operator simulation performance data including instruction pipeline diagrams by compiling the simulation operator executable and combining with the simulator.

**Operation Steps**

**1. Simulation operator compilation**

```bash
mkdir -p build && cd build
# Replace ${SOC_VERSION} with actual NPU model, can be queried via npu-smi info command, for example Ascend950PR_9599.
# Replace ${GATHER_VERSION} with operator version, value is v1 or v2.
cmake -DGATHER_VERSION=${GATHER_VERSION} -DRUN_MODE=sim -DSOC_VERSION=${SOC_VERSION} ..
make -j
```

**2. Configure runtime dependencies**

Add runtime dependency library path (replace {SOC_VERSION} with actual NPU model):
```bash
export LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib/:$LD_LIBRARY_PATH
```

**3. Execute simulation tuning command**

```bash
msprof op simulator ./gather
```

**4. View simulation performance data**

A folder with OPPROF_ prefix will be generated in the current directory. The directory structure is as follows:
```bash
OPPROF_202xxxxx_XXXXXX
├── dump                                    # Raw performance data, no need to focus on
└── simulation                              # Simulation performance data analysis results
    ├── core0.veccore0                      # Operator block-level sub-core
        ├── core0.veccore1_code_exe.csv     # Code line duration
        ├── core0.veccore1_instr_exe.csv    # Program code instruction details
        └── trace.json                      # Operator block-level sub-core pipeline diagram
    ├── ...
    ├── visualize_data.bin                  # Operator visualization file (can be loaded via MindStudio Insight to visually view operator performance)
    └── trace.json                          # Pipeline diagram for all operator cores
```
You can open the `visualize_data.bin` file through MindStudio Insight tool to visually view performance data.

**Additional Notes**
For detailed explanations of more performance metrics and tuning solutions, please refer to the "Operator Development Tools" manual.