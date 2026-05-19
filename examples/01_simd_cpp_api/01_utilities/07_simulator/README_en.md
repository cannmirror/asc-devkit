# CAmodel Simulation Sample Based on MatmulLeakyRelu Operator

## Overview

This sample focuses on CAmodel simulation and problem analysis workflow based on the MatmulLeakyRelu operator. Users can quickly identify software and hardware performance bottlenecks of operators based on the output performance data, improving operator performance analysis efficiency.

## Supported Products

- Ascend 950PR/Ascend 950DT
- Atlas A3 Training Series Products/Atlas A3 Inference Series Products
- Atlas A2 Training Series Products/Atlas A2 Inference Series Products

## Directory Structure

```
├── 07_simulator
│   ├── CMakeLists.txt          # Build project file
│   ├── data_utils.h            # Data read/write functions
│   ├── matmul_leakyrelu.asc    # Ascend C operator implementation and call sample
│   └── scripts
│       ├── gen_data.py         # Input data and golden data generation script
│       └── verify_result.py    # Golden comparison script
```

## Operator Description

The MatmulLeakyRelu operator implements matrix multiply-add (Matmul) combined with LeakyRelu activation function computation. For detailed functional description, refer to the [MatmulLeakyRelu Operator Details](../../00_introduction/03_fusion_operation/matmul_leakyrelu/README.md) section.

## Build and Run

Execute the following steps in the sample root directory to build and run the operator.

- Configure environment variables

  Select the appropriate command to configure environment variables based on the [installation method](../../../../docs/en/quick_start.md#prepare&install) of the CANN development kit on the current environment.

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

- Sample execution

  ```bash
  mkdir -p build && cd build
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-2201 ..; make -j
  python3 ../scripts/gen_data.py
  msprof op simulator --soc-version=Ascend910B1 ./demo    # Execute operator tuning via msprof op simulator
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin
  ```

  Select the appropriate `CMAKE_ASC_ARCHITECTURES` parameter based on the actual NPU hardware architecture being tested.

  - Build option description

    | Option | Description |
    | ------ | ----------- |
    | `CMAKE_ASC_RUN_MODE` | Set to `sim` to enable NPU simulation mode |
    | `CMAKE_ASC_ARCHITECTURES` | Specify the NPU architecture version. CMake will configure the corresponding CPU debugging dependency libraries based on this value. `dav-2201` corresponds to Atlas A2 Training Series Products/Atlas A2 Inference Series Products and Atlas A3 Training Series Products/Atlas A3 Inference Series Products. `dav-3510` corresponds to Ascend 950PR/Ascend 950DT |

  The following output indicates successful accuracy comparison:

  ```bash
  test pass!
  ```

## Simulation Tuning

Based on `./demo`, you can perform simulation performance analysis using msprof op simulator to generate visualized instruction pipeline diagrams and other information. The command is as follows:

```bash
msprof op simulator --soc-version=<soc_version> ./demo
```

  > Obtain the AI processor model <soc_version> as follows:
  > - For the following product models: Execute the `npu-smi info` command on the server with the Ascend AI processor installed to query the **Name** information. The actual configuration value is AscendName. For example, if **Name** is xxxyy, the actual configuration value is Ascendxxxyy.
  >   - Atlas A2 Training Series Products / Atlas A2 Inference Series Products
  >
  > - For the following product models, execute the `npu-smi info -t board -i <id> -c <chip_id>` command on the server with the Ascend AI processor installed to query the **Chip Name** and **NPU Name** information. The actual configuration value is Chip Name_NPU Name. For example, if **Chip Name** is Ascendxxx and **NPU Name** is 1234, the actual configuration value is Ascendxxx_1234. Where:
  >
  >   id: device ID, the NPU ID from the `npu-smi info -l` command is the device ID
  >
  >   chip_id: chip ID, the Chip ID from the `npu-smi info -m` command is the chip ID
  >   - Ascend 950PR/Ascend 950DT
  >   - Atlas A3 Training Series Products / Atlas A3 Inference Series Products

After the command completes, a folder named `OPPROF_{timestamp}_XXX` will be generated in the current directory with the following structure:

```
OPPROF_{timestamp}_XXX/
├── dump                    # Raw performance data, users do not need to focus on this
└── simulator
  ├── core0.cubecore0/     # Simulation instruction pipeline diagram files for each core
  ├── core0.veccore0/
  ├── core0.veccore1/
  ├── trace.json           # Simulation pipeline diagram and hotspot function visualization file
  └── visualize_data.bin   # MindStudio Insight presentation file
```

After execution, you can view the instruction pipeline diagram in the following ways:
- **MindStudio Insight**: Open `visualize_data.bin` or `trace.json` for visualization
- **Chrome Browser**: Enter `chrome://tracing` in the address bar and drag the `trace.json` file to the blank area

For more information on using the msProf tool, refer to the Operator Tuning (msProf) section in [MindStudio Tools](https://www.hiascend.com/document/redirect/CannCommercialToolOpDev).