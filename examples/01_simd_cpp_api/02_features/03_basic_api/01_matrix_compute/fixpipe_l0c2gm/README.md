# fixpipe_l0c2gm样例

## 概述

本样例介绍如何使用Fixpipe将矩阵乘的结果从L0C（CO1 Buffer）搬出到GM（Global Memory），支持多种输出格式（NZ、ND、DN）、数据类型转换、随路量化、ReLU以及ChannelSplit等功能。这些接口用于将L0C中的矩阵乘计算结果高效地传输到全局内存，并支持各种数据格式转换和预处理能力。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── fixpipe_l0c2gm
│   ├── scripts
│   │   ├── gen_data.py                // 输入数据和真值数据生成脚本
│   │   └── verify_result.py           // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                 // 编译工程文件
│   ├── data_utils.h                   // 数据读入写出函数
│   └── fixpipe_l0c2gm.asc             // Ascend C样例实现 & 调用样例
```

## 场景详细说明

本样例通过编译参数 `SCENARIO_NUM` 选择不同的输出场景，所有场景基于相同的矩阵乘规格：[M, N, K] = [128, 256, 128]，核函数名为 `fixpipe_l0c2gm`。

**场景1：输出格式NZ，输出数据类型float**
- 输入：A [128, 128] half类型，ND格式；B [128, 256] half类型，ND格式
- 输出：C [128, 256] float类型，NZ格式
- 实现：使用 `Fixpipe<outputType, l0cType, AscendC::CFG_NZ>` 将数据从CO1搬出到GM，输出为NZ格式
- 说明：CO1数据为NZ格式直接输出到GM的NZ格式，数据保持原格式不变

**场景2：输出格式ND，输出数据类型float**
- 输入：A [128, 128] half类型，ND格式；B [128, 256] half类型，ND格式
- 输出：C [128, 256] float类型，ND格式
- 实现：使用 `Fixpipe<outputType, l0cType, AscendC::CFG_ROW_MAJOR>` 指定ROW_MAJOR格式转换
- 说明：将CO1中的NZ格式数据转换为ND格式输出到GM

**场景3：输出格式DN，输出数据类型float**
- 输入：A [128, 128] half类型，ND格式；B [128, 256] half类型，ND格式
- 输出：C [256, 128] float类型，DN格式
- 实现：使用 `Fixpipe<outputType, l0cType, AscendC::CFG_COLUMN_MAJOR>` 指定COLUMN_MAJOR格式转换
- 说明：将CO1中的NZ格式数据转换为DN格式输出到GM（仅Ascend 950PR/Ascend 950DT支持）

**场景4：输出格式ND，输出数据类型int8_t，使能Scalar量化**
- 输入：A [128, 128] half类型，ND格式；B [128, 256] half类型，ND格式
- 输出：C [128, 256] int8_t类型，ND格式
- 实现：设置 `fixpipeParams.quantPre = QuantMode_t::QF322B8_PRE`，使用Scalar量化模式
- 说明：将float类型数据量化为int8_t类型，整个C矩阵使用一个量化参数

**场景5：输出格式ND，输出数据类型int8_t使能Vector量化**
- 输入：A [128, 128] half类型，ND格式；B [128, 256] half类型，ND格式
- 输出：C [128, 256] int8_t类型，ND格式
- 实现：设置 `fixpipeParams.quantPre = QuantMode_t::VQF322B8_PRE`，使用Vector量化模式，并通过quantAlphaTensor传入每列的量化参数
- 说明：将float类型数据量化为int8_t类型，C矩阵的每一列对应一个量化参数，使用的量化参数需要从GM拷贝量化参数到L1

**场景6：输出格式ND，输出数据类型float使能ReLU**
- 输入：A [128, 128] half类型，ND格式；B [128, 256] half类型，ND格式
- 输出：C [128, 256] float类型，ND格式
- 实现：设置 `fixpipeParams.reluEn = true` 开启ReLU功能
- 说明：在数据从CO1搬出到GM的过程中执行ReLU操作，即将负值置为0

**场景7：输出格式NZ，输出数据类型float使能ChannelSplit**
- 输入：A [128, 128] half类型，ND格式；B [128, 256] half类型，ND格式
- 输出：C [128, 512] float类型，NZ格式（使能通道拆分）
- 实现：设置 `fixpipeParams.isChannelSplit = true` 开启ChannelSplit功能
- 说明：在数据从CO1搬出到GM的过程中开启通道拆分功能，即将16x16小z分型矩阵拆分成两个独立的16X8小z分型矩阵输出到GM，Fixpipe接口的输入输出必须均为float类型，且仅支持NZ格式

| scenarioNum | L0C数据类型 | 输出数据类型 | 输出格式 | 是否使能ReLU | 是否使能ChannelSplit |
|:-----------:|:----------:|:-----------:|:--------:|:------------:|:--------------------:|
| 1 | float | float | NZ | 否 | 否 |
| 2 | float | float | ND | 否 | 否 |
| 3 | float | float | DN | 否 | 否 |
| 4 | float | int8_t | ND | 否 | 否 |
| 5 | float | int8_t | ND | 否 | 否 |
| 6 | float | float | ND | 是 | 否 |
| 7 | float | float | NZ | 否 | 是 |

## 编译运行

在本样例根目录下执行如下步骤，编译并执行样例。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- 样例执行
  ```bash
  SCENARIO_NUM=1
  mkdir -p build && cd build;      # 创建并进入build目录
  cmake .. -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # 编译工程（默认dav-2201 NPU模式）
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin ./output/golden.bin  # 验证输出结果是否正确
  ```
    使用NPU仿真模式时，添加 `-DRUN_MODE=sim` 参数即可。

  示例如下：
  ```bash
  cmake -DRUN_MODE=sim -DNPU_ARCH=dav-2201 -DSCENARIO_NUM=$SCENARIO_NUM ..;make -j; # NPU仿真模式
  ```

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明
  | 选项 | 可选值 | 说明 |
  |------|--------|------|
  | `RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
  | `NPU_ARCH` | `dav-2201`（默认）、`dav-3510` | NPU 架构，dav-2201 对应 Atlas A2/A3 系列，dav-3510 对应 Ascend 950PR/Ascend 950DT |

  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```