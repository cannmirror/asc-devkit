# Broatcast算子直调样例

## 概述

本样例展示了一个支持多种数据类型（如bfloat，int8，float，half等）和多种形状（如(32, 1024)，(8, 1023)等）的输入张量执行逐元素加法。本样例重点演示了：  
1.多核并行计算：通过精细的Tiling策略，将计算任务动态分配到多个核上，并处理核间负载不均（整核与尾核）的场景。  
2.输入广播支持：在计算阶段，可对输入进行指定轴上的维度扩展，使算子能处理形状不完全匹配的输入。  
3.复杂数据切：定义了包含20个参数的Tiling结构，已管理数据总长、数据类型、广播系数、切分块大小、核间任务分配等复杂逻辑，确保计算高效与内存访问优化。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── broadcast  
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── broadcast_custom.asc    // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  Broadcast算子实现了将输入数据按照输出shape进行广播的功能，比如A的shape为(4096,1)，广播的目标shape为(4096,3)，则会将原来的一列扩展为相同的3列, A的shape变为(4096,3)。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Broadcast</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape_range</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">4096 * 1</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">4096 * 3</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">broadcast_custom</td></tr>
  </table>

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```