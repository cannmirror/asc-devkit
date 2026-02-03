# DataCopy Tensor随路量化激活搬运样例

## 概述

本样例基于DataCopy实现数据搬运，支持在数据搬运过程中进行tensor量化和Relu激活等操作。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── tensor_quantized_activation_during
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── tensor_quantized_activation_during.asc      // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  

  支持在数据搬运过程中进行量化和Relu激活等操作，同时支持Local Memory到Global Memory通路NZ到ND格式的转换。
- 算子规格：  
  <table>
  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">4 * 4 * 32</td><td align="center">half</td><td align="center">NZ</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">z</td><td align="center">8 * 9 * 16</td><td align="center">half</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">kernel_fixpipe_custom</td></tr>
  </table>


- 算子实现：  
  - Kernel实现   
    Mmad左矩阵和右矩阵的数据类型为int8_t，结果矩阵的数据类型为int32_t。量化模式DEQF16，tensor量化参数为1，将Mmad计算出的结果由int32_t量化成half并搬出。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行  

在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  python3 ../scripts/verify_result.py output/output.bin output/golden_data.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```