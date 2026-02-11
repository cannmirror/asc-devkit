# Matmul算子直调样例

## 概述

本样例采用Tensor_API接口编写Mmad算子样例。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── 00_tensor_api_matmul
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── tensor_api_matmul.asc              // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  

  本样例中实现的是[M, K, N]固定为[128, 128, 128]的Matmul算子。 
  Matmul算子的数学表达式为：
  $$
  C = A * B
  $$
  其中A的形状为[128, 128]，B的形状为[128, 128]，C的形状为[128, 128]。
- 算子规格：  

  在核函数直调样例中，算子实现支持的shape为：M = 128, N = 128, K = 128。
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">a</td><td align="center">M * K</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">b</td><td align="center">K * N</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">tensor_api_matmul_custom</td></tr>
  </table>
- 算子实现：  
  本样例中实现的是[M, K, N]固定为[128, 128, 128]的Matmul算子。Matmul的计算公式为：C = A * B。  
  A、B为源操作数，A为左矩阵，形状为[M, K]；B为右矩阵，形状为[K, N]。  
  C为目的操作数，存放矩阵乘结果的矩阵，形状为[M, N]。  

  - Kernel实现  
    初始化Tiling数据，从设备全局内存中读取关键参数。   
    流程如下：   
    - 设置输入矩阵A、B和输出矩阵C的全局Tensor，并指定维度。  
    - 初始化Matmul对象，设置接口配置计算参数并执行计算。 

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
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```