# abs_duplicate算子直调样例

## 概述

本样例介绍无DataCopyPad的非对齐abs_duplicate算子实现，采用核函数<<<>>>调用，有效降低调度开销，实现高效的算子执行。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── abs_duplicate  
│   ├── scripts
│   │   ├── gen_data.py             // 输入数据和真值数据生成脚本
│   │   └── verify_result.py        // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── data_utils.h                // 数据读入写出函数
│   └── abs_duplicate_custom.asc    // Ascend C算子实现 & 调用样例
```

## 算子描述

- 算子功能：  
  本样例中实现的是固定shape为64*11的非对齐Abs算子。
  Abs算子的数学表达式为：
  ```
  z = abs(x)
  ```
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Abs</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td>
  <td align="center">name</td>
  <td align="center">shape</td>
  <td align="center">data type</td>
  <td align="center">format</td></tr>
  <tr><td align="center">x</td>
  <td align="center">11*64</td>
  <td align="center">half</td>
  <td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td>
  <td align="center">11*64</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">abs_duplicate_custom</td></tr>
  </table>
- 算子实现：  
  本样例中实现的是固定shape为64*11的非对齐Abs算子。  
  Abs算子的数学表达式为：
  ```
  z = abs(x)
  ```
  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成对输入参数取绝对值的运算，得到最终结果，再搬出到外部存储上。

  Abs算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入inputGM搬运到Local Memory，存储在inputLocal中。Compute任务负责先将inputLocal中的数据使用Duplicate对冗余数据进行清零处理，之后对inputLocal执行取绝对值操作，计算结果存储在outputLocal中。CopyOut任务负责将输出数据从outputLocal搬运至Global Memory上的输出outputGM中，为防止数据踩踏本例使用原子加进行搬出。

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