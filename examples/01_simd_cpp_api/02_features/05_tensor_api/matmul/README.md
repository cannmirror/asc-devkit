# 基于Tensor_api实现的Matmul样例

## 概述

本样例基于Tensor_API编程方式实现矩阵乘法运算。

## 支持的产品

- Ascend 950PR/Ascend 950DT

## 目录结构介绍

```
├── matmul_quant_relu
│   └── scripts
│       ├── common.py                 // Ascend C数据类型与python数据类型对应关系
│       ├── gen_demo_code_by_case.py  // 泛化样例数据注入
│       ├── gen_golden.py             // 输入数据和真值数据生成脚本文件
│       ├── run_smoke.py              // 泛化测试运行脚本文件
│       └── verify_result.py          // 真值对比文件
│   ├── testcases                    
│       ├── testcases.json            // 用例具体参数
│   ├── CMakeLists.txt                // 编译工程文件
│   ├── data_utils.h                  // 数据读入写出函数
│   ├── run.sh                        // 泛化用例执行脚本
│   └── matmul.asc                    // Ascend C样例实现 & 调用样例
```

## 样例描述

- 样例功能：

  本样例实现了Matmul功能。
  
  1、Matmul功能
  
  本样例实现了多核Matmul功能，Matmul计算规格参考表1。

- 样例规格：

  在核函数直调样例中，样例实现支持的shape具体查看[testcases.json](testcases/testcases.json)。
  <table border="2" align="center">
    <caption>表1：样例规格表</caption>
    <tr>
      <td rowspan="1" align="center">样例类型(OpType)</td>
      <td colspan="4" align="center">Matmul</td>
    </tr>
  </tr>
  <tr>
    <td rowspan="3" align="center">样例输入</td>
    <td align="center">name</td>
    <td align="center">shape</td>
    <td align="center">data type</td>
    <td align="center">format</td>
  </tr>
  <tr>
    <td align="center">a</td>
    <td align="center">[M, K]</td>
    <td align="center">A_TYPE</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td align="center">b</td>
    <td align="center">[K, N]</td>
    <td align="center">B_TYPE</td>
    <td align="center">ND</td>
  </tr>
  <tr>
    <td rowspan="1" align="center">样例输出</td>
    <td align="center">c</td>
    <td align="center">[M, N]</td>
    <td align="center">C_TYPE</td>
    <td align="center">ND</td>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_custom</td></tr>
  </table>

## 编译运行

在本样例根目录下执行如下步骤，编译并执行样例。
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
  bash run.sh   # 执行批跑样例脚本
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  🟢所有测试用例均成功通过！
  ```