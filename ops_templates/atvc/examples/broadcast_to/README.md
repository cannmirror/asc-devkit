<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->

## 概述

本样例介绍了利用ATVC实现BroadcastTo单算子并完成功能验证


## 样例支持产品型号：
- Atlas A2训练系列产品

## 目录结构

| 文件名                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [broadcast_to.cpp](./broadcast_to.cpp) | BroadcastTo算子代码实现以及调用样例               |

## 算子描述

BroadcastTo是对输入tensor的指定轴进行广播计算并输出结果的Broadcast类算子。

BroadcastTo算子规格：

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">BroadcastTo</td></tr>

<tr><td rowspan="4" align="center">算子输入</td></tr>
<tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">1 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr></tr>

<tr><td rowspan="2" align="center">算子输出</td></tr>
<tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>

<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">BroadcastCustom</td></tr>
</table>

## 算子运行
在ascendc-api-adv代码仓目录下执行：
```bash
$ cd ./atvc/tests
$ bash run_test.sh broadcast_to
...
Generate golden data successfully.
...
Accuracy verification passed.
```