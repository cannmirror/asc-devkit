# BlockReduce类接口多场景示例

## 概述

本样例在归约场景下，基于[BlockReduceMax](../../../../../../docs/api/context/BlockReduceMax.md)、[BlockReduceMin](../../../../../../docs/api/context/BlockReduceMin.md)、[BlockReduceSum](../../../../../../docs/api/context/BlockReduceSum.md)实现BlockReduce类接口的多场景归约功能，对输入Tensor的每个datablock（32字节）内所有元素进行归约运算（求最大值、最小值或求和）。

## 支持的产品

- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── block_reduce_min_max_sum
│   ├── scripts
│   │   ├── gen_data.py                    // 输入数据和真值数据生成脚本
│   │   └── verify_result.py              // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                    // 编译工程文件
│   ├── data_utils.h                      // 数据读入写出函数
│   └── block_reduce_min_max_sum.asc      // Ascend C样例实现 & 调用样例
```

## 场景详细说明

本样例通过编译参数 `SCENARIO_NUM` 选择不同的归约场景，所有场景数据格式为 ND，核函数名为 `block_reduce_custom`。

**场景1：BlockReduceMax**
- 输入：[1, 128]个half元素，mask=128（256/sizeof(half)），repeat=1
- 输出：[1, 64]个half元素（前8个为有效值，对应8个datablock各自的最大值）
- 实现：`BlockReduceMax<half>(dstLocal, srcLocal, repeat=1, mask=128, dstRepStride=1, srcBlkStride=1, srcRepStride=8)`
- 说明：对每个datablock内所有元素求最大值，一个datablock处理32字节即16个half元素，128个元素共8个datablock，输出8个最大值存入dstLocal前8个位置，硬件要求输出buffer为64个元素

**场景2：BlockReduceMin**
- 输入：[1, 128]个half元素，mask=128，repeat=1
- 输出：[1, 64]个half元素（前8个为有效值，对应8个datablock各自的最小值）
- 实现：`BlockReduceMin<half>(dstLocal, srcLocal, repeat=1, mask=128, dstRepStride=1, srcBlkStride=1, srcRepStride=8)`
- 说明：对每个datablock内所有元素求最小值，一个datablock处理32字节即16个half元素，128个元素共8个datablock，输出8个最小值存入dstLocal前8个位置，硬件要求输出buffer为64个元素

**场景3：BlockReduceSum**
- 输入：[1, 128]个half元素，mask=128，repeat=1
- 输出：[1, 64]个half元素（前8个为有效值，对应8个datablock各自的求和结果）
- 实现：`BlockReduceSum<half>(dstLocal, srcLocal, repeat=1, mask=128, dstRepStride=1, srcBlkStride=1, srcRepStride=8)`
- 说明：对每个datablock内所有元素求和，源操作数相加采用二叉树方式两两相加，一个datablock处理32字节即16个half元素，128个元素共8个datablock，输出8个求和结果存入dstLocal前8个位置，硬件要求输出buffer为64个元素

## 样例规格

<table border="2" align="center">
<caption>表1：样例输入输出规格（所有场景）</caption>
<tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">[1, 128]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="2" align="center">样例输出</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">y</td><td align="center">[1, 64]</td><td align="center">half</td><td align="center">ND</td></tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">block_reduce_custom</td></tr>
</table>

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
  cmake .. -DSCENARIO_NUM=$SCENARIO_NUM;make -j;    # 编译工程
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM   # 生成测试输入数据
  ./demo                           # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py -scenarioNum=$SCENARIO_NUM ./output/output.bin ./output/golden.bin  # 验证输出结果是否正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```
