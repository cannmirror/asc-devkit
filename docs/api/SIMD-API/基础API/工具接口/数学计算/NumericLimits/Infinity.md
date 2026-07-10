# Infinity

## 产品支持情况

<!-- npu="950" id9 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id9 -->
<!-- npu="A3" id10 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id10 -->
<!-- npu="910b" id11 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id11 -->
<!-- npu="310b" id12 -->
- Atlas 200I/500 A2 推理产品：不支持
<!-- end id12 -->
<!-- npu="310p" id13 -->
- Atlas 推理系列产品AI Core：不支持
<!-- end id13 -->
<!-- npu="310p" id14 -->
- Atlas 推理系列产品Vector Core：不支持
<!-- end id14 -->
<!-- npu="910" id15 -->
- Atlas 训练系列产品：不支持
<!-- end id15 -->

## 功能说明

头文件路径为：`"basic_api/kernel_operator_limits_intf.h"`。

返回指定数据类型的正无穷大值。

## 函数原型

- 标量接口，返回值为标量：

    ```cpp
    constexpr __aicore__ static inline T Infinity()
    ```

- 矢量接口，为dstLocal前count个元素赋正无穷大值：

    ```cpp
    __aicore__ static inline void Infinity(const LocalTensor<T>& dstLocal, uint32_t count)
    ```

## 参数说明

**表1** 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dstLocal | 输出 | 目的操作数。<br><br>类型为[LocalTensor](../../../数据结构/LocalTensor和GlobalTensor定义/LocalTensor/LocalTensor.md)，支持的TPosition为VECIN/VECCALC/VECOUT。<br><br>LocalTensor的起始地址需要32字节对齐。 |
| count | 输入 | 输入数据元素个数。 |

## 数据类型

<!-- npu="950" id1 -->
针对Ascend 950PR/Ascend 950DT，支持的数据类型为：half、bfloat16_t、float。
<!-- end id1 -->

## 返回值说明

标量接口返回值为对应数据类型的正无穷大值。

矢量接口无返回值。

## 约束说明

无。

## 调用示例

- 标量接口：

    ```cpp
    float value = AscendC::NumericLimits<float>::Infinity();
    ```

- 矢量接口：

    ```cpp
    AscendC::NumericLimits<float>::Infinity(dstLocal, 256); // 返回256个float类型的正无穷大值
    ```
