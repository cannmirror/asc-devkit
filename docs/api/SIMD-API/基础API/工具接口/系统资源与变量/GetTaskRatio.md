# GetTaskRatio

## 产品支持情况

<!-- npu="950" id8 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id8 -->
<!-- npu="A3" id9 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id9 -->
<!-- npu="910b" id10 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id10 -->
<!-- npu="310b" id11 -->
- Atlas 200I/500 A2 推理产品：不支持
<!-- end id11 -->
<!-- npu="310p" id12 -->
- Atlas 推理系列产品AI Core：支持
<!-- end id12 -->
<!-- npu="310p" id13 -->
- Atlas 推理系列产品Vector Core：不支持
<!-- end id13 -->
<!-- npu="910" id14 -->
- Atlas 训练系列产品：不支持
<!-- end id14 -->

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

[分离模式](https://gitcode.com/cann/asc-devkit/blob/9.1.0/docs/guide/编程指南/高级编程/硬件实现/基本架构.md)下，获取任务启动的Cube Core（AIC）或者Vector Core（AIV）的数量与逻辑AI Core数量的比例。

- 在AIC上调用时返回AIC数量与逻辑AI Core数量的比例。
- 在AIV上调用时返回AIV数量与逻辑AI Core数量的比例。

[耦合模式](https://gitcode.com/cann/asc-devkit/blob/9.1.0/docs/guide/编程指南/高级编程/硬件实现/基本架构.md)下，固定返回1。

## 函数原型

```cpp
__aicore__ inline int64_t GetTaskRatio()
```

## 参数说明

无

## 返回值说明

- 针对[分离模式](https://gitcode.com/cann/asc-devkit/blob/9.1.0/docs/guide/编程指南/高级编程/硬件实现/基本架构.md)，不同Kernel类型下（通过[设置Kernel类型](../../Kernel-Tiling/设置Kernel类型.md)接口设置），在AIC和AIV上调用该接口的返回值如下：

    **表1** 返回值列表

    | Kernel类型 | KERNEL_TYPE_AIV_ONLY | KERNEL_TYPE_AIC_ONLY | KERNEL_TYPE_MIX_AIC_1_2 | KERNEL_TYPE_MIX_AIC_1_1 | KERNEL_TYPE_MIX_AIC_1_0 | KERNEL_TYPE_MIX_AIV_1_0 |
    | --- | --- | --- | --- | --- | --- | --- |
    | AIV | 1 | - | 2 | 1 | - | 1 |
    | AIC | - | 1 | 1 | 1 | 1 | - |

- 针对[耦合模式](https://gitcode.com/cann/asc-devkit/blob/9.1.0/docs/guide/编程指南/高级编程/硬件实现/基本架构.md)，固定返回1。

## 约束说明

无

## 调用示例

```cpp
int64_t ratio = AscendC::GetTaskRatio(); // 返回AIC或AIV与逻辑AI Core数量的比例
AscendC::printf("task ratio is %ld", ratio);
```
