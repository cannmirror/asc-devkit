# Async

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
- Atlas 推理系列产品AI Core：不支持
<!-- end id12 -->
<!-- npu="310p" id13 -->
- Atlas 推理系列产品Vector Core：不支持
<!-- end id13 -->
<!-- npu="910" id14 -->
- Atlas 训练系列产品：不支持
<!-- end id14 -->

## 功能说明

头文件路径为：`"basic_api/kernel_operator_utils_intf.h"`。

基于[分离模式](https://gitcode.com/cann/asc-devkit/blob/9.1.0/docs/guide/编程指南/高级编程/硬件实现/基本架构.md)（AIC和AIV分离）开发融合算子时，算子逻辑中通常同时包含AIV和AIC的处理逻辑，此时需要使用Ascend C提供的宏`ASCEND_IS_AIV`/`ASCEND_IS_AIC`实现如下硬件条件分支来对AIV和AIC的代码进行隔离：

```cpp
if ASCEND_IS_AIV {
// AIV处理逻辑
}
if ASCEND_IS_AIC {
// AIC处理逻辑
}
```

Async通过模板函数的方式对这种隔离模式进行了封装，提供了一个统一的接口，用于在不同执行单元（AIC或AIV）下执行特定函数，从而避免在代码中使用硬件条件分支。

## 函数原型

```cpp
template <EngineType engine, auto funPtr, class... Args>
__aicore__ void Async(Args... args)
```

## 参数说明

**表1** 模板参数说明

| 参数名        | 描述                                                                |
| ------------- | ------------------------------------------------------------------- |
| engine        | EngineType枚举类型，可取值如下，对应不同的硬件执行单元：<br>&bull; AIC <br>&bull; AIV |
| funPtr        | 函数指针，指定要执行的函数，函数签名和参数类型由class... Args决定。 |
| class... Args | 可变参数模板，表示函数参数的类型列表，用于传递给funPtr。            |

**表2** 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| Args... args | 输入 | 与class... Args对应的参数列表，表示传递给funPtr的实际参数。 |

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
__aicore__ inline void cubeProcess(KernelMmad &op, GM_ADDR A, GM_ADDR B, GM_ADDR c)
{
    op.InitAIC(A, B, c);
    op.ProcessAIC();
}
__aicore__ inline void vectorProcess(KernelMmad &op, GM_ADDR a, GM_ADDR b, GM_ADDR A, GM_ADDR B, GM_ADDR c)
{
    op.InitAIV(a, b, A, B, c);
    op.ProcessAIV();
}
__global__ __mix__(1,2) void mmad_custom(GM_ADDR a, GM_ADDR b, GM_ADDR A, GM_ADDR B, GM_ADDR c)
{
    AscendC::InitSocState();
    KernelMmad op;
    AscendC::Async<AscendC::EngineType::AIC, cubeProcess>(op, A, B, c);
    AscendC::Async<AscendC::EngineType::AIV, vectorProcess>(op, a, b, A, B, c);
}
```
