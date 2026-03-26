# CopyTraits

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|
|Atlas A3 训练系列产品/Atlas A3 推理系列产品|√|
|Atlas A2 训练系列产品/Atlas A2 推理系列产品|√|
|Atlas 200I/500 A2 推理产品|x|
|Atlas 推理系列产品AI Core|x|
|Atlas 推理系列产品Vector Core|x|
|Atlas 训练系列产品|x|
|Atlas 200/300/500 推理产品|x|


## 功能说明

CopyTraits用于定义搬运操作的特性，包括搬运操作类型和相关参数。

## 结构体定义

```cpp
template <typename CopyOperation, typename... CopyOpArgs>
struct CopyTraits {
    using OperationType = CopyOperation;
    std::tuple<CopyOpArgs...> args;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| OperationType | CopyOperation | 复制操作类型。 |
| args | std::tuple<CopyOpArgs...> | 复制操作参数。 |

## 约束说明

- CopyOperation必须是有效的复制操作类型。
- CopyOpArgs的数量和类型必须与CopyOperation的要求匹配。

## 调用示例

```cpp
// 创建CopyTraits
auto copyTraits = AscendC::MakeCopyTraits<CopyOperation>(arg1, arg2, arg3);
```