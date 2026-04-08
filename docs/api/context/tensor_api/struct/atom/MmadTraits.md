# MmadTraits

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

MmadTraits用于定义矩阵乘加操作的特性，包括矩阵乘加操作类型和相关参数。

## 结构体定义

```cpp
template <typename MadOperation, typename... MadOpArgs>
struct MmadTraits {
    using OperationType = MadOperation;
    std::tuple<MadOpArgs...> args;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| OperationType | MadOperation | 矩阵乘加操作类型。 |
| args | std::tuple<MadOpArgs...> | 矩阵乘加操作参数。 |

## 约束说明

- MadOperation必须是有效的矩阵乘加操作类型。
- MadOpArgs的数量和类型必须与MadOperation的要求匹配。

## 调用示例

```cpp
// 创建MmadTraits
auto mmadTraits = AscendC::MakeMmadTraits<MmadOperation>(arg1, arg2, arg3);
```