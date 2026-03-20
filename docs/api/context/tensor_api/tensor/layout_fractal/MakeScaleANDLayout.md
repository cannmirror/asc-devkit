# MakeScaleANDLayout

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

构造ScaleAND格式的Layout对象。ScaleAND格式是一种支持缩放和ND格式的布局。

## 函数原型

```cpp
template <typename T, typename U, typename S>
__aicore__ inline decltype(auto) MakeScaleANDLayout(size_t row, size_t column)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| row | 输入 | 矩阵的行数。 |
| column | 输入 | 矩阵的列数。 |

## 返回值说明

返回ScaleAND格式的Layout对象。

## 约束说明

- T必须是有效的数据类型，如half、float、int32_t等。
- row和column必须为正整数。
- ScaleAND格式不使用分块存储。

## 调用示例

```cpp

```