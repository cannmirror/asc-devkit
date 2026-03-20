# MakeL0CLayout

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

构造L0C格式的Layout对象。L0C格式用于L0C Buffer，存储矩阵计算的结果。

## 函数原型

```cpp
template <typename U, typename S>
__aicore__ inline decltype(auto) MakeL0CLayout(size_t row, size_t column)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| row | 输入 | 矩阵的行数。 |
| column | 输入 | 矩阵的列数。 |

## 返回值说明

返回L0C格式的Layout对象。

## 约束说明

- U和S必须是有效的类型。
- row和column必须为正整数。
- 内层矩阵的大小固定为16 * (32 / sizeof(T)) 。

## 调用示例

```cpp

```