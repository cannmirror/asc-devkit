# MakeUBmemPtr

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

构造Coord对象，用于定义张量的坐标。

## 函数原型

```cpp
template <typename T, typename U>
__aicore__ inline auto MakeUBmemPtr(const U& byteOffset)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| T | 输入 |数据类型 。 |
| U | 输入 | 待偏移的数据类型。 |
| byteOffset | 输入 | 偏移量。 |

## 返回值说明

返回对应内存的地址。

## 约束说明

- 参数数量必须与对应的Shape维度数量一致。
- 各参数必须为非负整数。
- 坐标值必须在对应Shape维度的有效范围内。
- 支持的数据类型包括：size_t、int等整数类型。

## 调用示例

```cpp
```