# Coord

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

Coord用于定义张量的坐标，用于访问张量中特定位置的元素。

## 结构体定义

```cpp
template <typename... Ts>
struct Coord {
    std::tuple<Ts...> value;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| value | std::tuple<Ts...> | 存储各维度坐标的元组。 |

## 约束说明

- Coord的维度数量必须与对应的Shape维度数量一致。
- 各维度的坐标值必须在对应Shape维度的有效范围内。
- 支持的数据类型包括：size_t、int等整数类型。

## 调用示例

```cpp
// 创建一个3维张量的坐标
auto coord = AscendC::MakeCoord(5, 10, 15);

// 获取各维度的坐标
auto coord0 = AscendC::Std::get<0>(coord.value); // coord0 = 5
auto coord1 = AscendC::Std::get<1>(coord.value); // coord1 = 10
auto coord2 = AscendC::Std::get<2>(coord.value); // coord2 = 15
```