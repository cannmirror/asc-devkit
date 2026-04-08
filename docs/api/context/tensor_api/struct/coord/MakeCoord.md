# MakeCoord

## 产品支持情况

| 产品     | 是否支持 |
| ----------- |:----:|
|Ascend 950PR/Ascend 950DT|√|

## 功能说明

构造Coord对象，用于定义张量的坐标。

## 函数原型

```cpp
template <typename... Ts>
__aicore__ inline constexpr Coord<Ts...> MakeCoord(const Ts&... t)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| t | 输入 | 各维度的坐标，可变参数。 |

## 返回值说明

返回Coord<Ts...>对象。

## 约束说明

- 参数数量必须与对应的Shape维度数量一致。
- 各参数必须为非负整数。
- 坐标值必须在对应Shape维度的有效范围内。
- 支持的数据类型包括：size_t、int等整数类型。

## 理论性能说明


## 调用示例

```cpp
// 创建一个3维张量的坐标
auto coord = AscendC::MakeCoord(5, 10, 15);

// 获取各维度的坐标
auto coord0 = AscendC::Std::get<0>(coord.value); // coord0 = 5
auto coord1 = AscendC::Std::get<1>(coord.value); // coord1 = 10
auto coord2 = AscendC::Std::get<2>(coord.value); // coord2 = 15
```