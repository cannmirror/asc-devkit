# MakeTile

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

构造Tile对象，用于定义张量的分块。

## 函数原型

```cpp
template <typename... Ts>
__aicore____ inline constexpr Tile<Ts...> MakeTile(const Ts&... t)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| t | 输入 | 各维度的分块大小，可变参数。 |

## 返回值说明

返回Tile<Ts...>对象。

## 约束说明

- 参数数量必须与对应的Shape维度数量一致。
- 各参数必须为正整数。
- 分块大小不能超过对应Shape维度的大小。
- 支持的数据类型包括：size_t、int等整数类型。

## 调用示例

```cpp
// 创建一个3维张量的分块
auto tile = AscendC::MakeTile(2, 5, 3);

// 获取各维度的分块大小
auto tile0 = AscendC::Std::get<0>(tile.value); // tile0 = 2
auto tile1 = AscendC::Std::get<1>(tile.value); // tile1 = 5
auto tile2 = AscendC::Std::get<2>(tile.value); // tile2 = 3
```