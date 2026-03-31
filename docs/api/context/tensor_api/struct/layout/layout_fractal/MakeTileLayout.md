# MakeTileLayout

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

构造分块Layout对象，用于将大张量分割成多个小块。

## 函数原型

```cpp
template <typename Layout, typename TileShape>
__aicore__ inline decltype(auto) MakeTileLayout(const Layout& layout, const TileShape& tileShape)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| layout | 输入 | 原始Layout对象。 |
| tileShape | 输入 | 分块大小，Tile类型。 |

## 返回值说明

返回分块后的Layout对象。

## 约束说明

- TileShape的维度数量必须与layout的Shape维度数量一致。
- 各维度的分块大小不能超过对应Shape维度的大小。
- 分块大小必须为正整数。