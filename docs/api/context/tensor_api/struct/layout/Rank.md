# Rank

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

返回Layout的秩，即Layout的维度数量。

## 函数原型

```cpp
template <size_t... Is, typename Shape, typename Stride>
__aicore__ inline constexpr auto Rank(const Layout<Shape, Stride>& layout)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| layout | 输入 | Layout对象。 |

## 返回值说明

返回Layout的维度数量。

## 约束说明

- layout必须是有效的Layout对象。
- 返回值为size_t类型。

## 调用示例

```cpp
auto shape = AscendC::MakeShape(10, 20, 30);
auto layout = AscendC::MakeLayout(shape);

auto rank = AscendC::Rank(layout); // rank = 3
```