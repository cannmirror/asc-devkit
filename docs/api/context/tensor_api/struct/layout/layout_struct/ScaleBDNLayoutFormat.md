# ScaleBDNLayoutFormat

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

ScaleBDNLayoutFormat用于定义ScaleBDN格式的布局，ScaleBDN格式是一种支持缩放和BDN格式的布局。

## 结构体定义

```cpp
template <typename T>
struct ScaleBDNLayoutFormat {
    template <size_t row, size_t column>
    using type = ScaleBDNFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeScaleBDNLayout<T, U, S>(row, column);
    }  
};
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|------|------|
| row | 输入 | 内层矩阵的行数，固定为1。 |
| column | 输入 | 内层矩阵的列数，固定为1。 |

## 约束束说明

- T必须是有效的数据类型，如half、float、int32_t等。
- ScaleBDN格式不使用分块存储。

## 调用示例

```cpp
// 创建ScaleBDN格式Layout
using T = half;
size_t mLength = 128;
size_t nLength = 64;

auto layout = AscendC::MakeScaleBDNLayout<T>(mLength, nLength);
```