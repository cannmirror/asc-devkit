# ZzLayoutFormat

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

ZzLayoutFormat用于定义ZZ格式的布局，ZZ格式是一种特殊的分块存储格式。

## 结构体定义

```cpp
template <typename T>
struct ZzLayoutFormat {
    template <size_t row, size_t column>
    using type = ZZFormatLayout<T, row, column>;

    template <typename U, typename S>
    __aicore__ inline decltype(auto) operator()(U row, S column) {
        return MakeZzLayout<T, U, S>(row, column);
    }  
};
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|------|------|
| row | 输入 | 内层矩阵的行数，固定为1。 |
| column | 输入 | 内层矩阵的列数，固定为1。 |

## 约束说明

- T必须是有效的数据类型，如half、float、int32_t等。
- 内层矩阵的大小固定为(32 / sizeof(T)) * 16 。

## 调用示例

```cpp
// 创建Zz格式Layout
using T = half;
size_t mLength = 128;
size_t kLength = 64;

auto layout = AscendC::MakeZzLayout<T>(mLength, kLength);
```