# Int

## 产品支持情况

| 产品 | 是否支持 |
| ----------- | :----: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

Int用于表示编译期整数常量。除通用形式Int<v>外，还提供了一组常用别名，如_0、_1、_16、_32、_64等简写方式。

Int提供了以下配套接口：
- 算术运算：+、-、*、/、%
- 比较函数：max、min
- 数学函数：divide、ceil_division、ceil_align

## 原型定义

```cpp
template <size_t v>
using Int = integral_constant<size_t, v>;
```

## 参数说明

| 参数名 | 类型 | 描述 |
|--------|------|------|
| v | 输入 | 编译期非负整数常量值。 |

## 返回值说明

Int<v>本身是一个类型别名，表示值为v的编译期整数常量类型。

## 约束说明

v必须是编译期常量。

## 调用示例

```cpp
using namespace AscendC::Std;

using A = Int<16>;
using B = Int<32>;

auto a = A{};
auto b = B{};

auto sum = a + b;                         // Int<48>
auto product = _2{} * _16{};             // Int<32>
auto q = ceil_division(Int<33>{}, _16{}); // Int<3>
auto aligned = ceil_align(Int<33>{}, _16{}); // Int<48>
auto mx = max(_16{}, _32{});             // Int<32>
auto mn = min(_16{}, _32{});             // Int<16>
```
