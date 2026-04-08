# Layout简介<a name="ZH-CN_TOPIC_0000002331693889"></a>

Layout<Shape, Stride>数据结构是描述多维张量内存布局的基础模板类，通过编译时的形状（Shape）和步长（Stride）信息，实现逻辑坐标空间到一维内存地址空间的映射，为复杂张量操作和硬件优化提供基础支持。借助模板元编程技术，该类在编译时完成计算和代码生成，从而降低运行时开销。

Layout包含两个核心组成部分：

-   Shape：定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。
-   Stride：定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与Shape的维度信息一一对应。

例如，一个二维矩阵的Shape为\(4, 2\)，Stride为\(4, 1\)，表示：

-   矩阵有4行、2列。
-   列方向上的步长为1，即每行中相邻元素在内存中的间隔为1个元素；行方向上的步长为4，即相邻行的起始地址间隔为4个元素。

[表1](#table12525201245615)中给出了一维内存地址空间视图，[表2](#table17519406546)中给出了该二维矩阵的逻辑视图。

**表 1** **线性地址视图**

<a name="table12525201245615"></a>
| 地址 | 0 | 1 | 2、3 | 4 | 5 | 6、7 | 8 | 9 | 10、11 | 12 | 13 |
|------|---|---|------|---|---|------|---|---|--------|----|----|
| 元素 | a00 | a01 | - | a10 | a11 | - | a20 | a21 | - | a30 | a31 |

**表 2** **矩阵逻辑视图**

<a name="table17519406546"></a>
| 索引 | 列0 | 列1 |
|------|-----|-----|
| 行0 | a00 (地址0) | a01 (地址1) |
| 行1 | a10 (地址4) | a11 (地址5) |
| 行2 | a20 (地址8) | a21 (地址9) |
| 行3 | a30 (地址12) | a31 (地址13) |

## 需要包含的头文件<a name="zh-cn_topic_0000002213064918_section78885814919"></a>

```
#include "kernel_operator_layout.h"
```

## 原型定义<a name="section10580930144614"></a>

```
template <typename ShapeType, typename StrideType>
struct Layout : private Std::tuple<ShapeType, StrideType> {
    __aicore__ inline constexpr Layout(const ShapeType& shape  = {}, const StrideType& stride = {}) : Std::tuple<ShapeType, StrideType>(shape, stride) {}

    __aicore__ inline constexpr decltype(auto) layout() {}
    __aicore__ inline constexpr decltype(auto) layout() const {}
    
    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Shape() {}
    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Shape() const {}

    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Stride() {}
    template <size_t... I>
    __aicore__ inline constexpr decltype(auto) Stride() const {}
    
    template <typename CoordType>
    __aicore__ inline constexpr auto operator()(const CoordType& coord) const {}
}
```

## 模板参数<a name="section116801320102618"></a>

**表 3**  模板参数说明

<a name="table13588175515344"></a>
| 参数名 | 描述 |
|--------|------|
| ShapeType | [`Std::tuple`](../../../容器函数.md)结构类型，用于定义数据的逻辑形状，例如二维矩阵的行数和列数或多维张量的各维度大小。 |
| StrideType | [`Std::tuple`](../../../容器函数.md)结构类型，用于定义各维度在内存中的步长，即同维度相邻元素在内存中的间隔，间隔的单位为元素，与Shape的维度信息一一对应。 |

