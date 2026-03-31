# Shape

## 功能说明

Shape用于定义张量的形状，描述张量在各维度上的大小。

## 结构体定义

```cpp
template <typename... Ts>
struct Shape {
    std::tuple<Ts...> value;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| value | std::tuple<Ts...> | 存储各维度大小的元组。 |

## 约束说明

- Shape的维度数量不能超过硬件支持的最大维度数。
- 各维度的值必须为正整数。
- 支持的数据类型包括：size_t、int等整数类型。

## 调用示例

```cpp
// 创建一个3维张量的Shape
auto shape = AscendC::MakeShape(10, 20, 30);

// 获取各维度的大小
auto dim0 = AscendC::Std::get<0>(shape.value); // dim0 = 10
auto dim1 = AscendC::Std::get<1>(shape.value); // dim1 = 20
auto dim2 = AscendC::Std::get<2>(shape.value); // dim2 = 30
```