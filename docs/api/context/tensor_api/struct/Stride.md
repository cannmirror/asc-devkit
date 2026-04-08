# Stride

## 功能说明

Stride用于定义张量的步长，描述张量在各维度上相邻元素在内存中的间隔，间隔的单位为元素。

## 结构体定义体定义

```cpp
template <typename... Ts>
struct Stride {
    std::tuple<Ts...>... value;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| value | std::tuple<Ts...> | 存储各维度步长的元组。 |

## 约束说明

- Stride的维度数量必须与对应的Shape维度数量一致。
- 各维度的步长值必须为正整数。
- 支持的数据类型包括：`size_t`、int等整数类型。

## 调用示例

```cpp
// 创建一个3维张量的Stride
auto stride = AscendC::MakeStride(1, 100, 200);

// 获取各维度的步长
auto stride0 = AscendC::Std::get<0>(stride.value); // stride0 = 1
auto stride1 = AscendC::Std::get<1>(stride.value); // stride1 = 100
auto stride2 = AscendC::Std::get<2>(stride.value); // stride2 = 200
```