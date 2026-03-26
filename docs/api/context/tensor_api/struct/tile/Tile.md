# Tile

## 功能说明

Tile用于定义张量的分块，用于将大张量分割成多个小块进行并行处理。

## 结构体定义

```cpp
template <typename... Ts>
struct Tile {
    std::tuple<Ts...> value;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| value | std::tuple<Ts...> | 存储各维度分块大小的元组。 |

## 约束说明

- Tile的维度数量必须与对应的Shape维度数量一致。
- 各维度的分块大小必须为正整数。
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