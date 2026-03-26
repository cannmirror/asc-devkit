# Layout

## 功能说明

Layout用于定义张量的布局，包含Shape和Stride信息，描述张量在内存中的组织方式。

## 结构体定义

```cpp
template <typename ShapeType, typename StrideType>
struct Layout {
    ShapeType shape;
    StrideType stride;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| shape | ShapeType | 张量的形状。 |
| stride | StrideType | 张量的步长。 |

## 约束说明

- Shape和Stride的维度数量必须一致。
- Layout支持最多4个维度的数据配置。
- Shape、Stride等数值数据，仅支持size_t类型和int类型。

## 调用示例

```cpp
// 创建Layout
auto shape = AscendC::MakeShape(10, 20, 30);
auto stride = AscendC::MakeStride(1, 100, 200);
auto layout = AscendC::MakeLayout(shape, stride);

// 获取Shape和Stride
auto layoutShape = layout.shape;
auto layoutStride = layout.stride;
```