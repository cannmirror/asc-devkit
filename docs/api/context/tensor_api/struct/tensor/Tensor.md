# Tensor

## 功能说明

Tensor用于表示张量数据结构，包含数据指针和布局信息，支持多维数据的存储和访问。

## 结构体定义

```cpp
template <typename Trait>
struct Tensor {
    void* data;
    Trait trait;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| data | void* | 指向数据的指针。 |
| trait | Trait | 张量的特性信息，包含布局等。 |

## 约束说明

- data指针必须指向有效的内存空间。
- trait中的Layout必须正确描述数据的形状和步长。
- Tensor支持LocalTensor和GlobalTensor两种类型。

## 调用示例

```cpp
// 创建Tensor
auto layout = AscendC::MakeLayout(shape, stride);
auto trait = AscendC::MakeTensorTrait<half, TPosition::UB>(layout);

using traitType = decltype(trait);
LocalTensor<traitType> tensor;
tensor.SetTensorTrait(trait);
```