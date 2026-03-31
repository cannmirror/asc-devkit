# MmadAtom

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

MmadAtom用于定义矩阵乘加原子操作，封装了矩阵乘加操作的所有信息。

## 结构体定义

```cpp
template <typename... Args>
struct MmadAtom {
    std::tuple<Args...> value;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| value | std::tuple<Args...> | 存储矩阵矩阵乘加操作参数的元组。 |

## 约束说明

- Args的数量和类型必须与矩阵乘加操作的要求匹配。
- MmadAtom通常与Mad函数配合使用。

## 调用示例

```cpp
// 创建MmadAtom
auto mmadAtom = AscendC::MakeMad(arg1, arg2, arg3);

// 执行矩阵乘加操作
AscendC::Mad(mmadAtom, dst, src0, src1);
```