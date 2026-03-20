# CopyAtom

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

CopyAtom用于定义搬运原子操作，封装了搬运操作的所有信息。

## 结构体定义

```cpp
template <typename... Args>
struct CopyAtom {
    std::tuple<Args...> value;
};
```

## 字段说明

| 字段名 | 类型 | 描述 |
|--------|------|------|
| value | std::tuple<Args...> | 存储复制操作参数的元组。 |

## 约束说明

- Args的数量和类型必须与复制操作的要求匹配。
- CopyAtom通常与Copy函数配合使用。

## 调用示例

```cpp
// 创建CopyAtom
auto copyAtom = AscendC::MakeCopy(arg1, arg2, arg3);

// 执行复制操作
AscendC::Copy(copyAtom, dst, src);
```