# Copy

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

执行数据复制操作，支持不同存储空间之间的数据搬运。

## 函数原型

```cpp
template <typename Tp, const Tp& traits, typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)

template <typename T, typename... Params>
__aicore__ inline void Copy(const CopyAtom<T>& atomCopy, const Params& ...params)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
|--------|-----------|------|
| atomCopy | 输入 | 复制原子操作对象。 |
| params | 输入 | 复制操作的参数，可变参数。 |

## 返回值说明

无

## 约束说明

- atomCopy必须是有效的CopyAtom对象。
- params的数量和类型必须与复制操作的要求匹配。
- 源操作数和目的操作数的内存空间必须支持复制操作。

## 调用示例

```cpp
// 创建CopyAtom
auto copyAtom = AscendC::MakeCopy(arg1, arg2, arg3);

// 执行复制操作
AscendC::Copy(copyAtom, dst, src);
```