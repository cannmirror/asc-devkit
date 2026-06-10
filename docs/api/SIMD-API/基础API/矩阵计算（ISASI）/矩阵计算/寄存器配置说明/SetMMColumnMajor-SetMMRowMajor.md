# SetMMColumnMajor/SetMMRowMajor

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

控制Mmad/MmadWithSparse优先通过M/N的哪个方向。

## 函数原型

```cpp
__aicore__ inline void SetMMColumnMajor() // 代表CUBE将首先通过M方向，然后通过N方向生成结果
__aicore__ inline void SetMMRowMajor()   // 代表CUBE将首先通过N方向，然后通过M方向产生结果
```

## 参数说明

无

## 返回值说明

无

## 约束说明

无

## 调用示例

```cpp
AscendC::SetMMRowMajor();
```
