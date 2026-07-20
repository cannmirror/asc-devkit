# GetProgramCounter(ISASI)

## 产品支持情况

<!-- npu="950" id8 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id8 -->
<!-- npu="A3" id9 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id9 -->
<!-- npu="910b" id10 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id10 -->
<!-- npu="310b" id11 -->
- Atlas 200I/500 A2 推理产品：不支持
<!-- end id11 -->
<!-- npu="310p" id12 -->
- Atlas 推理系列产品AI Core：支持
<!-- end id12 -->
<!-- npu="310p" id13 -->
- Atlas 推理系列产品Vector Core：不支持
<!-- end id13 -->
<!-- npu="910" id14 -->
- Atlas 训练系列产品：不支持
<!-- end id14 -->

## 功能说明

头文件路径为：`"basic_api/kernel_operator_sys_var_intf.h"`。

获取程序计数器的值，以追踪当前程序执行的位置。

## 函数原型

```cpp
__aicore__ inline int64_t GetProgramCounter()
```

## 参数说明

无

## 返回值说明

返回当前程序计数器的值。

## 约束说明

无

## 调用示例

```cpp
int64_t pc = AscendC::GetProgramCounter(); // 获取程序计数器的值，保存到变量pc
```
