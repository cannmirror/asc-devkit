# asc_get_ctrl

## AI处理器支持情况

| AI处理器类型 | 是否支持 |
| :-----------| :------: |
| Ascend 910C |    √    |
| Ascend 910B |    √    |

## 功能说明

读取CTRL寄存器（控制寄存器）特定比特位上的值。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_ctrl()
```

## 参数说明

无

## 返回值说明

CTRL寄存器的值。具体含义请参考[asc_set_ctrl](asc_set_ctrl.md)中的描述

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

无