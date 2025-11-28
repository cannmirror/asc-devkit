# asc_set_mask_norm

## AI处理器支持情况

|AI处理器类型|是否支持|
| :------------ | :------------: |
| <term>Ascend 910C</term> | √ |
| <term>Ascend 910B</term> | √ |

## 功能说明

设置mask模式为Normal模式，该模式为系统默认模式。

## 函数原型

```cpp
__aicore__ inline void asc_set_mask_norm()
```

## 参数说明

无

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

```cpp
asc_set_mask_norm();
asc_set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);
... //计算操作
```