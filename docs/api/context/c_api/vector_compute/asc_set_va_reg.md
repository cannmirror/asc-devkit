# asc_set_va_reg

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

用于设置transpose的地址，将操作数地址序列与地址寄存器关联。

## 函数原型

```cpp
__aicore__ inline void asc_set_va_reg(ub_addr8_t addr, uint64_t* src_array)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|addr|输出|地址寄存器，类型为ub_addr8_t，可取值为：<br>&bull; VA0<br>&bull; VA1<br>&bull; VA2<br>&bull; VA3<br>&bull; VA4<br>&bull; VA5<br>&bull; VA6<br>&bull; VA7<br>数字代表寄存器顺序，使用方法请参考调用示例|
|src_array|输入|操作数地址序列。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址对齐约束请参考[通用地址对齐约束](../general_instruction.md#通用地址对齐约束)。

## 调用示例

```cpp
__ubuf__ half dst_list[16];
__ubuf__ half src_list[16];
const int32_t VA_REG_ARRAY_LEN = 8;

asc_set_va_reg(VA0, dst_list);
asc_set_va_reg(VA1, dst_list + VA_REG_ARRAY_LEN);
asc_set_va_reg(VA2, src_list);
asc_set_va_reg(VA3, src_list + VA_REG_ARRAY_LEN);
```