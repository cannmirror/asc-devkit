# asc_set_cmp_mask

## AI处理器支持情况

|AI处理器类型|是否支持|
| :------------ | :------------: |
| <term>Ascend 910C</term> | √ |
| <term>Ascend 910B</term> | √ |

## 功能说明

为[asc_select](asc_select.md)操作设置作用于选择的Mask掩码。

## 函数原型

```cpp
__aicore__ inline void asc_set_cmp_mask(__ubuf__ void* sel_mask)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|sel_mask|输出|用于选择的Mask掩码的起始地址|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

需和[asc_select](asc_select.md)操作配合使用。

## 调用示例

```cpp
__ubuf__ void* sel_mask = asc_get_phy_buf_addr(0);
...     //计算sel_mask的值
asc_set_cmp_mask(sel_mask);
...    // 进行Select操作
```
