# asc_bf162float

## 产品支持情况

|产品|是否支持|
| :------------ | :------------: |
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

## 功能说明

将bfloat16_t类型数据转为float，无舍入模式。

## 函数原型

- 前n个数据计算
```cpp
__aicore__ inline void asc_bf162float(__ubuf__ float* dst, __ubuf__ bfloat16_t* src, uint32_t count)
```
- 高维切分计算
```cpp
__aicore__ inline void asc_bf162float(__ubuf__ float* dst, __ubuf__ bfloat16_t* src, const asc_unary_config& config)
```
- 同步计算
```cpp
__aicore__ inline void asc_bf162float_sync(__ubuf__ float* dst, __ubuf__ bfloat16_t* src, uint32_t count)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|dst|输出|目的操作数。|
|src|输入|源操作数。|
|count|输入|参与计算的元素个数。|
|config|输入|在非连续场景下使用的计算配置参数。<br/>详细说明请参考[asc_unary_config](../struct/asc_unary_config.md)。|

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
// total_length指参与计算的数据总长度
uint64_t offset = 0;
__ubuf__ bfloat16_t* src = (__ubuf__ bfloat16_t*)asc_get_phy_buf_addr(offset);
offset += total_length * sizeof(bfloat16_t);
__ubuf__ float* dst = (__ubuf__ float*)asc_get_phy_buf_addr(offset);
asc_bf162float(dst, src, total_length);
```