# asc_half2float

## AI处理器支持情况

| AI处理器类型     | 是否支持 |
| ----------- | ---- |
| Ascend 910C | √    |
| Ascend 910B | √    |

## 功能说明

将Half类型转化为Float类型，并支持RINT舍入模式。

- RINT舍入模式：四舍六入五成双舍入。

## 函数原型

* 前n个数据计算
  
  ```cpp
  //在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_half2float(__ubuf__ float* dst, __ubuf__ half* src, uint32_t count)
  ```
- 高维切分计算
  
  ```cpp
  //在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_half2float(__ubuf__ float* dst, __ubuf__ half* src, const asc_unary_config& config)
  ```

- 同步计算
  
  ```cpp
  //在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_half2float_sync(__ubuf__ float* dst, __ubuf__ half* src, uint32_t count)
  ```

## 参数说明

表1 参数说明

| 参数名    | 输入/输出 | 描述                |
| ------ | ----- | ----------------- |
| dst    | 输出    | 目的操作数。            |
| src    | 输入    | 源操作数。             |
| count  | 输入    | 参与连续计算的元素个数。      |
| config | 输入    | 在高维切分计算场景下使用的计算配置参数。<br/>详细说明请参考[asc_unary_config](../struct/asc_unary_config.md)。 |

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
//total_length 指参与计算的数据长度
uint64_t offset = 0;
__ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(0);
offset += total_length* sizeof(half);
__ubuf__ float* dst = (__ubuf__ float*)asc_get_phy_buf_addr(offset);
asc_half2float(dst, src, total_length);
```
