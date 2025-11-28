# asc_float2half

## AI处理器支持情况

| AI处理器类型 | 是否支持 |
| :-----------| :-----:|
| Ascend 910C |   √   |
| Ascend 910B |   √   |

## 功能说明

将float类型数据转换为half类型，支持多种舍入模式：

- RINT舍入模式：四舍六入五成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入
- ODD舍入模式：Von Neumann rounding，最近邻奇数舍入

## 函数原型

- 前n个数据计算

  ```cpp
  // 在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_float2half(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // RINT舍入模式
  __aicore__ inline void asc_float2half_r(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // ROUND舍入模式
  __aicore__ inline void asc_float2half_a(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // FLOOR舍入模式
  __aicore__ inline void asc_float2half_f(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // CEIL舍入模式
  __aicore__ inline void asc_float2half_c(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // TRUNC舍入模式
  __aicore__ inline void asc_float2half_z(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  
  // ODD舍入模式
  __aicore__ inline void asc_float2half_o(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  ```

- 高维切分计算

  ```cpp
  // 在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_float2half(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)

  // RINT舍入模式
  __aicore__ inline void asc_float2half_r(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)

  // ROUND舍入模式
  __aicore__ inline void asc_float2half_a(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)

  // FLOOR舍入模式
  __aicore__ inline void asc_float2half_f(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)

  // CEIL舍入模式
  __aicore__ inline void asc_float2half_c(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)

  // TRUNC舍入模式
  __aicore__ inline void asc_float2half_z(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)
  
  // ODD舍入模式
  __aicore__ inline void asc_float2half_o(__ubuf__ half* dst, __ubuf__ float* src, const asc_unary_config& config)
  ```

- 同步计算

    ```cpp
  // 在转换有精度损失时表示RINT舍入模式，不涉及精度损失时表示不舍入
  __aicore__ inline void asc_float2half_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // RINT舍入模式
  __aicore__ inline void asc_float2half_r_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // ROUND舍入模式
  __aicore__ inline void asc_float2half_a_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // FLOOR舍入模式
  __aicore__ inline void asc_float2half_f_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // CEIL舍入模式
  __aicore__ inline void asc_float2half_c_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)

  // TRUNC舍入模式
  __aicore__ inline void asc_float2half_z_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  
  // ODD舍入模式
  __aicore__ inline void asc_float2half_o_sync(__ubuf__ half* dst, __ubuf__ float* src, uint32_t count)
  ```

## 参数说明

表1 参数说明

| 参数名 | 输入/输出 | 描述 |
| :----| :-----| :-----|
| dst | 输出 | 目的操作数。 |
| src  | 输入 | 源操作数。|
| count | 输入 | 参与计算的元素个数。 |
| config | 输入 | 在非连续场景下使用的计算配置参数，请参考[asc_binary_config](../数据结构/asc_binary_config.md)。|

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

## 调用示例

```cpp
// total_length指参与计算的数据长度
uint64_t offset = 0;
__ubuf__ float* src = (__ubuf__ float*)asc_get_phy_buf_addr(0);
offset += total_length * sizeof(float);
__ubuf__ half* dst = (__ubuf__ half*)asc_get_phy_buf_addr(offset);
asc_float2half(dst, src, total_length);
```
