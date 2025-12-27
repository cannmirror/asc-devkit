
# 5.17 asc_float2float

## 产品支持情况

| 产品         | 是否支持 |
| :-----------------------| :-----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √ |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √ |

### 功能说明

对float类型数据进行精度转换处理, 支持多种舍入模式：

- Rint舍入模式, 四舍六入五成双舍入
- Floor舍入模式, 向负无穷舍入
- Ceil舍入模式, 向正无穷舍入
- Round舍入模式, 四舍五入舍入
- Trunc舍入模式, 向零舍入

### 函数原型

- 前n个数据计算

    ```cpp
    // Rint模式
    __aicore__ inline void asc_float2float_r(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    // Floor模式
    __aicore__ inline void asc_float2float_f(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    // Ceil模式
    __aicore__ inline void asc_float2float_c(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    // Round模式
    __aicore__ inline void asc_float2float_a(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    // Trunc模式
    __aicore__ inline void asc_float2float_z(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    ```

- 高维切分计算

    ```cpp
    // Rint模式
    __aicore__ inline void asc_float2float_r(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config)
    // Floor模式
    __aicore__ inline void asc_float2float_f(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config)
    // Ceil模式
    __aicore__ inline void asc_float2float_c(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config)
    // Round模式
    __aicore__ inline void asc_float2float_a(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config)
    // Trunc模式
    __aicore__ inline void asc_float2float_z(__ubuf__ float* dst, __ubuf__ float* src, const asc_unary_config& config)
    ```

- 同步模式

    ```cpp
    // Rint模式
    __aicore__ inline void asc_float2float_r_sync(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    // Floor模式
    __aicore__ inline void asc_float2float_f_sync(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    // Ceil模式
    __aicore__ inline void asc_float2float_c_sync(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    // Round模式
    __aicore__ inline void asc_float2float_a_sync(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    // Trunc模式
    __aicore__ inline void asc_float2float_z_sync(__ubuf__ float* dst, __ubuf__ float* src, const uint32_t count)
    ```

### 参数说明

| 参数名  | 输入/输出 | 描述 |
| :----- | :------- | :------- |
| dst | 输出 | 目的操作数地址。 |
| src | 输入 | 源操作数地址。 |
| count | 输入 | 参与计算的元素个数。 |
| config | 输入 | 在高维切分计算场景下使用的计算配置参数, 详细说明请参考[asc_unary_config](../struct/asc_unary_config.md)。 |

### 返回值说明

无

### 流水类型

PIPE_TYPE_V

### 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。

### 调用示例

```cpp
// total_length指参与计算的数据长度
uint64_t offset = 0;
__ubuf__ float* src0 = (__ubuf__ float*)asc_get_phy_buf_addr(0);
offset += totalLength * sizeof(float);
__ubuf__ float* src1 = (__ubuf__ float*)asc_get_phy_buf_addr(offset);
asc_float2float_r(dst, src, total_length);
```