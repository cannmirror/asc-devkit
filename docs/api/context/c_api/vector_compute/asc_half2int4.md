# asc_half2int4

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :----------------------- | :------: |
| <term>Ascend 910C</term> |    √     |
| <term>Ascend 910B</term> |    √     |

## 功能说明

将half类型数据转换为int4类型，支持多种舍入模式：

- RINT舍入模式：四舍六入五成双舍入
- ROUND舍入模式：四舍五入舍入
- FLOOR舍入模式：向负无穷舍入
- CEIL舍入模式：向正无穷舍入
- TRUNC舍入模式：向零舍入

## 函数原型

- 前n个数据计算

    ```c++
    //在转换有精度损失时表示RINT舍入模式，不涉及精度损失时代表不舍入
    __aicore__ inline void asc_half2int4(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //RINT舍入模式
    __aicore__ inline void asc_half2int4_r(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //FLOOR舍入模式
    __aicore__ inline void asc_half2int4_f(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //ROUND舍入模式
    __aicore__ inline void asc_half2int4_a(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //CEIL舍入模式
    __aicore__ inline void asc_half2int4_c(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //TRUNC舍入模式
    __aicore__ inline void asc_half2int4_z(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)
    ```

- 高维切分计算

    ```c++
    //在转换有精度损失时表示RINT舍入模式，不涉及精度损失时代表不舍入
    __aicore__ inline void asc_half2int4(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config)

    //RINT舍入模式
    __aicore__ inline void asc_half2int4_r(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config)

    //FLOOR舍入模式
    __aicore__ inline void asc_half2int4_f(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config)

    //ROUND舍入模式
    __aicore__ inline void asc_half2int4_a(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config)

    //CEIL舍入模式
    __aicore__ inline void asc_half2int4_c(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config)

    //TRUNC舍入模式
    __aicore__ inline void asc_half2int4_z(__ubuf__ void* dst, __ubuf__ half* src, const asc_unary_config& config)
    ```

- 同步转换

    ```c++
    //在转换有精度损失时表示RINT舍入模式，不涉及精度损失时代表不舍入
    __aicore__ inline void asc_half2int4_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //RINT舍入模式
    __aicore__ inline void asc_half2int4_r_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //FLOOR舍入模式
    __aicore__ inline void asc_half2int4_f_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //ROUND舍入模式
    __aicore__ inline void asc_half2int4_a_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //CEIL舍入模式
    __aicore__ inline void asc_half2int4_c_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)

    //TRUNC舍入模式
    __aicore__ inline void asc_half2int4_z_sync(__ubuf__ void* dst, __ubuf__ half* src, uint32_t count)
    ```

## 参数说明

表1 参数说明
| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数。 |
| src | 输入 | 源操作数。 |
| count | 输入 | 参与计算的元素个数。 |
| config | 输入 | 在高维切分场景下使用的计算配置参数。详细说明请参考[asc_unary_config](../struct/asc_unary_config.md) 。 |

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐。
- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- 当dst为int4b_t时，前n个数据计算接口的count必须为偶数；

## 调用示例

```cpp
//total_length指参与转换的数据总长度
uint64_t offset = 0;
__ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(0);
offset += total_length * sizeof(half);
__ubuf__ int4b_t* dst = (__ubuf__ int4b_t*)asc_get_phy_buf_addr(offset);
asc_half2int4(dst, src, total_length);
```
