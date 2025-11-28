# asc_repeat_reduce_min

## AI处理器支持情况

| AI处理器类型            | 是否支持 |
| :-----------------------   | :------: |
| <term>Ascend 910C</term>   |    √    |
| <term>Ascend 910B</term>   |    √    |

## 功能说明

对每个Repeat内所有元素求最小值。

## 函数原型

- 前n个数据计算

```cpp
__aicore__ inline void asc_repeat_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count, order_t order)

__aicore__ inline void asc_repeat_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count, order_t order)
```

- 高维切分计算

```cpp
__aicore__ inline void asc_repeat_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, const asc_repeat_reduce_config& config, order_t order)

__aicore__ inline void asc_repeat_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, const asc_repeat_reduce_config& config, order_t order)
```

- 同步计算

```cpp
__aicore__ inline void asc_repeat_reduce_min_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count, order_t order)

__aicore__ inline void asc_repeat_reduce_min_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count, order_t order)
```

## 参数说明

表1 参数说明

| 参数名 | 输入/输出 | 描述 |
|----|-----|-----|
| dst | 输出 | 目的操作数。 |
| src  | 输入 | 源操作数。|
| count | 输入 | 参与计算的元素个数。 |
| config | 输入 | 在高维切分计算场景下使用的计算配置参数。详细说明请参考[asc_repeat_reduce_config](../struct/asc_repeat_reduce_config.md)|
| order | 输入 | 使用order参数指定dst中index与value的相对位置以及返回结果行为，取值范围如下：<br>VALUE_INDEX：表示value位于低半部，返回结果存储顺序为[value, index]。<br>INDEX_VALUE：表示index位于低半部，返回结果存储顺序为[index, value]。<br>ONLY_VALUE：表示只返回最值，返回结果存储顺序为[value]。<br>ONLY_INDEX：表示只返回最值索引，返回结果存储顺序为[index]。 |

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。


## 调用示例

```cpp
// 假设有128个half类型的数据待处理。
uint64_t offset = 0;                                   // 首先为src申请内存，从0开始。
__ubuf__ half* src = asc_get_phy_buf_addr(offset);    // 获取src的地址，通过__ubuf__关键字指定该地址指向UB内存。
offset += 128 * sizeof(half);                           // 通过offset将dst的起始地址设置在src之后。
__ubuf__ half* dst = asc_get_phy_buf_addr(offset);     // 获取dst的地址，通过__ubuf__关键字指定该地址指向UB内存。
...... // 将源操作数搬运到src。
asc_repeat_reduce_mmin(dst, src, 128, ONLY_VALUE);
...... // 使用dst中的数据进行后续计算或数据搬运操作。
```