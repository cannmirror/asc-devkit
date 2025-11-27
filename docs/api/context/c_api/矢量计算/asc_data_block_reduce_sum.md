# asc_data_block_reduce_sum

## AI处理器支持情况

| AI处理器类型 | 是否支持 |
| :-----------| :------: |
| Ascend 910C |    √    |
| Ascend 910B |    √    |

## 功能说明

对每个datablock内所有元素求和。源操作数相加采用二叉树的方式，两两相加。以128个half类型数据求和为例，每个datablock可以计算16个half类型数据，分成8个datablock计算；每个datablock内，通过二叉树的方式，两两相加。

需要注意的是，两两相加计算过程中，计算结果大于65504时结果保存为65504。例如，源操作数为[60000,60000,-30000,100]，首先60000+60000溢出，结果为65504，然后计算-30000+100=-29900，最后计算65504-29900=35604。

## 函数原型

- 前n个数据计算

```cpp
__aicore__ inline void asc_datablock_reduce_sum(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)

__aicore__ inline void asc_datablock_reduce_sum(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
```

- 高维切分计算

```cpp
__aicore__ inline void asc_datablock_reduce_sum(__ubuf__ half* dst, __ubuf__ half* src, const asc_reduce_config& config)

__aicore__ inline void asc_datablock_reduce_sum(__ubuf__ float* dst, __ubuf__ float* src, const asc_reduce_config& config)
```

- 同步计算

```cpp
__aicore__ inline void asc_datablock_reduce_sum_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)

__aicore__ inline void asc_datablock_reduce_sum_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
```

## 参数说明

表1 参数说明

| 参数名 | 输入/输出 | 描述 |
| :----| :-----| :-----|
| dst | 输出 | 目的操作数。 |
| src  | 输入 | 源操作数。|
| count | 输入 | 参与计算的元素个数。 |
| config | 输入 | 在非连续场景下使用的计算配置参数。|

## 返回值说明

无

## 流水类型

PIPE_TYPE_V

## 约束说明

- 源操作数地址对齐要求请参考[通用地址对齐约束](docs\api\context\c_api\通用说明和约束.md#通用地址对齐约束)。
- 操作数地址重叠约束请参考[通用地址重叠约束](docs\api\context\c_api\通用说明和约束.md#通用地址重叠约束)。


## 调用示例

```cpp
// 假设src操作数包含128个half类型的数据，dst操作数均包含8个half类型的数据。
uint64_t offset = 0;                                   // 首先为src申请内存，从0开始。
__ubuf__ half* src = asc_get_phy_buf_addr(offset);    // 获取src的地址，通过__ubuf__关键字指定该地址指向UB内存。
offset += 128 * sizeof(half);                           // 通过offset将dst的起始地址设置在src之后。
__ubuf__ half* dst = asc_get_phy_buf_addr(offset);     // 获取dst的地址，通过__ubuf__关键字指定该地址指向UB内存。
...... // 将源操作数搬运到src0、src1.
asc_data_block_reduce_sum(dst, src, 128);
...... // 使用dst中的数据进行后续计算或数据搬运操作。
```