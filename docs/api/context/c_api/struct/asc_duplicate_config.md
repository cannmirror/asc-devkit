# asc_duplicate_config

asc_duplicate_config为控制操作数地址步长的数据结构。具体使用方法可参考[如何使用高维切分计算API](../general_instruction.md#如何使用高维切分计算api)。

## 结构体具体定义

```cpp
constexpr uint64_t CAPI_DEFAULT_DUPLICATE_CONFIG_VALUE = 0x0100800800010001;
union asc_duplicate_config {
    uint64_t config = CAPI_DEFAULT_DUPLICATE_CONFIG_VALUE;
    struct {
        uint64_t dst_block_stride: 16;
        uint64_t src_block_stride: 16;
        uint64_t dst_repeat_stride: 12;
        uint64_t src_repeat_stride: 12;
        uint64_t repeat: 8;
    };
};
```

## 字段详解

|字段名|字段含义|
|----------|----------|
| dst_block_stride | 目的操作数单次迭代内不同DataBlock间地址步长，默认值：1。 |
| src_block_stride | 源操作数单次迭代内不同DataBlock间地址步长，默认值：1。<br>用于asc_duplicate API时，该参数固定填1。 |
| dst_repeat_stride | 目的操作数相邻迭代间相同DataBlock的地址步长，默认值：8。 |
| src_repeat_stride | 源操作数0相邻迭代间相同DataBlock的地址步长，默认值：8。<br>用于asc_duplicate API时，该参数固定填0。 |
| repeat | 迭代次数，默认值：1。 |