# asc_repeat_reduce_config

asc_repeat_reduce_config是repeat reduce类操作中控制操作数地址步长的数据结构。具体使用方法可参考[如何使用高维切分计算API](../general_instruction.md#如何使用高维切分计算api)。

## 结构体具体定义

```cpp
constexpr uint64_t CAPI_DEFAULT_REDUCE_CONFIG_VALUE = 0x0100000800010001;
union asc_repeat_reduce_config {
    uint64_t config = CAPI_DEFAULT_REDUCE_CONFIG_VALUE;
    struct {
        uint64_t dst_repeat_stride : 16;
        uint64_t src_block_stride : 16;
        uint64_t src_repeat_stride : 16;
        uint64_t reserved : 8;
        uint64_t repeat : 8;
    };
};
```

## 字段详解

|字段名|字段含义|
|----------|----------|
| dst_repeat_stride | 目的操作数相邻迭代间的地址步长。以一个repeat归约后的长度为单位。例如：<br>对于asc_repeat_reduce_sum API：单位为dst数据类型所占字节长度。<br>对于asc_repeat_reduce_max/asc_repeat_reduce_min API：返回索引和最值时，单位为dst数据类型所占字节长度的两倍；仅返回最值时，单位为dst数据类型所占字节长度；仅返回索引时，单位为uint32_t类型所占字节长度。<br>默认值：1。 |
| src_block_stride | 源操作数单次迭代内不同DataBlock间地址步长，默认值：1。 |
| src_repeat_stride | 源操作数相邻迭代间相同DataBlock的地址步长，默认值：8。 |
| repeat | 迭代次数，默认值：1。 |