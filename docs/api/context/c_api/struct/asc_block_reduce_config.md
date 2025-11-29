# asc_block_reduce_config

asc_block_reduce_config是block reduce类操作中控制操作数地址步长的数据结构。具体使用方法可参考[如何使用高维切分计算API](../general_instruction.md#如何使用高维切分计算api)。

## 结构体具体定义

```cpp
constexpr uint64_t CAPI_DEFAULT_REDUCE_CONFIG_VALUE = 0x0100000800010001;
union asc_block_reduce_config {
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
| dst_repeat_stride | 目的操作数相邻迭代间相同DataBlock的地址步长。<br>输入类型位宽为16bit时，单位为16Byte，输入类型位宽为32bit时，单位为32Byte。默认值：1。 |
| src_block_stride | 源操作数单次迭代内不同DataBlock间地址步长，默认值：1。 |
| src_repeat_stride | 源操作数相邻迭代间相同DataBlock的地址步长，默认值：8。 |
| repeat | 迭代次数，默认值：1。 |