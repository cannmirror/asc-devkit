# asc_binary_config

asc_binary_config为控制操作数地址步长的数据结构。具体使用方法可参考[如何使用Tensor高维切分计算API](../通用说明和约束.md#如何使用tensor高维切分计算api)。

结构体具体定义为：

```cpp
constexpr uint64_t CAPI_DEFAULT_BINARY_CONFIG_VALUE = 0x0100080808010101;
union asc_binary_config {
    uint64_t config = CAPI_DEFAULT_BINARY_CONFIG_VALUE;
    struct {
        uint64_t dst_block_stride: 8;
        uint64_t src0_block_stride: 8;
        uint64_t src1_block_stride: 8;
        uint64_t dst_repeat_stride: 8;
        uint64_t src0_repeat_stride: 8;
        uint64_t src1_repeat_stride: 8;
        uint64_t reserved: 8;
        uint64_t repeat: 8;
    };
};
```

|字段名|字段含义|
|----------|----------|
|dst_block_stride|目的操作数单次迭代内不同datablock间地址步长，默认值：1。|
|src0_block_stride|源操作数0单次迭代内不同datablock间地址步长，默认值：1。|
|src1_block_stride|源操作数1单次迭代内不同datablock间地址步长，默认值：1。|
|dst_repeat_stride|目的操作数相邻迭代间相同datablock的地址步长，默认值：8。|
|src0_repeat_stride|源操作数0相邻迭代间相同datablock的地址步长，默认值：8。|
|src1_repeat_stride|源操作数1相邻迭代间相同datablock的地址步长，默认值：8。|
|repeat|迭代次数，默认值：1。|
