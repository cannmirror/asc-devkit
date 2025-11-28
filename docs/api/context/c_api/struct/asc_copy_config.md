# asc_copy_config

asc_copy_config为数据搬运操作的控制参数。

## 结构体具体定义

```cpp
union asc_copy_config {
    uint64_t config;
    struct {
        uint64_t sid: 4;
        uint64_t n_burst: 12;
        uint64_t burst_len: 16;
        uint64_t src_gap: 16;
        uint64_t dst_gap: 16;
    };
};
```

## 字段详解

|字段名|字段含义|
|----------|----------|
| sid | 保留，未使用。填0即可。 |
| n_burst | 待搬运的连续传输数据块个数。取值范围：[1, 4095]。 |
| burst_len | 待搬运的每个连续传输数据块的长度，单位为DataBlock（32字节）。取值范围：[1, 65535]。 |
| src_gap | 源操作数相邻连续数据块的间隔（前面一个数据块的尾鱼后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。取值范围：[1, 65535]。 |
| dst_gap | 目的操作数相邻连续数据块的间隔（前面一个数据块的尾鱼后面一个数据块的头的间隔）。<br>单位为DataBlock（32字节）。取值范围：[1, 65535]。 |