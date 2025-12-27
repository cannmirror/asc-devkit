# asc_brcb

## 产品支持情况

| 产品 | 是否支持  |
| :-----------| :------: |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 |    √     |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 |    √     |

## 功能说明

给定一个输入张量，每一次取输入张量中的8个数填充到结果张量的8个datablock（32Bytes）中去，每个数对应一个datablock。

## 函数原型

```cpp
__aicore__ inline void asc_brcb(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, const asc_brcb_config& config)
__aicore__ inline void asc_brcb(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, const asc_brcb_config& config)

__aicore__ inline void asc_brcb_sync(__ubuf__ uint16_t* dst, __ubuf__ uint16_t* src, const asc_brcb_config& config)
__aicore__ inline void asc_brcb_sync(__ubuf__ uint32_t* dst, __ubuf__ uint32_t* src, const asc_brcb_config& config)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数地址 |
| src | 输入 | 源操作数地址 |
| config | 输入 | 在高维切分计算场景下使用的计算配置参数。详细说明请参考[asc_brcb_config](../struct/asc_brcb_config.md) |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 不支持src与dst为同一块内存地址。

## 调用示例

```cpp
//total_length 指参与计算的数据长度
uint64_t offset = 0;
__ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(0);
offset += total_length * sizeof(half);
__ubuf__ half* dst= (__ubuf__ half*)asc_get_phy_buf_addr(offset);
asc_brcb(dst, src, config);
```
