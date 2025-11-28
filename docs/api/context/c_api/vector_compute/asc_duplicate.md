# asc_duplicate

## AI处理器支持情况

| AI处理器类型 | 是否支持  |
| :-----------| :------: |
| Ascend 910C |    √     |
| Ascend 910B |    √     |

## 功能说明

执行矢量复制（Duplicate）操作，将标量值复制填充到矢量中。

## 函数原型

- 前n个数据计算

    ```cpp
    __aicore__ inline void asc_duplicate(__ubuf__ half* dst, half src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ int16_t* dst, int16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ uint16_t* dst, uint16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ float* dst, float src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ int32_t* dst, int32_t src, uint32_t count)
    __aicore__ inline void asc_duplicate(__ubuf__ uint32_t* dst, uint32_t src, uint32_t count)
    ```

- 高维切分计算

    ```cpp
    __aicore__ inline void asc_duplicate(__ubuf__ half* dst, half src, const asc_duplicate_config& config)
    __aicore__ inline void asc_duplicate(__ubuf__ int16_t* dst, int16_t src, const asc_duplicate_config& config)
    __aicore__ inline void asc_duplicate(__ubuf__ uint16_t* dst, uint16_t src, const asc_duplicate_config& config)
    __aicore__ inline void asc_duplicate(__ubuf__ bfloat16_t* dst, bfloat16_t src, const asc_duplicate_config& config)
    __aicore__ inline void asc_duplicate(__ubuf__ float* dst, float src, const asc_duplicate_config& config)
    __aicore__ inline void asc_duplicate(__ubuf__ int32_t* dst, int32_t src, const asc_duplicate_config& config)
    __aicore__ inline void asc_duplicate(__ubuf__ uint32_t* dst, uint32_t src, const asc_duplicate_config& config)
    ```

- 同步计算

    ```cpp
    __aicore__ inline void asc_duplicate_sync(__ubuf__ half* dst, half src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ int16_t* dst, int16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ uint16_t* dst, uint16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ bfloat16_t* dst, bfloat16_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ float* dst, float src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ int32_t* dst, int32_t src, uint32_t count)
    __aicore__ inline void asc_duplicate_sync(__ubuf__ uint32_t* dst, uint32_t src, uint32_t count)
    ```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| dst | 输出 | 目的操作数地址 |
| src | 输入 | 源标量值 |
| count | 输入 | 参与连续复制的元素个数 |
| config | 输入 | 在非连续场景下使用的复制配置参数 |

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- dst、src的起始地址需要32字节对齐。

## 调用示例

```cpp
__ubuf__ half* dst = (__ubuf__ half*)asc_GetPhyBufAddr(0);
half val(18.0);
asc_duplicate(dst, val, 128);
```
