# asc_datablock_reduce_min

## AI处理器支持情况

| AI处理器类型   | 是否支持 |
| ------------|:----:|
| Ascend 910C | √    |
| Ascend 910B | √    |

## 功能说明

执行数据块内的求最大值规约（Reduce Min）操作。

## 函数原型

- 前n个数据连续计算

 ```c++
 __aicore__ inline void asc_datablock_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
 __aicore__ inline void asc_datablock_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
 ```

- 非连续数据计算

 ```c++
 __aicore__ inline void asc_datablock_reduce_min(__ubuf__ half* dst, __ubuf__ half* src, const asc_reduce_config& config)
 __aicore__ inline void asc_datablock_reduce_min(__ubuf__ float* dst, __ubuf__ float* src, const asc_reduce_config& config)
 ```

- 数据同步计算

 ```c++
 __aicore__ inline void asc_datablock_reduce_min_sync(__ubuf__ half* dst, __ubuf__ half* src, uint32_t count)
 __aicore__ inline void asc_datablock_reduce_min_sync(__ubuf__ float* dst, __ubuf__ float* src, uint32_t count)
 ```

## 参数说明

表1 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| dst     | 输出     | 目的操作数。   |
| src     | 输入     | 源操作数。|
| count   | 输入     | 参与连续计算的元素个数。|
| config  | 输入     | 在非连续场景下使用的计算配置参数。|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

- 操作数地址重叠约束请参考[通用地址重叠约束](../general_instruction.md#通用地址重叠约束)。
- dst、src的起始地址需要32字节对齐。

## 调用示例

```c++
// total_length指参与计算的数据总长度
uint64_t offset = 0;
__ubuf__ half* src = (__ubuf__ half*)asc_get_phy_buf_addr(0);
offset += total_length * sizeof(half);
__ubuf__ half* dst = (__ubuf__ half*)asc_get_phy_buf_addr(offset);
asc_datablock_reduce_min(dst, src, total_length);
```
