# asc_get_vms4_sr

## AI处理器支持情况

|AI处理器类型|是否支持|
| :------------ | :------------: |
| <term>Ascend 910C</term> | √ |
| <term>Ascend 910B</term> | √ |

## 功能说明

此接口用于获取MrgSort已经处理过的队列里的Region Proposal个数，并依次存储在四个出参中。

## 函数原型

```cpp
__aicore__ inline void asc_get_vms4_sr(uint16_t sortedNum[4])
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|val|输出|规约操作Reduce Max/Min所有迭代中的最大值/最小值|
|index|输入|规约操作Reduce Max/Min所有迭代中的最大值/最小值的索引(存在多个最大值/最小值时，则为最小的索引)|

## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

需通过同步操作确保规约操作执行完成后再调用本接口获取结果。

## 调用示例

```cpp
...                                                         // 初始化dst、 src、config参数
asc_datablock_reduce_sum(dst, src, config)                  // 规约操作
asc_sync_notify(PIPE_V, PIPE_S, 0);                         // 设置等待和同步信号
asc_sync_wait(PIPE_V, PIPE_S, 0);
half val;
uint32_t index;
int64_t result = asc_get_reduce_max_min_cnt(val, index);    // 获取结果
```