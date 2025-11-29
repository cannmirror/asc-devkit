# asc_data_cache_clean_and_invalid

## 昇腾AI处理器支持情况

|昇腾AI处理器类型   | 是否支持 |
| ------------|:----:|
| Ascend 910C | √    |
| Ascend 910B | √    |

## 功能说明

刷新并无效化数据缓存（Cache Flush），确保Cache一致性，通常在数据搬运前后使用。

## 函数原型

- 前n个数据连续计算

```c++
__aicore__ inline void asc_data_cache_clean_and_invalid(__gm__ void* dst, uint64_t entire)
```

## 参数说明

表1 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| dst     | 输入     | 目的地址指针。   |
| entire     | 输入     | 操作范围或模式标识。|

## 返回值说明

无

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

```c++
//dst为外部输入的待刷新的目标地址。
asc_data_cache_clean_and_invalid(dst, 0);
```
