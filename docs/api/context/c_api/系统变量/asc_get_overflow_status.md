# asc_get_overflow_status

## AI处理器支持情况

| AI处理器类型 | 是否支持 |
| :-----------| :------: |
| Ascend 910C |    √    |
| Ascend 910B |    √    |

## 功能说明

执行计算类的操作（例如asc_add）后，获取溢出状态

## 函数原型

```cpp
__aicore__ inline uint64_t asc_get_overflow_status()
```

## 参数说明

无

## 返回值说明

溢出状态。

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

无