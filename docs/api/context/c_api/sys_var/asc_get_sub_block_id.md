# asc_get_sub_block_id

## AI处理器支持情况

| AI处理器类型 | 是否支持 |
| :-----------| :------: |
| Ascend 910C |    √    |
| Ascend 910B |    √    |

## 功能说明

获取AI Core上Vector核的ID。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_sub_block_id()
```

## 参数说明

无

## 返回值说明

返回Vector核ID。

## 流水类型

PIPE_TYPE_S

## 约束说明

无

## 调用示例

无