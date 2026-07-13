# asc_get_squeeze_status

## 产品支持情况

| 产品 | 是否支持 |
| :-----------| :------: |
| Ascend 950PR/Ascend 950DT | √ |

## 功能说明

头文件路径：`"c_api/sys_var/sys_var.h"`。

读取squeeze操作后保存至AR特殊寄存器的有效数据长度值，用于配合[asc_squeeze_with_status](../reg/compare_and_select/asc_squeeze_with_status.md)和[asc_storeunalign_postupdate](../reg/reg_store/asc_storeunalign_postupdate.md)接口完成不等长数据存储。

## 函数原型

```cpp
__aicore__ inline int64_t asc_get_squeeze_status()
```

## 参数说明

无

## 返回值说明

返回int64_t类型的squeeze有效数据长度值。

## 流水类型

PIPE_S

## 约束说明

无

## 调用示例

无
