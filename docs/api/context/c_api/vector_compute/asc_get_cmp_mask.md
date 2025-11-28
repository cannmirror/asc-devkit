# asc_get_cmp_mask

## AI处理器支持情况

|AI处理器类型|是否支持|
| :------------ | :------------: |
| <term>Ascend 910C</term> | √ |
| <term>Ascend 910B</term> | √ |

## 功能说明

此接口用于获取Compare操作的比较结果。

## 函数原型

```cpp
__aicore__ inline void asc_get_cmp_mask(__ubuf__ void* dst)
```

## 参数说明

|参数名|输入/输出|描述|
| ------------ | ------------ | ------------ |
|dst|输出|存放比较操作结果的地址|

## 返回值说明

无

## 流水类型

PIPE_V

## 约束说明

需和Compare操作配合使用。

## 调用示例

```
__ubuf__ void* dst = asc_get_phy_buf_addr(0);
...     // 进行Compare操作
asc_get_cmp_mask(dst);
```