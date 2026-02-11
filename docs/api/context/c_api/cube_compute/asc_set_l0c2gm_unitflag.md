# asc_set_l0c2gm_unitflag
## AI处理器支持情况

|AI处理器类型   | 是否支持 |
| ------------|:----:|
| <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term> | √    |
| <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term> | √    |

## 功能说明

数据搬运过程中进行随路量化时，通过调用该接口设置量化流程中的矢量量化参数。

## 函数原型

```c++
__aicore__ inline void asc_set_l0c2gm_unitflag(uint64_t unitflag)
```

## 参数说明

|参数名|输入/输出|描述|
|------------|------------|-----------|
| unitflag |  输入     | unitflag配置项，类型为bool。预留参数，暂未启用，为后续的功能扩展做保留，可保持默认值false即可。|
## 返回值说明

无

## 流水类型

PIPE_S

## 约束说明

## 调用示例

```c++

```