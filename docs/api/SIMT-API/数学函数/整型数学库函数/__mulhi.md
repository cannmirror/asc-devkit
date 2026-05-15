# \_\_mulhi

## 产品支持情况

| 产品 | 是否支持 |
| --- | --- |
| Ascend 950PR/Ascend 950DT | √ |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | x |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | x |
| Atlas 200I/500 A2 推理产品 | x |
| Atlas 推理系列产品AI Core | x |
| Atlas 推理系列产品Vector Core | x |
| Atlas 训练系列产品 | x |

## 功能说明

获取输入int32类型数据x和y乘积的高32位。

![](../../../figures/zh-cn_formulaimage_0000002562010439.png)

## 函数原型

```
int __mulhi(int x, int y)
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| x | 输入 | 源操作数，乘数。 |
| y | 输入 | 源操作数，乘数。 |

## 返回值说明

输入int32类型数据x和y乘积的高32位。

## 约束说明

无

## 需要包含的头文件

使用该接口需要包含"simt\_api/device\_functions.h"头文件。

```
#include "simt_api/device_functions.h"
```

## 调用示例

-   SIMT编程场景：

    ```
    __global__ __launch_bounds__(1024) void KernelMulhi(int* dst, int* x, int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mulhi(x[idx], y[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    ```
    __simt_vf__ __launch_bounds__(1024) inline void KernelMulhi(__gm__ int* dst, __gm__ int* x, __gm__ int* y)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        dst[idx] = __mulhi(x[idx], y[idx]);
    }
    ```

