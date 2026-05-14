# asc\_atomic\_sub

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

对Unified Buffer或Global Memory上的数据与指定数据执行原子减操作，即在这些内存区域的数据中减去指定数据。

## 函数原型

```
inline int32_t asc_atomic_sub(int32_t *address, int32_t val)
```

```
inline uint32_t asc_atomic_sub(uint32_t *address, uint32_t val)
```

```
inline float asc_atomic_sub(float *address, float val)
```

```
inline int64_t asc_atomic_sub(int64_t *address, int64_t val)
```

```
inline uint64_t asc_atomic_sub(uint64_t *address, uint64_t val)
```

```
inline half2 asc_atomic_sub(half2 *address, half2 val)
```

```
inline bfloat16x2_t asc_atomic_sub(bfloat16x2_t *address, bfloat16x2_t val)
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| address | 输出 | Unified Buffer或Global Memory的地址。 |
| val | 输入 | 源操作数。 |

不同数据类型支持的内存范围说明如下：

**表 2**  不同数据类型支持的内存范围

| 参数数据类型 | 支持的内存空间 |
| --- | --- |
| int32_t、uint32_t、float、half2、bfloat16x2_t | Unified Buffer、Global Memory |
| int64_t、uint64_t | Global Memory |

## 返回值说明

Unified Buffer或Global Memory上的初始数据。

## 约束说明

无

## 需要包含的头文件

使用除half2、bfloat16x2\_t类型之外的接口需要包含"simt\_api/device\_atomic\_functions.h"头文件，使用half2类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16x2\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

```
#include "simt_api/device_atomic_functions.h"
```

```
#include "simt_api/asc_fp16.h"
```

```
#include "simt_api/asc_bf16.h"
```

## 调用示例

-   SIMT编程场景：

    ```
    __global__ __launch_bounds__(1024) void KernelAtomicSub(float* dst, float* src)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        asc_atomic_sub(dst + idx, src[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    SIMD与SIMT混合编程场景，需要显式使用地址空间限定符表示地址空间：\_\_gm\_\_表示Global Memory内存空间，\_\_ubuf\_\_表示Unified Buffer内存空间。

    ```
    __simt_vf__ __launch_bounds__(1024) inline void KernelAtomicSub(__gm__ float* dst, __gm__ float* src)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        asc_atomic_sub(dst + idx, src[idx]);
    }
    ```

