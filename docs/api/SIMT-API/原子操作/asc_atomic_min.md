# asc\_atomic\_min

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

对Unified Buffer或Global Memory数据做原子求最小值操作，即将Unified Buffer或Global Memory的数据与指定数据中的最小值赋值到Unified Buffer或Global Memory地址中。

## 函数原型

```
inline int32_t asc_atomic_min(int32_t *address, int32_t val)
```

```
inline uint32_t asc_atomic_min(uint32_t *address, uint32_t val)
```

```
inline float asc_atomic_min(float *address, float val)
```

```
inline int64_t asc_atomic_min(int64_t *address, int64_t val)
```

```
inline uint64_t asc_atomic_min(uint64_t *address, uint64_t val)
```

```
inline half asc_atomic_min(half *address, half val)
```

```
inline bfloat16_t asc_atomic_min(bfloat16_t *address, bfloat16_t val)
```

```
inline half2 asc_atomic_min(half2 *address, half2 val)
```

```
inline bfloat16x2_t asc_atomic_min(bfloat16x2_t *address, bfloat16x2_t val)
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
| int32_t、uint32_t、float、half、bfloat16_t、half2、bfloat16x2_t | Unified Buffer、Global Memory |
| int64_t、uint64_t | Global Memory |

## 返回值说明

Unified Buffer或Global Memory上的初始数据。

注意，由于底层硬件约束，half和bfloat16\_t类型的返回值不准确，避免直接使用这些类型的返回值。

## 约束说明

无

## 需要包含的头文件

使用除half、half2、bfloat16\_t、bfloat16x2\_t类型之外的接口需要包含"simt\_api/device\_atomic\_functions.h"头文件，使用half和half2类型接口需要包含"simt\_api/asc\_fp16.h"头文件，使用bfloat16\_t和bfloat16x2\_t类型接口需要包含"simt\_api/asc\_bf16.h"头文件。

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
    __global__ __launch_bounds__(1024) void KernelAtomicMin(float* dst, float* src)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        asc_atomic_min(dst + idx, src[idx]);
    }
    ```

-   SIMD与SIMT混合编程场景：

    SIMD与SIMT混合编程场景，需要显式使用地址空间限定符表示地址空间：\_\_gm\_\_表示Global Memory内存空间，\_\_ubuf\_\_表示Unified Buffer内存空间。

    ```
    __simt_vf__ __launch_bounds__(1024) inline void KernelAtomicMin(__gm__ float* dst, __gm__ float* src)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;
        asc_atomic_min(dst + idx, src[idx]);
    }
    ```

