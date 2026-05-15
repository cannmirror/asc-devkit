# ballot

## 功能说明

判断协作组内每个活跃线程的输入是否为0。

组内活跃线程执行本接口时，对组内所有活跃线程的输入操作数`predicate`进行判断，返回一个32bit的无符号整数，若线程输入的`predicate`不为0，则返回值中与线程rank对应的bit位为1，否则为0。组内所有活跃线程返回相同的结果。

## 函数原型

```C++
unsigned int ballot(int predicate) const
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| predicate | 输入 | 操作数。 |

## 返回值说明

32bit的无符号整数：若组内活跃线程输入的`predicate`不为0，则返回值中与线程rank对应的bit位为1，否则为0。

## 约束说明

无

## 调用示例

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        uint32_t result = tile4.ballot(1);          // 返回0xf
        ...
    }
    ```

- SIMD与SIMT混合编程场景：

    ```c++
    using namespace cooperative_groups;
    __simt_vf__ inline void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        auto tile4 = tiled_partition<4>(block);
        uint32_t result = tile4.ballot(1);          // 返回0xf
        ...
    }
    ```