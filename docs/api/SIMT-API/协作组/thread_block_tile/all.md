# all

## 功能说明

判断协作组内所有活跃线程的输入是否均不为0。
组内活跃线程执行本接口时，对组内所有活跃线程的输入操作数`predicate`进行判断，所有线程的`predicate`均不为0，返回1，否则返回0。组内所有活跃线程返回相同的结果。

## 函数原型

```C++
int all(int predicate) const
```

## 参数说明

**表 1**  参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| predicate | 输入 | 操作数。 |

## 返回值说明

协作组内所有活跃线程的输入均不为0，返回1，否则返回0。

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
        uint32_t result = tile4.all(1);             // 返回1
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
        uint32_t result = tile4.all(1);             // 返回1
        ...
    }
    ```