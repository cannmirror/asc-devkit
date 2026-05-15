# thread_block_tile构造函数

`thread_block_tile`不提供构造函数，用户通过接口创建对象。

## 函数原型

```C++
template <unsigned int Size, typename ParentT>
thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)
```

## 参数说明

**表 1**  模板版本参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| g | 输入 | 被划分的父组，类型只能是`thread_block`或`thread_block_tile`。 |
| Size | 输入 | 模板参数，指定划分出的`thread_block_tile`组大小。 |

## 返回值说明

返回`thread_block_tile`对象。

## 约束说明

- `Size`必须是$2^n$，并且必须小于等于32（warpSize），当前可选值范围（1、2、4、8、16、32）。

## 调用示例

- SIMT编程场景：

    ```c++
    using namespace cooperative_groups;
    __global__ void simt_kernel(...)
    {
        ...
        thread_block block = this_thread_block();
        thread_block_tile<32> tile32 = tiled_partition<32>(block);
        auto tile32 = tiled_partition<32>(block);
        thread_block_tile<4, thread_block> tile4 = tiled_partition<4>(block);
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
        thread_block_tile<32> tile32 = tiled_partition<32>(block);
        auto tile32 = tiled_partition<32>(block);
        thread_block_tile<4, thread_block> tile4 = tiled_partition<4>(block);
        ...
    }
    ```
