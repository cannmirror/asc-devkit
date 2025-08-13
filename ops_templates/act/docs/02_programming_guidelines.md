# 编程指南

## 概述
本文档主要介绍使用ACT开发BasicMatmul算子的实现细节，通过该算子示例帮助开发者快速掌握矩阵乘法的基本实现方法。ACT的分层包括Tile层、Block层、Kernel层、Device层，详细的分层介绍请参考[ACT分层结构](./04_design.md)。本文档中的算子示例使用ACT提供的下层基础组件实现Device层和Kernel层的组装，并完成算子调用。完整的BasicMatmul样例执行方式请参考[快速入门](./01_quickstart.md)，样例源码为[`examples/00_basic_matmul/main.cpp`](../examples/00_basic_matmul/main.cpp)。

掌握矩阵乘法的基本实现方法后，您可以根据实际需求，对样例中的代码进行扩展或修改，以适应更复杂的计算任务。例如，增加更多的矩阵操作、优化数据传输和计算逻辑等。

## Kernel层组装

Kernel层模板由ProblemShape和Block层组件构成，定义如下：

```cpp
KernelMatmul<class ProblemShape, class BlockMmadBuilder, class BlockEpilogue, class BlockScheduler>
```

- ProblemShape
  ProblemShape定义Matmul计算的Shape的数据类型，定义如下：

   ```cpp
   using ProblemShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
   ```
- BlockMmadBuilder
  BlockMmadBuilder为Block层Mmad计算接口，定义方式如下：

   ```cpp
   using L1TileShape = AscendC::Shape<Int<128>, Int<256>, Int<128>>; // L1 base block shape
   using L0TileShape = AscendC::Shape<Int<128>, Int<256>, Int<64>>;  // L0 base block shape

   // Define matrix date type and layout
   using AType = half;
   using BType = half;
   using CType = float;
   using BiasType = float;

   // Define matrix layout, using row-major as an example here
   using LayoutA = layout::RowMajor;
   using LayoutB = layout::RowMajor;
   using LayoutC = layout::RowMajor;
   using LayoutBias = layout::RowMajor;

   // Define scheduler
   using BlockScheduler = IterateKScheduler;

   // Define dispatch policy
   using DispathPolicy = MatmulMultiBlockWithLayout<>;

   // Define BlockMmad
   using BlockMmad = Block::BlockMmadBuilder<AType, LayoutA, BType, LayoutB, CType, LayoutC, BiasType, LayoutBias,
                                           L1TileShape, L0TileShape, BlockScheduler, DispathPolicy>;
   ```

   矩阵的数据排布格式主要有两种：

  - 行优先（RowMajor）：矩阵的每一行数据在内存中是连续存储的。

  - 列优先（ColMajor）：矩阵的每一列数据在内存中是连续存储的。

   该示例支持A、B矩阵以行优先（RowMajor）或列优先（ColMajor）的数据排布方式作为输入。

- BlockEpilogue
  BlockEpilogue为Block层的后处理，本示例构建基础Matmul，无后处理，所以传入空类即可。

   ```cpp
   // The first parameter is the post-processing move-out type, the second parameter is the post-processing move-in type
   using BlockEpilogue = Block::BlockEpilogueEmpty<CType, CType>;
   ```

- BlockScheduler
BlockScheduler模板类定义矩阵计算中循环处理数据时的方向顺序，提供计算地址偏移的方法。本示例使用基础的[`IterateKScheduler`](../include/matmul/block/block_scheduler_iterateK.h)。

   ```cpp
   using BlockScheduler = IterateKScheduler;
   ```

基于上述组件完成对BasicMatmul示例的Kernel层组装，如下代码所示。

```cpp
using MatmulKernel = Kernel::KernelMatmul<ProblemShape, BlockMmad, BlockEpilogue, BlockScheduler>;
```

## Device层调用

基于组装的Kernel层MatmulKernel，完成组装参数和调用Device层DeviceMatmul的函数编写。

MatmulKernel的调用接口为`()`运算符重载函数，该函数的入参为`MatmulKernel::Arguments`，通过Arguments传入m, n, k, batch的数值、各输入输出的地址。使用ACT提供的Device层实现，定义Device层对象DeviceMatmul，调用该类的InitParams函数进行参数初始化，通过DeviceMatmul的`()`运算符重载函数，调用并执行核函数KernelFunc。

```cpp
void MatmulOp(uint8_t* x1, uint8_t* x2, uint8_t* y, uint8_t* bias, int64_t m, int64_t n, int64_t k,
    void* stream = nullptr)
{
    MatmulShape shape {m, n, k, batch};
    MatmulKernel::Arguments args = {
        shape,              // problem shape
        {x1, x2, y, bias},  // mmad args
        {}                  // epilogue args
    };
    // Instantiate matmul with specfied kernel
    DeviceMatmul mm;
    // Initialize kernel with arguments and workspace pointer
    mm.InitParams(args, workspaceDevice);

    // Launch kernel
    mm();

    ...
}
```

ACT提供的Device层DeviceMatmul类实现的部分代码如下，DeviceMatmul的`()`运算符重载函数调用核函数KernelFunc，实例化一个Kernel层对象，并执行该算子。开发者可直接使用DeviceMatmul类，具体代码详见[device_matmul.h](../include/matmul/device/device_matmul.h)。

```cpp
tempalte <class MatmulKernel>
__global__ __aicore__ void KernelFunc(typename MatmulKernel::Params params)
{
    MatmulKernel op;
    op(kernelArgs);
}

template <class MatmulKernel>
class DeviceMatmul {
public:
    typename MatmulKernel::Params params_{};
    ...
    void InitParams(typename MatmulKernel::Arguments& args, GM_ADDR workspace)
    {
        params_ = MatmulKernel::InitParams(args, workspace);
    }
    void operator()(bool isQuant = false, void* stream = nullptr)
    {
        int64_t blockNum = MatmulKernel::GetBlockNum(params_.problemShape);
        KernelFunc<MatmulKernel><<<blockNum, nullptr, stream>>>(params_);
    }
}
```

## 算子调用

在Kernel层，由开发者指定输入输出的类型和数据排布信息，在ACT提供的Device层，使用内核调用符`<<<>>>`的方式调用核函数，内核调用符的具体介绍请参考[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中的“编程模型 > 核函数”章节。

```cpp
KernelFunc<MatmulKernel><<<blockNum, nullptr, stream>>>(params_);
```

## 算子编译

调用[act_examples_add_executable函数](../examples/CMakeLists.txt)指定`target`名称和编译文件。如下所示，`00_basic_matmul`为`target`名称，`main.cpp`为需要编译的文件。

```cmake
act_examples_add_executable(
    00_basic_matmul
    main.cpp
)
```

在样例根目录下，执行`run.sh`或使用[毕昇编译器](https://hiascend.com/document/redirect/CannCommunityBiSheng)进行算子编译。

### 使用毕昇编译器编译

首先确认当前环境是否支持使用毕昇编译器，执行`bishengcc --help`，如果支持使用，会打印如下提示信息。

```bash
Usage   : bishengcc [options] file...

Options :
=======================================================
```

在样例根目录下执行如下命令，即可进行算子编译。
```bash
bishengcc "main.cpp" -arch <soc_version> -I<act_dir>
```
其中的参数说明如下：

- soc_version：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行`npu-smi info`命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。
- act_dir：头文件搜索路径，需要指定为ACT代码的根目录，即`docs`的上一层目录。

编译成功后，生成一个二进制文件`a.out`，将其重命名为`ascendc_matmul_bbit`。

## 算子执行

在样例根目录下，调用run.sh脚本编译并执行，或者在编译成功后执行如下命令，进行特定Shape的用例执行。

```bash
ascendc_matmul_bbit <m> <n> <k> <batch>
```
算子执行后，若在结果打屏中提示“Success”，说明基于ACT编写的算子已成功执行。


