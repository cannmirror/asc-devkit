# 概览 {ignore=true}
[TOC]
# 自定义高阶API开发指南
## 介绍
高阶API（Advanced API）是基于单核对常见算法的抽象和封装，用于提高编程开发效率。高阶API是通过封装基础API或微指令来实现的，主要包括数学库、Matmul、量化反量化、数据归一化等API，详细API列表请见[Ascend C高阶API列表](./aicore/adv_api/README.md)。
对高阶API仓库的贡献主要存在两种场景，一是开发新的高阶API，二是基于原有高阶API进行开发。完成开发后，用户可以将高阶API编译部署到CANN软件环境中使用。

----
本指南主要以数学库高阶API——`axpy`为例，可以在[这里](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0500.html)查看该具体接口信息。
## 高阶API开发流程
AscendC高阶API开发流程主要包括：
- 设计API
- 开发API
    - 编写API对外接口
    - 内部实现
- 测试
    - UT测试
    - 自定义算子工程
- 提交PR
---
下面将以高阶API——`axpy`为例，讲解如何从零开始，开发一个高阶API。在案例中，我们删除了一些非必要代码，如果对这些代码感兴趣可以在代码仓中进行查看。
## 设计API
### 功能
源操作数`srcTensor`中每个元素与标量求积后和目的操作数`dstTensor`中的对应元素相加，计算公式如下，其中`PAR`表示矢量计算单元一个迭代能够处理的元素个数。
$$dstTensor_i = Axpy(srcTensor_i,scalarValue),i\in[0,PAR]$$
$$dstTensor_i = srcTensor_i \times scalarValue+dstTensor_i$$
参考[AscendC编程范式](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/opdevg/Ascendcopdevg/atlas_ascendc_10_0016.html)的数据流向。Host侧申请GlobalMemory空间，向GlobalMemory写入数据，在Kernel侧从GlobalMemory搬运数据到LocalMemory，计算单元从LocalMemory中获取数据进行计算，将结果写回LocalMemory中，其中的中间结果也保存在LocalMemory中。最后把LocalMemory的运算结果搬运到GlobalMemory。调用高阶API在计算这一步骤中，Kernel侧把拷贝好数据的LocalTensor以及其它参数传递给高阶API进行计算。
了解了背景，我们从公式中可知，`Axpy`需要传递输入参数`srcTensor`和`scalarValue`，输出参数`dstTensor`，还需要一个输入参数`calCount`参数来表示计算量。由于计算中产生了中间结果，所以在LocalMemory中需要额外分配空间进行存储，因此还有一个`shardTmpBuffer`作为输入参数。模板参数中的`isReuseSource`为原代码的预留参数，用户在开发时可以不添加该参数。
除此之外，公式涉及到一个tensor与一个scalar的相乘运算，可以使用AscendC提供的`Muls`函数实现，以及两个tensor的加法运算，可以使用AscendC提供的`Add`函数实现。


### 函数原型
- Kernel侧函数原型
    ```c++
    template <typename T, typename U, bool isReuseSource = false>
    __aicore__ inline void Axpy(const LocalTensor<T>& dstTensor, const LocalTensor<U>& srcTensor, const U scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    ```
    模板参数说明
    | 参数名      | 描述 |
    | ----------- | ----------- |
    | T      |    目的操作数数据类型，支持数据类型为half/float    |
    | U      |    源操作数数据类型，支持数据类型为half/float   |
    | isReuseSource   |   是否允许修改源操作数。该参数预留，传入默认值false即可      |

    接口参数说明
    | 参数名      | 输入/输出 |  描述   |
    | ----------- | ----------- |----------- |
    | dstTensor      | 输出  | 目的操作数数据类型，支持数据类型为half/float    |
    | srcTensor      |  输入  |源操作数数据类型，支持数据类型为half/float   |
    | scalarValue   | 输入  |scalar标量。支持的数据类型为half/float。scalar操作数的类型需要和srcTensor保持一致。      |
    | sharedTmpBuffer   | 输入  | 临时缓存。类型为[LocalTensor](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0006.html),支持的TPosition为VECIN/VECCALC/VECOUT。由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间需要开发者通过sharedTmpBuffer入参传入。临时空间大小BufferSize的获取方式请参考[GetAxpyMaxMinTmpSize](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/ascendcopapi/atlasascendc_api_07_0586.html)。      |
    | calCount   | 输入 | 计算数据量大小，单位字节。     | 
- Tiling侧函数原型
    Kernel侧接口的计算需要开发者预留/申请临时空间，本接口用于在host侧获取预留/申请的最大(`maxValue`)和最小临时空间大小(`minValue`)。期望输入一个源操作数tensor的大小和数据类型，获取整个tensor完成计算需要的临时空间大小，对于tensor的大小我们用[ge::Shape](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/API/basicdataapi/atlasopapi_07_00422.html)类型的数据作为入参。对于数据类型，我们用一个`uint32_t`的常量入参来表示数据类型所占的字节数。同时传递minValue和maxValue两个变量来作为输出参数。参数中的`isReuseSource`为预留参数，与Axpy接口对齐。
    ```c++
    void GetAxpyMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue);
    ```
    接口参数说明
    | 参数名      | 输入/输出 |  描述   |
    | ----------- | ----------- |----------- |
    | srcShape      | 输出  | 输入的shape信息   |
    | typeSize      |  输入  | 算子输入的数据类型大小，单位为字节。比如算子输入的数据类型为half，此处应该传入2   |
    | isReuseSource   | 输入  |  预留参数，可以不关注。是否复用源操作数输入的空间，与Axpy接口一致      |
    | maxValue   | 输入  | Axpy接口能完成计算所需的最大临时空间大小，超出该值的空间不会被接口使用。在最小临时空间-最大临时空间范围内，随着临时空间增大，kernel侧接口计算性能会有一定程度的优化提升。为了达到更好的性能，开发者可以根据实际的内存使用情况进行空间申请，最大空间大小为0表示计算不需要临时空间。      |
    | minValue   | 输入 | Axpy接口能完成计算所需的最小临时空间大小。为保证功能正确，接口计算时申请的临时空间不能小于该数值。最小空间大小为0表示计算不需要临时空间。   | 
## 编写API对外接口
- [include/aicore/adv_api/math/axpy.h](../include/aicore/adv_api/math/axpy.h)。
    参考API设计中函数原型编写。
    ```c++
    template <typename T, typename U, bool isReuseSource = false>
    __aicore__ inline void Axpy(const LocalTensor<T>& dstTensor, const LocalTensor<U>& srcTensor, const U scalarValue,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    {
        AxpyImpl<T, U, isReuseSource>(dstTensor, srcTensor, scalarValue, sharedTmpBuffer, calCount);
    }
    ```
- [include/aicore/adv_api/math/axpy_tiling.h](../include/aicore/adv_api/math/axpy_tiling.h)。
    参考API设计中函数原型编写。
    ```c++
    void GetAxpyMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource, uint32_t& maxValue, uint32_t& minValue);
    ```
- 在[include/aicore/adv_api/kernel_api.h](../include/aicore/adv_api/kernel_api.h)引入Axpy接口文件。
    一般建议引入，之后要调用高阶API时，只需要引入`"kernel_api.h"`。
    ```c++
    #if defined(__CCE_AICORE__) && (__CCE_AICORE__ < 300) && (__NPU_ARCH__ != 5102)
    // ...
    #include "math/axpy.h"
    // ...
    #endif // __CCE_AICORE__ < 300
    ```
- 在[include/aicore/adv_api/tiling_api.h](../include/aicore/adv_api/tiling_api.h)引入Tiling接口文件。
    一般建议引入，之后要调用高阶API的Tiling函数时，只需要引入`"tiling_api.h"`。
    ```c++
    #include "math/axpy_tiling.h"
    ```

## 内部实现
- Kernel侧对应文件为[impl/aicore/adv_api/detail/math/axpy/axpy_common_impl.h](../impl/aicore/adv_api/detail/math/axpy/axpy_common_impl.h)。
引入必要的头文件。
    ```c++
    #include "kernel_tensor.h"
    ```
    如果源操作数tensor元素类型为float，直接调用基础API的Axpy，否则，调用自定义函数，通过Muls和Add组合计算，提供更优的精度。
    ```c++
    template <typename T, typename U, bool isReuseSource>
    __aicore__ inline void AxpyImpl(const LocalTensor<T>& dstTensor, const LocalTensor<U>& srcTensor, const U scalarValue,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    {
        CHECK_FUNC_HIGHLEVEL_API(
            Axpy, (T, U, isReuseSource), (dstTensor, srcTensor, scalarValue, sharedTmpBuffer, calCount));

        if constexpr (sizeof(U) == sizeof(float)) {
            Axpy<T, U>(dstTensor, srcTensor, scalarValue, calCount);
        } else {
            AxpySub<T, U, isReuseSource>(dstTensor, srcTensor, scalarValue, sharedTmpBuffer, calCount);
        }
    }
    ```
    `AxpySub`函数主要是根据临时空间大小制定计算策略，由于传入的sharedTmpBuffer是uint8类型，而源操作数元素是float类型，需要将tmpBuffer的元素类型转化为float。由于硬件结构的限制，一次计算的数据量需要对32B对齐，`axpyTmpCalc`函数就是为了实现对齐，此外如果目标操作数元素类型为half，为了提升精度，在计算过程中需要对源操作数和目标操作数都做一次cast计算，临时空间有一半被目标操作数cast后的结果所占用，所以此时能够计算的数据量为32B对齐后大小的一半。已知总计算量`calCount`，可以得到根据可以使用的tmpBuffer大小得到单次计算量`stackSize`、计算次数`round`。计算公式如下：
   $$stackSize = \begin{cases}
    Align32(tmpbufferSize \div 2) &T = half
    \\ 
    Align32(tmpbufferSize) &T = float
    \end{cases}$$
    $$round = calCount/stackSize$$
    如果存在尾块，还需要多一次计算处理尾块。在计算前，可以通过`setMaskCount`设置计算单元的工作方式为Count，具体可参考[如何使用掩码操作API](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0024.html)。
    ```c++
    template <typename T, typename U, bool isReuseSource = false>
    __aicore__ inline void AxpySub(const LocalTensor<T>& dstTensor, const LocalTensor<U>& srcTensor, const U& scalarValue,
        const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
    {
        uint32_t bufferSize = sharedTmpBuffer.GetSize();
        CheckTmpBufferSize(bufferSize, 0, bufferSize);

        LocalTensor<float> tmpBuffer = sharedTmpBuffer.ReinterpretCast<float>();
        uint32_t tmpBufferSize = tmpBuffer.GetSize();

        uint32_t stackSize = axpyTmpCalc<T>(tmpBufferSize);

        const uint32_t round = calCount / stackSize;
        const uint32_t tail = calCount % stackSize;

        SetMaskCount();
        SetVectorMask<T, MaskMode::COUNTER>(0, stackSize);

        uint32_t offset = 0;
        for (uint32_t i = 0; i < round; i++) {
            AxpyIntrinsicsImpl(dstTensor[offset], srcTensor[offset], scalarValue, tmpBuffer, stackSize);
            offset = offset + stackSize;
        }

        if (tail != 0) {
            SetVectorMask<T, MaskMode::COUNTER>(0, tail);
            AxpyIntrinsicsImpl(dstTensor[offset], srcTensor[offset], scalarValue, tmpBuffer, stackSize);
        }

        SetMaskNorm();
        ResetMask();
    }
    ```
    `AxpyIntrinsicsImpl`函数主要涉及具体的算法，内部包含对基础API的调用。源操作数和目标操作数的元素类型均为half，在上文提到此时stackBuffer临时空间能容纳$2 * stackSize$个字节，源操作数和目的操作数的cast中间结果各占一半。因此`tmpDst`的起始偏移量为`stackSize`。随后就是分别对源操作数和目标操作数做一次half到float的cast计算，然后做一次Muls和一次Add，最后将结果做一次float到half的cast计算。在Count模式下调用基础API时，源操作数和目的操作数以外的参数可以按照示例代码中这么填写，对这些参数感兴趣的话，可以去查看对应的基础API文档。
    ```c++
    template <>
    __aicore__ inline void AxpyIntrinsicsImpl(const LocalTensor<half>& dstTensor, const LocalTensor<half>& srcTensor,
        const half& scalarValue, LocalTensor<float> stackBuffer, uint32_t stackSize)
    {
        LocalTensor<float> tmpSrc = stackBuffer[0];
        LocalTensor<float> tmpDst = stackBuffer[stackSize];

        const UnaryRepeatParams unaryParams;
        const BinaryRepeatParams binaryParams;

        Cast<float, half, false>(tmpSrc, srcTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();

        Cast<float, half, false>(tmpDst, dstTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();

        Muls<float, false>(tmpSrc, tmpSrc, (float)scalarValue, MASK_PLACEHOLDER, 1, unaryParams);
        PipeBarrier<PIPE_V>();

        Add<float, false>(tmpDst, tmpSrc, tmpDst, MASK_PLACEHOLDER, 1, binaryParams);
        PipeBarrier<PIPE_V>();

        Cast<half, float, false>(dstTensor, tmpDst, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
            {1, 1, HALF_DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE});
        PipeBarrier<PIPE_V>();
    }
    ```
- Tiling侧实现代码对应[impl/aicore/adv_api/tiling/math/axpy_tiling_impl.cpp](../impl/aicore/adv_api/tiling/math/axpy_tiling_impl.cpp)。
    引入头文件。
    ```c++
    #include "lib/math/axpy_tiling.h"  // Tiling接口头文件
    #include <cstdint>  // 类型库
    #include "graph/tensor.h" // ge::Shape使用此库
    #include "impl/host_log.h" // 日志库
    ```
    定义了一些常量，在下面会做介绍。
    ```c++
    constexpr uint32_t AXPY_HALF_CALC_PROC = 4;
    constexpr uint32_t AXPY_FLOAT_CALC_PROC = 1;
    constexpr uint32_t AXPY_ONE_REPEAT_BYTE_SIZE = 256;
    ```
    在`GetAxpyMaxMinTmpSize`接口中，调用了`GetAxpyMaxTmpSize`获取所需临时空间最大值，调用了`GetAxpyMinTmpSize`获取所需临时空间最小值。
   
    ```c++
    void GetAxpyMaxMinTmpSize(const ge::Shape& srcShape, const uint32_t typeSize, const bool isReuseSource,
    uint32_t& maxValue, uint32_t& minValue)
    {
        (void)isReuseSource;
        const uint32_t inputSize = srcShape.GetShapeSize();
        ASCENDC_HOST_ASSERT(inputSize > 0, return, "Input Shape size must be greater than 0.");
        if (typeSize == sizeof(float)) {
            minValue = 0;
            maxValue = 0;
            return;
        }
        minValue = GetAxpyMinTmpSize(typeSize);
        maxValue = GetAxpyMaxTmpSize(inputSize, typeSize);
    }
    ```
    对于最大值的计算，当`typeSize`等于`sizeof(float)`时，`Axpy`伪代码可以表示为：
    ```
    Cast(tmpSrc, srcTensor)
    Muls(tmpSrc, dstTensor, scalar)
    Add(dstTensor, tmpSrc, dstTensor)
    ```
    > 使用了一倍的数据大小的临时空间，对应上面`AXPY_FLOAT_CALC_PROC`的值。

    当`typeSize`等于`sizeof(half)`时，`Axpy`伪代码可以表示为：
    ```
    Cast(tmp1, srcTensor)
    Cast(tmp2, dstTensor)
    Muls(tmp3, tmp1, scalar)
    Add(tmp4, tmp2, tmp3)
    Cast(dstTensor, tmp4)
    ```
    > 使用了四倍数据大小的临时空间，对应上面`AXPY_HALF_CALC_PROC`的值。
    
    考虑到Vector Core中矢量计算单元单次Repeat计算的大小为256B，256对应上面`AXPY_ONE_REPEAT_BYTE_SIZE`的值，那么最终最大临时空间的计算公式如下：
    $$MaxValue=\begin{cases}\max(inputSize*typeSize,\tiny AXPY\_ONE\_REPEAT\_BYTE\_SIZE \normalsize)*\tiny AXPY\_HALF\_CALC\_PROC &dstType = half\\\max(inputSize*typeSize,\tiny AXPY\_ONE\_REPEAT\_BYTE\_SIZE \normalsize)*\tiny AXPY\_FLOAT\_CALC\_PROC &dstType = float\end{cases}$$
    ```c++
   inline uint32_t GetAxpyMaxTmpSize(const uint32_t inputSize, const uint32_t typeSize)
    {
        const uint8_t calcPro = typeSize == sizeof(float) ? AXPY_FLOAT_CALC_PROC : AXPY_HALF_CALC_PROC;
        return inputSize * typeSize > AXPY_ONE_REPEAT_BYTE_SIZE ?
                calcPro * inputSize * typeSize :
                calcPro * AXPY_ONE_REPEAT_BYTE_SIZE; // All temporary variables are float.
    }
    ```
    对于最小临时空间，对总的计算量进行切分，每次计算数据量固定256B大小，那么临时空间计算公式如下:
    $$MinValue=\begin{cases}\tiny AXPY\_ONE\_REPEAT\_BYTE\_SIZE*AXPY\_HALF\_CALC\_PROC &dstType = half\\\tiny AXPY\_ONE\_REPEAT\_BYTE\_SIZE*AXPY\_FLOAT\_CALC\_PROC &dstType = float\end{cases}$$
    ```c++
    inline uint32_t GetAxpyMinTmpSize(const uint32_t typeSize)
    {
        return AXPY_ONE_REPEAT_BYTE_SIZE * (typeSize == sizeof(float) ? AXPY_FLOAT_CALC_PROC : AXPY_HALF_CALC_PROC);
    }
    ```
    编写完Tiling侧实现文件后还需要在[impl/aicore/adv_api/tiling/CMakeLists.txt](../impl/aicore/adv_api/tiling/CMakeLists.txt)中引入Tiling侧实现文件：
    在`add_library(tiling_api STATIC ...)`语句中新增文件路径`${CMAKE_CURRENT_SOURCE_DIR}/math/axpy_tiling_impl.cpp`。
## 测试
对于新增API，用户可以编写对应的测试用例以及测试代码。测试主要包含UT测试和搭建简易自定义算子工程进行的单算子调用。
### UT测试
使用gTest作为测试框架，UT测试一般验证接口编译是否正常，并不能为接口功能的正常性做看护。
#### Kernel侧UT测试用例
对应文件为[test/unit/aicore/adv_api/math/axpy/test_operator_axpy.cpp](../test/unit/aicore/adv_api/math/axpy/test_operator_axpy.cpp)。
主要分为三部分：
1. 引入头文件
    ```c++
    #include <gtest/gtest.h>
    #include "kernel_operator.h"
    ``` 
2. 算子调用代码
    - 初始化变量
        ```c++
        TPipe tpipe;
        TQue<TPosition::VECIN, 1> vecInQue;
        TQue<TPosition::VECIN, 1> vecOutQue;
        TQue<TPosition::VECIN, 1> vecTmpQue;
        GlobalTensor<U> inputGlobal;
        GlobalTensor<T> outputGlobal;
        inputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ U*>(srcGm), dataSize);
        outputGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(dstGm), dataSize);
        tpipe.InitBuffer(vecInQue, 1, dataSize * sizeof(U));
        tpipe.InitBuffer(vecOutQue, 1, dataSize * sizeof(T));
        if (sizeof(U) == sizeof(float)) {
            tpipe.InitBuffer(vecTmpQue, 1, dataSize * sizeof(float));
        } else {
            tpipe.InitBuffer(vecTmpQue, 1, dataSize * 4 * sizeof(half));
        }
        ```
    - GlobalMemory数据拷贝到LocalMemory上
        ```c++
        LocalTensor<U> inputLocal = vecInQue.AllocTensor<U>();
        LocalTensor<T> outputLocal = vecOutQue.AllocTensor<T>();
        LocalTensor<uint8_t> tmpLocal = vecTmpQue.AllocTensor<uint8_t>();

        AscendCUtils::SetMask<uint8_t>(256);
        DataCopy(inputLocal, inputGlobal, dataSize);
        ```
    - 调用高阶API进行计算
        ```c++
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendCUtils::SetMask<uint8_t>(128);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        U scalar = 4;

        Axpy<T, U, false>(outputLocal, inputLocal, scalar, tmpLocal, dataSize);
        ```
    - LocalMemory数据拷贝到GlobalMemory上
        ```c++
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        DataCopy(outputGlobal, outputLocal, dataSize);
        PipeBarrier<PIPE_ALL>();
        ```
    - 释放LocalMemory空间
        ```c++
        vecInQue.FreeTensor(inputLocal);
        vecOutQue.FreeTensor(outputLocal);
        vecTmpQue.FreeTensor(tmpLocal);
        ```
3. 测试代码
    - 定义入参的数据结构。
        ```c++
        struct AxpyTestParams {
        int32_t dataSize;
        int32_t dataBitSize;
        void (*calFunc)(uint8_t*, uint8_t*, int32_t);
        };
        ```
    - 编写测试类。
        ```c++
        class AxpyTestsuite : public testing::Test, public testing::WithParamInterface<AxpyTestParams> {
        protected:
            void SetUp() {}
            void TearDown() {}
        };
        ``` 
    - 注入用例数据。
        ```c++
        INSTANTIATE_TEST_CASE_P(TEST_AXPY, AxpyTestsuite,
            ::testing::Values(AxpyTestParams{256, 2, AxpyKernel<half, half>}, 
                AxpyTestParams{256, 4, AxpyKernel<float, float>}));
        ```
    - 编写`TEST_P`。
        ```c++
        TEST_P(AxpyTestsuite, AxpyTestCase)
        {
            auto param = GetParam();
            uint8_t srcGm[param.dataSize * param.dataBitSize] = {0};
            uint8_t dstGm[param.dataSize * param.dataBitSize] = {0};

            param.calFunc(srcGm, dstGm, param.dataSize);
            for (int32_t i = 0; i < param.dataSize; i++) { EXPECT_EQ(dstGm[i], 0x00); }
        }
        ```
#### Tiling侧UT测试用例
对应文件为[test/unit/aicore/adv_api/tiling/test_tiling.cpp](../test/unit/aicore/adv_api/tiling/test_tiling.cpp)，该文件调用大量`TEST_F`，对Tiling的UT不需要新增文件，只需要在其中添加一段测试函数即可。
```c++
TEST_F(TestTiling, TestAxpyTiling)
{
    uint32_t maxVal = 0;
    uint32_t minVal = 0;
    GetAxpyMaxMinTmpSize(ge::Shape({128}), 4, false, maxVal, minVal);
    EXPECT_EQ(maxVal, 0);
    EXPECT_EQ(minVal, 0);
    GetAxpyMaxMinTmpSize(ge::Shape({256}), 2, false, maxVal, minVal);
    EXPECT_EQ(maxVal, 256 * 4 * 2);
    EXPECT_EQ(minVal, 256 * 4);
}
```
#### 修改cmake文件
- [test/unit/aicore/adv_api/CMakeLists.txt](../test/)
    在`file(GLOB ASCENDC_TEST_ascend910B1_AIV_CASE_SRC_FILES ...)`语句中新增文件路径`${ASCENDC_ADV_API_TESTS_DIR}/math/axpy/test_operator_axpy.cpp`
#### 执行UT
进入到仓库主目录下，执行`bash build.sh -t`进行测试。
如果想要仅运行新增的UT测试用例，可以在[test/unit/aicore/adv_api/main.cpp](../test/unit/aicore/adv_api/main.cpp)和[test/unit/aicore/adv_api/tiling/test_tiling.cpp](../test/unit/aicore/adv_api/tiling/test_tiling.cpp)下main函数中添加下面的代码，利用gTest的过滤器根据单元测试单元的名字过滤测试用例，这两个分别是gTest的入口文件。
```c++
 //   <---- 加上这行
```
### 简易自定义算子工程验证
编写高阶API后，如果需要验证单算子API调用，可以参考下面的两步。
- 编译安装。参考主仓库[README](../README.md)。
- 编写简易自定义算子工程。详见[官网指南-简易自定义算子工程](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0101.html)。

## 基于原有API进行开发
有时原有API存在部分数据类型不支持的情况，假设Axpy不支持目标操作数元素为float类型，那么就需要对`AxpyIntrinsicsImpl`进行重载，当dstTensor中数据类型为float时会执行下面的函数。
```c++
template <>
__aicore__ inline void AxpyIntrinsicsImpl(const LocalTensor<float>& dstTensor, const LocalTensor<half>& srcTensor,
    const half& scalarValue, LocalTensor<float> stackBuffer, uint32_t stackSize)
{
    const UnaryRepeatParams unaryParams;
    const BinaryRepeatParams binaryParams;

    Cast<float, half, false>(stackBuffer, srcTensor, RoundMode::CAST_NONE, MASK_PLACEHOLDER, 1,
        {1, 1, DEFAULT_REPEAT_STRIDE, HALF_DEFAULT_REPEAT_STRIDE});
    PipeBarrier<PIPE_V>();

    Muls<float, false>(stackBuffer, stackBuffer, (float)scalarValue, MASK_PLACEHOLDER, 1, unaryParams);
    PipeBarrier<PIPE_V>();

    Add<float, false>(dstTensor, stackBuffer, dstTensor, MASK_PLACEHOLDER, 1, binaryParams);
    PipeBarrier<PIPE_V>();
}
```
在修改API后，添加源操作数元素为half类型，目标操作数元素float类型的测试用例。
```c++
INSTANTIATE_TEST_CASE_P(TEST_AXPY, AxpyTestsuite,
::testing::Values(AxpyTestParams{256, 2, AxpyKernel<half, half>}, 
AxpyTestParams{256, 4, AxpyKernel<float, half>}, //  新增
AxpyTestParams{256, 4, AxpyKernel<float, float>}));
```
进入到仓库主目录下，执行`bash build.sh -t`进行测试。如果想要仅测试当前用例可以参考上文。
## PR
具体可参考[贡献指南](../README.md#贡献指南)。
## 注意事项
- 提交PR前，需要签署[CLA](https://clasign.osinfra.cn/sign/gitee_ascend-1720446461942705242)。
