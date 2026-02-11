/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "kernel_operator.h"
// #include "api_check/kernel_cpu_check.h"

class TEST_ACTE_DATACOPY : public testing::Test {
protected:
    void SetUp()
    {
        AscendC::SetGCoreType(1);
    }
    void TearDown()
    {
        AscendC::SetGCoreType(0);
    }
};


using namespace AscendC;
enum class CubeFormat {
    ND = 0,
    NZ,
    DN,
    ZN,
    ZZ,
    NN,
    ND_ALIGN,
    SCALAR,
    VECTOR,
};

template <TPosition POSITION, CubeFormat FORMAT, typename TYPE> struct InputInfo {
    constexpr static TPosition pos = POSITION;
    constexpr static CubeFormat format = FORMAT;
    using T = TYPE;
};
constexpr int32_t NZ_MASK_VAlUE = 2;
int32_t constexpr GetNdNzMask(CubeFormat dstFormat, CubeFormat srcFormat)
{
    if ((srcFormat == CubeFormat::ND) && (dstFormat == CubeFormat::NZ)) {
        return 1;
    } else if ((srcFormat == CubeFormat::NZ) && (dstFormat == CubeFormat::ND)) {
        return NZ_MASK_VAlUE;
    }
    return 0;
}

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE, bool HAS_BIAS> class E2eCase {
    using SrcT = typename A_TYPE::T;
    using Src1T = typename B_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using L0cT = typename L0C_TYPE::T;

public:
    __aicore__ inline E2eCase() {}
    __aicore__ inline void Init(TPipe *tpipe, int32_t m, int32_t n, int32_t k, int32_t enableBias)
    {
        pipe = tpipe;
        mLength = m;
        nLength = n;
        kLength = k;
        enBias = enableBias;

        pipe->InitBuffer(qidA1_, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB1_, 1, n * k * sizeof(Src1T));

        pipe->InitBuffer(qidA2_, 1, m * k * sizeof(SrcT));
        pipe->InitBuffer(qidB2_, 1, n * k * sizeof(Src1T));

        if (enBias == 1) {
            pipe->InitBuffer(inQueueC1, 1, n * sizeof(BiasT));
            pipe->InitBuffer(outQueueC2, 1, n * sizeof(L0cT));
        }


        pipe->InitBuffer(qidCO1_, 1, m * n * sizeof(L0cT));
    }
    __aicore__ inline void SetTensorA(const GlobalTensor<SrcT> &gm)
    {
        aGlobal_ = gm;
    }
    __aicore__ inline void SetTensorB(const GlobalTensor<Src1T> &gm)
    {
        bGlobal_ = gm;
    }
    __aicore__ inline void SetBias(const GlobalTensor<BiasT> &biasGlobal)
    {
        biasGlobal_ = biasGlobal;
    }
    __aicore__ inline void SetQuant(const GlobalTensor<uint64_t> &quantGlobal)
    {
        quantGlobal_ = quantGlobal;
    }
    __aicore__ inline uint16_t CeilDiv(uint16_t num1, uint16_t num2)
    {
        ASSERT(num2 > 0);
        return (num1 + num2 - 1) / num2;
    }

    __aicore__ inline void CopyGmToA1Nd2Nz()
    {
        LocalTensor<SrcT> leftMatrix = qidA1_.template AllocTensor<SrcT>();

        auto srcLayout = AscendC::MakeRowMajorLayout<SrcT>(mLength, kLength);
        auto dstLayout = AscendC::MakeNZLayout<SrcT>(mLength, kLength);

        auto srcTrait = AscendC::MakeTensorTrait<SrcT, TPosition::GM>(srcLayout);
        auto dstTrait = AscendC::MakeTensorTrait<SrcT, TPosition::A1>(dstLayout);

        using srcTraitType = decltype(srcTrait);
        using dstTraitType = decltype(dstTrait);

        GlobalTensor<srcTraitType> aGlobalTrait;
        aGlobalTrait.SetTensorTrait(srcTrait);
        aGlobalTrait.address_ = aGlobal_.address_;
        aGlobalTrait.oriAddress_ = aGlobal_.oriAddress_;
        aGlobalTrait.bufferSize_ = aGlobal_.bufferSize_;
        aGlobalTrait.cacheMode_ = aGlobal_.cacheMode_;

        LocalTensor<dstTraitType> leftMatrixTrait;
        leftMatrixTrait.SetTensorTrait(dstTrait);
        leftMatrixTrait.SetAddr(leftMatrix.address_);

        AscendC::DataCopy(leftMatrixTrait, aGlobalTrait);
        qidA1_.EnQue(leftMatrix);
    }

    __aicore__ inline void CopyGmToB1Nd2Nz()
    {
        LocalTensor<Src1T> rightMatrix = qidB1_.template AllocTensor<Src1T>();
        Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue = kLength;
        nd2nzParams.dValue = nLength;
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = nLength;
        nd2nzParams.dstNzC0Stride = kLength;
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;
        DataCopy(rightMatrix, bGlobal_, nd2nzParams);
        qidB1_.EnQue(rightMatrix);
    }

    __aicore__ inline void CopyGmToA1()
    {
        LocalTensor<SrcT> leftMatrix = qidA1_.template AllocTensor<SrcT>();

        auto srcLayout = AscendC::MakeNZLayout<SrcT>(mLength, kLength);
        auto dstLayout = AscendC::MakeNZLayout<SrcT>(mLength, kLength);

        auto srcTrait = AscendC::MakeTensorTrait<SrcT, TPosition::GM>(srcLayout);
        auto dstTrait = AscendC::MakeTensorTrait<SrcT, TPosition::A1>(dstLayout);

        using srcTraitType = decltype(srcTrait);
        using dstTraitType = decltype(dstTrait);

        GlobalTensor<srcTraitType> aGlobalTrait;
        aGlobalTrait.SetTensorTrait(srcTrait);
        aGlobalTrait.address_ = aGlobal_.address_;
        aGlobalTrait.oriAddress_ = aGlobal_.oriAddress_;
        aGlobalTrait.bufferSize_ = aGlobal_.bufferSize_;
        aGlobalTrait.cacheMode_ = aGlobal_.cacheMode_;

        LocalTensor<dstTraitType> leftMatrixTrait;
        leftMatrixTrait.SetTensorTrait(dstTrait);
        leftMatrixTrait.SetAddr(leftMatrix.address_);

        AscendC::DataCopy(leftMatrixTrait, aGlobalTrait);

        qidA1_.EnQue(leftMatrix);
    }

    __aicore__ inline void CopyGmToB1()
    {
        LocalTensor<Src1T> rightMatrix = qidB1_.template AllocTensor<Src1T>();
        DataCopy(rightMatrix, bGlobal_, kLength * nLength);
        qidB1_.EnQue(rightMatrix);
    }
    __aicore__ inline void Load3DA1ToL0A()
    {
        auto leftMatrix = qidA1_.template DeQue<SrcT>();
        LocalTensor<SrcT> a2 = qidA2_.AllocTensor<SrcT>();
        LoadData3DParamsV2<SrcT> loadData3dParams;
        loadData3dParams.l1W = 1;
        loadData3dParams.l1H = mLength;
        loadData3dParams.channelSize = kLength;
        loadData3dParams.kExtension = kLength;
        loadData3dParams.mExtension = mLength;
        loadData3dParams.kStartPt = 0;
        loadData3dParams.mStartPt = 0;
        loadData3dParams.strideW = 1;
        loadData3dParams.strideH = 1;
        loadData3dParams.filterW = 1;
        loadData3dParams.filterH = 1;
        loadData3dParams.dilationFilterW = 1;
        loadData3dParams.dilationFilterH = 1;
        loadData3dParams.enTranspose = false;
        loadData3dParams.enSmallK = false;
        loadData3dParams.padValue = 0;
        loadData3dParams.filterSizeW = 0;
        loadData3dParams.filterSizeH = 0;
        loadData3dParams.fMatrixCtrl = false;
        uint16_t dstStride = DivCeil(mLength, 16);
        SetLoadDataRepeat({0, 1, 0, dstStride});
        LoadData(a2, leftMatrix, loadData3dParams);
        qidA2_.EnQue(a2);
        qidA1_.FreeTensor(leftMatrix);
    }

    __aicore__ inline void Load3DB1ToL0B()
    {
        auto rightMatrix = qidB1_.template DeQue<Src1T>();
        LocalTensor<Src1T> b2 = qidB2_.AllocTensor<Src1T>();
        LoadData3DParamsV2<Src1T> loadData3dParams;
        loadData3dParams.l1W = 1;
        loadData3dParams.l1H = kLength;
        loadData3dParams.channelSize = nLength;
        loadData3dParams.kExtension = nLength;
        loadData3dParams.mExtension = kLength;
        loadData3dParams.kStartPt = 0;
        loadData3dParams.mStartPt = 0;
        loadData3dParams.strideW = 1;
        loadData3dParams.strideH = 1;
        loadData3dParams.filterW = 1;
        loadData3dParams.filterH = 1;
        loadData3dParams.dilationFilterW = 1;
        loadData3dParams.dilationFilterH = 1;
        loadData3dParams.enTranspose = true;
        loadData3dParams.enSmallK = false;
        loadData3dParams.padValue = 0;
        loadData3dParams.filterSizeW = 0;
        loadData3dParams.filterSizeH = 0;
        loadData3dParams.fMatrixCtrl = false;
        uint8_t padList[PAD_SIZE] = {0, 0, 0, 0};
        SetFmatrix(1, kLength, padList, FmatrixMode::FMATRIX_LEFT);
        SetLoadDataPaddingValue(0); // bit cast
        uint16_t dstStride = DivCeil(nLength, 16);
        SetLoadDataRepeat({0, 1, 0, dstStride});
        static constexpr IsResetLoad3dConfig LOAD3D_CONFIG = {false, false};
        LoadData<Src1T, LOAD3D_CONFIG>(b2, rightMatrix, loadData3dParams);
        qidB2_.EnQue(b2);
        qidB1_.FreeTensor(rightMatrix);
    }

    __aicore__ inline void Compute()
    {
        MmadParams mmadParams;
        mmadParams.m = mLength;
        mmadParams.n = nLength;
        mmadParams.k = kLength;
        mmadParams.isBias = enBias;

        auto co1Local = qidCO1_.AllocTensor<L0cT>();
        auto a2 = qidA2_.DeQue<SrcT>();
        auto b2 = qidB2_.DeQue<Src1T>();
        pipe_barrier(PIPE_ALL);
        Mmad(co1Local, a2, b2, mmadParams);

        qidA2_.FreeTensor(a2);
        qidB2_.FreeTensor(b2);
        qidCO1_.EnQue(co1Local);
    }


    __aicore__ inline void CopyL0CToGm(const GlobalTensor<DstT> &gm)
    {
        if constexpr (C_TYPE::format ==  CubeFormat::NZ) {
            // L0C shape is (16, row), (16, column)
            // Dst shape is (16, row), (C0Size, column)
            constexpr uint32_t c0Size = TileInternal::C0_SIZE / sizeof(DstT);
            uint32_t l0cRow = mLength / 16;
            uint32_t l0cColumn = nLength / 16;

            uint32_t dstRow = mLength / 16 * (16 / c0Size);
            uint32_t dstColumn = nLength / c0Size;
            auto l0cShape = AscendC::MakeShape(AscendC::MakeShape(Std::Int<16>{}, l0cRow), AscendC::MakeShape(Std::Int<16>{}, l0cColumn));
            auto dstShape = AscendC::MakeShape(AscendC::MakeShape(Std::Int<16>{}, dstRow), AscendC::MakeShape(Std::Int<c0Size>{}, dstColumn));

            auto l0cStride = AscendC::MakeStride(AscendC::MakeStride(Std::Int<16>{}, Std::Int<256>{}), AscendC::MakeStride(Std::Int<1>{}, l0cRow * 256));
            auto dstStride = AscendC::MakeStride(AscendC::MakeStride(Std::Int<c0Size>{}, 16 * c0Size), AscendC::MakeStride(Std::Int<1>{}, dstRow * c0Size * 16));

            auto l0cLayout = AscendC::MakeLayout(l0cShape, l0cStride);
            auto dstLayout = AscendC::MakeLayout(dstShape, dstStride);

            auto l0cTrait = AscendC::MakeTensorTrait<L0cT, TPosition::CO1>(l0cLayout);
            auto dstTrait = AscendC::MakeTensorTrait<DstT, TPosition::GM>(dstLayout);

            using l0cTraitType = decltype(l0cTrait);
            using dstTraitType = decltype(dstTrait);
        
            LocalTensor<l0cTraitType> l0cLocal = qidCO1_.DeQue<l0cTraitType>();
            l0cLocal.SetTensorTrait(l0cTrait);

            GlobalTensor<dstTraitType> dstGlobal;
            dstGlobal.SetTensorTrait(dstTrait);
            dstGlobal.address_ = gm.address_;
            dstGlobal.oriAddress_ = gm.oriAddress_;
            dstGlobal.bufferSize_ = gm.bufferSize_;
            AscendC::Fixpipe(dstGlobal, l0cLocal);

            qidCO1_.FreeTensor(l0cLocal);
        } else if constexpr (C_TYPE::format ==  CubeFormat::ND) {
            // L0C shape is (16, row), (16, column)
            // Dst shape is (1, row), (1, column) stride (Int<0>, N * column), (Int<0>, Int<1>))
            uint32_t l0cRow = mLength / 16;
            uint32_t l0cColumn = nLength / 16;

            uint32_t dstRow = mLength;
            uint32_t dstColumn = nLength;
            auto l0cShape = AscendC::MakeShape(AscendC::MakeShape(Std::Int<16>{}, l0cRow), AscendC::MakeShape(Std::Int<16>{}, l0cColumn));
            auto dstShape = AscendC::MakeShape(AscendC::MakeShape(Std::Int<1>{}, dstRow), AscendC::MakeShape(Std::Int<1>{}, dstColumn));

            auto l0cStride = AscendC::MakeStride(AscendC::MakeStride(Std::Int<16>{}, Std::Int<256>{}), AscendC::MakeStride(Std::Int<1>{}, l0cRow * 256));
            auto dstStride = AscendC::MakeStride(AscendC::MakeStride(Std::Int<0>{}, dstColumn), AscendC::MakeStride(Std::Int<0>{}, Std::Int<1>{}));

            auto l0cLayout = AscendC::MakeLayout(l0cShape, l0cStride);
            auto dstLayout = AscendC::MakeLayout(dstShape, dstStride);

            auto l0cTrait = AscendC::MakeTensorTrait<L0cT, TPosition::CO1>(l0cLayout);
            auto dstTrait = AscendC::MakeTensorTrait<DstT, TPosition::GM>(dstLayout);

            using l0cTraitType = decltype(l0cTrait);
            using dstTraitType = decltype(dstTrait);
        
            LocalTensor<l0cTraitType> l0cLocal = qidCO1_.DeQue<l0cTraitType>();
            l0cLocal.SetTensorTrait(l0cTrait);

            GlobalTensor<dstTraitType> dstGlobal;
            dstGlobal.SetTensorTrait(dstTrait);
            dstGlobal.address_ = gm.address_;
            dstGlobal.oriAddress_ = gm.oriAddress_;
            dstGlobal.bufferSize_ = gm.bufferSize_;
            AscendC::Fixpipe(dstGlobal, l0cLocal);
            qidCO1_.FreeTensor(l0cLocal);
        } else if constexpr (C_TYPE::format ==  CubeFormat::DN) {
            // L0C shape is (16, row), (16, column)
            // Dst shape is (1, row), (1, column) stride (Int<0>, Int<1>),  (Int<0>, N * row))
            uint32_t l0cRow = mLength / 16;
            uint32_t l0cColumn = nLength / 16;

            uint32_t dstRow = mLength;
            uint32_t dstColumn = nLength;
            auto l0cShape = AscendC::MakeShape(AscendC::MakeShape(Std::Int<16>{}, l0cRow), AscendC::MakeShape(Std::Int<16>{}, l0cColumn));
            auto dstShape = AscendC::MakeShape(AscendC::MakeShape(Std::Int<1>{}, dstRow), AscendC::MakeShape(Std::Int<1>{}, dstColumn));

            auto l0cStride = AscendC::MakeStride(AscendC::MakeStride(Std::Int<16>{}, Std::Int<256>{}), AscendC::MakeStride(Std::Int<1>{}, l0cRow * 256));
            auto dstStride = AscendC::MakeStride(AscendC::MakeStride(Std::Int<0>{}, Std::Int<1>{}), AscendC::MakeStride(Std::Int<0>{}, dstRow));

            auto l0cLayout = AscendC::MakeLayout(l0cShape, l0cStride);
            auto dstLayout = AscendC::MakeLayout(dstShape, dstStride);

            auto l0cTrait = AscendC::MakeTensorTrait<L0cT, TPosition::CO1>(l0cLayout);
            auto dstTrait = AscendC::MakeTensorTrait<DstT, TPosition::GM>(dstLayout);

            using l0cTraitType = decltype(l0cTrait);
            using dstTraitType = decltype(dstTrait);
        
            LocalTensor<l0cTraitType> l0cLocal = qidCO1_.DeQue<l0cTraitType>();
            l0cLocal.SetTensorTrait(l0cTrait);

            GlobalTensor<dstTraitType> dstGlobal;
            dstGlobal.SetTensorTrait(dstTrait);
            dstGlobal.address_ = gm.address_;
            dstGlobal.oriAddress_ = gm.oriAddress_;
            dstGlobal.bufferSize_ = gm.bufferSize_;

            AscendC::Fixpipe(dstGlobal, l0cLocal);
            qidCO1_.FreeTensor(l0cLocal);
        }
    }

    __aicore__ inline void CopyInQuant()
    {
        LocalTensor<uint64_t> quantLocalL1 = inQueueQuantL1.AllocTensor<uint64_t>();
        DataCopy(quantLocalL1, quantGlobal_, nLength);
        uint32_t l1Row = 1;
        uint32_t l1Column = nLength;

        uint32_t fbRow = 1;
        uint32_t fbColumn = nLength;
        auto l1Shape = AscendC::MakeShape(AscendC::MakeShape(Std::Int<1>{}, l1Row), AscendC::MakeShape(Std::Int<1>{}, l1Column));
        auto fbShape = AscendC::MakeShape(AscendC::MakeShape(Std::Int<1>{}, fbRow), AscendC::MakeShape(Std::Int<1>{}, fbColumn));

        auto l1Stride = AscendC::MakeStride(AscendC::MakeStride(Std::Int<0>{}, l1Column), AscendC::MakeStride(Std::Int<0>{}, Std::Int<1>{}));
        auto fbStride = AscendC::MakeStride(AscendC::MakeStride(Std::Int<0>{}, fbColumn), AscendC::MakeStride(Std::Int<0>{}, Std::Int<1>{}));

        auto l1Layout = AscendC::MakeLayout(l1Shape, l1Stride);
        auto fbLayout = AscendC::MakeLayout(fbShape, fbStride);

        auto l1Trait = AscendC::MakeTensorTrait<uint64_t, TPosition::A1>(l1Layout);
        auto fbTrait = AscendC::MakeTensorTrait<uint64_t, TPosition::C2PIPE2GM>(fbLayout);

        using l1TraitType = decltype(l1Trait);
        using fbTraitType = decltype(fbTrait);

        LocalTensor<fbTraitType> quantLocalFB = inQueueQuantFB.AllocTensor<fbTraitType>();
        quantLocalFB.SetTensorTrait(fbTrait);

        LocalTensor<l1TraitType> quantLocalL1TensorTrait;
        quantLocalL1TensorTrait.SetTensorTrait(l1Trait);
        quantLocalL1TensorTrait.address_ = quantLocalL1.address_;

        DataCopy(quantLocalFB, quantLocalL1TensorTrait);
        inQueueQuantFB.EnQue(quantLocalFB);
        inQueueQuantL1.FreeTensor(quantLocalL1TensorTrait);
    }
    __aicore__ inline void CopyInBias()
    {
        LocalTensor<BiasT> bias1Local = inQueueC1.AllocTensor<BiasT>();
        // bt:gm->l1
        DataCopy(bias1Local, biasGlobal_, nLength);
        inQueueC1.EnQue(bias1Local);
    }

    __aicore__ inline void SplitBias()
    {
        LocalTensor<BiasT> bias1Local = inQueueC1.DeQue<BiasT>();
        LocalTensor<L0cT> bias2Local = outQueueC2.AllocTensor<L0cT>();

        auto srcLayout = AscendC::MakeRowMajorLayout<BiasT>(1, nLength);
        auto dstLayout = AscendC::MakeRowMajorLayout<L0cT>(1, nLength);

        auto srcTrait = AscendC::MakeTensorTrait<BiasT, TPosition::C1>(srcLayout);
        auto dstTrait = AscendC::MakeTensorTrait<L0cT, TPosition::C2>(dstLayout);

        using srcTraitType = decltype(srcTrait);
        using dstTraitType = decltype(dstTrait);

        LocalTensor<srcTraitType> bias1LocalTrait;
        bias1LocalTrait.SetTensorTrait(srcTrait);
        bias1LocalTrait.SetAddr(bias1Local.address_);

        LocalTensor<dstTraitType> bias2LocalTrait;
        bias2LocalTrait.SetTensorTrait(dstTrait);
        bias2LocalTrait.SetAddr(bias2Local.address_);

        AscendC::DataCopy(bias2LocalTrait, bias1LocalTrait);

        outQueueC2.EnQue<L0cT>(bias2Local);
        inQueueC1.FreeTensor(bias1Local);
    }

    __aicore__ inline void ComputeBt()
    {
        MmadParams mmadParams;
        mmadParams.m = mLength;
        mmadParams.n = nLength;
        mmadParams.k = kLength;
        mmadParams.isBias = enBias;

        auto co1Local = qidCO1_.AllocTensor<L0cT>();
        auto a2 = qidA2_.DeQue<SrcT>();
        auto b2 = qidB2_.DeQue<Src1T>();
        auto bias2Local = outQueueC2.DeQue<L0cT>();
        pipe_barrier(PIPE_ALL);

        Mmad(co1Local, a2, b2, bias2Local, mmadParams);

        qidA2_.FreeTensor(a2);
        qidB2_.FreeTensor(b2);
        qidCO1_.EnQue(co1Local);
        outQueueC2.FreeTensor(bias2Local);
    }
    __aicore__ inline void Load2DA1ToL0A()
    {
        auto srcLayout = AscendC::MakeNZLayout<SrcT>(mLength, kLength);
        auto dstLayout = AscendC::MakeNZLayout<SrcT>(mLength, kLength);

        auto srcTrait = AscendC::MakeTensorTrait<SrcT, TPosition::A1>(srcLayout);
        auto dstTrait = AscendC::MakeTensorTrait<SrcT, TPosition::A2>(dstLayout);

        using srcTraitType = decltype(srcTrait);
        using dstTraitType = decltype(dstTrait);
        
        auto a1Matrix = qidA1_.template DeQue<srcTraitType>();
        a1Matrix.SetTensorTrait(srcTrait);
        LocalTensor<SrcT> a2 = qidA2_.AllocTensor<SrcT>();
        LocalTensor<dstTraitType> a2Matrix;
        a2Matrix.SetTensorTrait(dstTrait);
        a2Matrix.SetAddr(a2.address_);
        LoadData(a2Matrix, a1Matrix);
      
        qidA2_.EnQue(a2Matrix);
        qidA1_.FreeTensor(a1Matrix);
    }

    __aicore__ inline void Load2DA1ToL0B()
    {
        auto srcLayout = AscendC::MakeNZLayout<Src1T>(kLength, nLength);
        auto shape = MakeShape(MakeShape(Std::Int<AscendC::TileInternal::C0_SIZE / sizeof(Src1T)>{}, kLength / (AscendC::TileInternal::C0_SIZE / sizeof(Src1T))),
            MakeShape(Std::Int<AscendC::TileInternal::FRACTAL_FIXED>{}, nLength / AscendC::TileInternal::FRACTAL_FIXED));
        auto stride = MakeStride(MakeStride(Std::Int<1>{}, AscendC::TileInternal::C0_SIZE / sizeof(Src1T) * nLength),
            MakeStride(Std::Int<AscendC::TileInternal::C0_SIZE / sizeof(Src1T)>{}, Std::Int<AscendC::TileInternal::C0_SIZE / sizeof(Src1T) * AscendC::TileInternal::FRACTAL_FIXED>{}));
        auto dstLayout = MakeLayout(shape, stride);

        auto srcTrait = AscendC::MakeTensorTrait<Src1T, TPosition::B1>(srcLayout);
        auto dstTrait = AscendC::MakeTensorTrait<Src1T, TPosition::B2>(dstLayout);

        using srcTraitType = decltype(srcTrait);
        using dstTraitType = decltype(dstTrait);
        
        auto b1Matrix = qidB1_.template DeQue<srcTraitType>();
        b1Matrix.SetTensorTrait(srcTrait);
        LocalTensor<Src1T> b2 = qidB2_.AllocTensor<Src1T>();
        LocalTensor<dstTraitType> b2Matrix;
        b2Matrix.SetTensorTrait(dstTrait);
        b2Matrix.SetAddr(b2.address_);
        LoadData(b2Matrix, b1Matrix);
      
        qidB2_.EnQue(b2Matrix);
        qidB1_.FreeTensor(b1Matrix);
    }

    __aicore__ inline void CopyL0CToL1ToGm(const GlobalTensor<DstT> &gm)
    {
        LocalTensor<DstT> resMatrix = qidA1_.template AllocTensor<DstT>();
        auto co1Local = qidCO1_.DeQue<L0cT>();
        if constexpr (C_TYPE::format == CubeFormat::ND) {
            FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams(nLength,
                mLength, mLength, nLength);
            fixpipeParams.params = { 1, 0, 0 };
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe<DstT, L0cT, CFG_ROW_MAJOR>(resMatrix, co1Local, fixpipeParams);
        } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
            FixpipeParamsC310<CO2Layout::NZ> fixpipeParams(nLength,
                mLength, mLength, nLength * BLOCK_CUBE);
            if (IsSameType<DstT, half>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322F16;
            } else if (IsSameType<DstT, bfloat16_t>::value) {
                fixpipeParams.quantPre = QuantMode_t::F322BF16;
            }
            Fixpipe<DstT, L0cT, CFG_NZ>(resMatrix, co1Local, fixpipeParams);
        }
        qidCO1_.FreeTensor(co1Local);
        pipe_barrier(PIPE_ALL);
        DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = 1;
        dataCopyInfo.blockLen = nLength * mLength * sizeof(DstT) / ONE_BLK_SIZE;
        DataCopy(gm, resMatrix, dataCopyInfo);
        qidA1_.FreeTensor(resMatrix);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void IterateAll(const GlobalTensor<DstT> &gm)
    {
        if (A_TYPE::format == CubeFormat::ND) {
            CopyGmToA1Nd2Nz();
            CopyGmToB1Nd2Nz();
            Load2DA1ToL0A();
            Load2DA1ToL0B();
        } else if (A_TYPE::format == CubeFormat::NZ) {
            // gm ->l1(nz->nz), l1 ->loa/lob(with transpose)
            CopyGmToA1();
            CopyGmToB1();
            Load2DA1ToL0A();
            Load2DA1ToL0B();
        }

        if constexpr(HAS_BIAS) {
            CopyInBias();
            SplitBias();
            ComputeBt();
            if constexpr(HAS_BIAS) {
                if (C_TYPE::format == CubeFormat::ND) {
                    CopyL0CToGm(gm);
                } else if (C_TYPE::format == CubeFormat::NZ) {
                    CopyL0CToGm(gm); 
                } else if (C_TYPE::format == CubeFormat::DN) {
                    CopyL0CToGm(gm); 
                }
            }
            return;
        }

        Compute();
        if (C_TYPE::format == CubeFormat::ND) {
            CopyL0CToGm(gm);
        } else if (C_TYPE::format == CubeFormat::NZ) {
            CopyL0CToGm(gm); 
        } else if (C_TYPE::format == CubeFormat::DN) {
            CopyL0CToGm(gm); 
        }
    }

private:
    TPipe *pipe = nullptr;
    int32_t mLength = 0;
    int32_t nLength = 0;
    int32_t kLength = 0;
    int32_t enBias = 0;

    TQue<TPosition::A1, 1, GetNdNzMask(CubeFormat::NZ, A_TYPE::format)> qidA1_;
    TQue<TPosition::B1, 1, GetNdNzMask(CubeFormat::NZ, B_TYPE::format)> qidB1_;
    TQue<TPosition::A2, 1> qidA2_;
    TQue<TPosition::B2, 1> qidB2_;
    TQue<TPosition::CO1, 1> qidCO1_;
    TQue<TPosition::C1, 1> inQueueC1;
    TQue<TPosition::C2, 1> outQueueC2;
    TQue<TPosition::C2PIPE2GM, 1> inQueueQuantFB;
    TQue<TPosition::A1, 1> inQueueQuantL1;

    LocalTensor<SrcT> leftMatrix_;
    LocalTensor<Src1T> rightMatrix_;
    LocalTensor<BiasT> inputBias_;
    GlobalTensor<SrcT> aGlobal_;
    GlobalTensor<Src1T> bGlobal_;
    GlobalTensor<BiasT> biasGlobal_;
    GlobalTensor<uint64_t> quantGlobal_;
};

template <class A_TYPE, class B_TYPE, class L0CType, class C_TYPE, class BIAS_TYPE, bool HAS_BIAS>
__aicore__ inline void E2eKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR quantGM, int32_t m, int32_t n, int32_t k,
    int32_t usedCoreNum, int hasBias)
{
    // cube core cases, ignore vector core
    if (g_coreType == AIV) {
        return;
    }

    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using L0C_T = typename L0CType::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    if (block_idx >= usedCoreNum) {
        return;
    }
    GlobalTensor<A_T> aGlobal;
    GlobalTensor<B_T> bGlobal;
    GlobalTensor<C_T> cGlobal;
    GlobalTensor<BiasT> biasGlobal;
    GlobalTensor<uint64_t> quantGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(aGM), m * k);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(bGM), k * n);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(cGM), m * n);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BiasT *>(biasGM), n);
    quantGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(quantGM), n);

    int offsetA = 0;
    int offsetB = 0;
    int offsetC = 0;
    int offsetBias = 0;
    int offsetQuant = 0;

    auto gmA = aGlobal[offsetA];
    auto gmB = bGlobal[offsetB];
    auto gmC = cGlobal[offsetC];
    auto gmBias = biasGlobal[offsetBias];
    auto gmQuant = quantGlobal[offsetQuant];

    set_atomic_none();
    TPipe que;
    E2eCase<A_TYPE, B_TYPE, L0CType, C_TYPE, BIAS_TYPE, HAS_BIAS> ins;
    ins.Init(&que, m, n, k, hasBias);
    ins.SetTensorA(gmA);
    ins.SetTensorB(gmB);
    if (hasBias == 1) {
        ins.SetBias(gmBias);
    }

    ins.IterateAll(gmC);
    set_atomic_none();
}

#define KERNEL_ACTE_E2E(coreNum, M, N, K, A_Pos, B_Pos, C_Pos, BIAS_Pos, A_Format, B_Format, C_Format, BIAS_Format, A_DType, B_DType, C_DType, BIAS_DType, HAS_BIAS) \
    TEST_F(TEST_ACTE_DATACOPY, kernel_acte_datacopy_##coreNum##_##M##_##N##_##K##_##A_Pos##_##B_Pos##_##C_Pos##_##BIAS_Pos##_##A_Format##_##B_Format##_##C_Format##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##HAS_BIAS) \
    { \
        uint8_t aGM[M * K * sizeof(A_DType)] = {0}; \
        uint8_t bGM[K * M * sizeof(B_DType)] = {0}; \
        uint8_t cGM[M * N * sizeof(C_DType)] = {0}; \
        uint8_t biasGM[N * sizeof(BIAS_DType)] = {0}; \
        uint8_t quantGM[N * sizeof(C_DType)] = {0}; \
        typedef InputInfo<TPosition::A_Pos, CubeFormat::A_Format, A_DType> aType; \
        typedef InputInfo<TPosition::B_Pos, CubeFormat::B_Format, B_DType> bType; \
        typedef InputInfo<TPosition::CO1, CubeFormat::C_Format, C_DType> l0cType; \
        typedef InputInfo<TPosition::C_Pos, CubeFormat::C_Format, C_DType> cType; \
        typedef InputInfo<TPosition::BIAS_Pos, CubeFormat::BIAS_Format, BIAS_DType> biasType; \
        E2eKernel<aType, bType, l0cType, cType, biasType, HAS_BIAS>(aGM, bGM, cGM, biasGM, quantGM, M, N, K, coreNum, HAS_BIAS); \
        for (uint32_t i = 0; i < M * N; i++) { \
            EXPECT_EQ(cGM[i], 0x00); \
        } \
    }

KERNEL_ACTE_E2E(1, 16, 16, 16, GM, GM, GM, GM, ND, ND, DN, ND, bfloat16_t, bfloat16_t, float, float, 1)
KERNEL_ACTE_E2E(1, 16, 16, 16, GM, GM, GM, GM, ND, ND, ND, ND, bfloat16_t, bfloat16_t, float, float, 1)
KERNEL_ACTE_E2E(1, 16, 16, 16, GM, GM, GM, GM, NZ, NZ, NZ, ND, bfloat16_t, bfloat16_t, float, float, 1)
KERNEL_ACTE_E2E(1, 16, 16, 16, GM, GM, GM, GM, ND, ND, DN, ND, half, half, float, float, 1)
KERNEL_ACTE_E2E(1, 16, 16, 16, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 1)
KERNEL_ACTE_E2E(1, 16, 16, 16, GM, GM, GM, GM, NZ, NZ, NZ, ND, half, half, float, float, 1)
KERNEL_ACTE_E2E(1, 16, 16, 16, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 1)
KERNEL_ACTE_E2E(1, 16, 16, 16, GM, GM, GM, GM, ND, ND, DN, ND, float, float, float, float, 1)
KERNEL_ACTE_E2E(1, 16, 32, 32, GM, GM, GM, GM, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 1) // LOAD L1 TO L0B 3Dv2 not support
KERNEL_ACTE_E2E(1, 128, 64, 128, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 1)
KERNEL_ACTE_E2E(1, 16, 16, 32, GM, GM, GM, GM, ND, ND, ND, ND, half, half, float, float, 1)
KERNEL_ACTE_E2E(1, 128, 64, 128, GM, GM, GM, GM, ND, ND, ND, ND, float, float, float, float, 1)
KERNEL_ACTE_E2E(1, 128, 128, 128, GM, GM, GM, GM, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 1)
KERNEL_ACTE_E2E(1, 32, 64, 64, GM, GM, GM, GM, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 1)