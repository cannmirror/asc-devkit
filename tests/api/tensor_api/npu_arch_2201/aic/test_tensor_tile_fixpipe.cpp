/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <gtest/gtest.h>
#include "tensor_api/stub/cce_stub.h"
#include "include/experimental/tensor_api/tensor.h"

class TEST_TENSOR_TILE_FIXPIPE : public testing::Test {
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

template <CubeFormat FORMAT, typename TYPE> struct InputInfo {
    constexpr static CubeFormat format = FORMAT;
    using T = TYPE;
};

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class Q0C_TYPE, class C_TYPE, class BIAS_TYPE, int HAS_BIAS> class E2eCase {
    using SrcT = typename A_TYPE::T;
    using Src1T = typename B_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using L0cT = typename L0C_TYPE::T;
    using Q0cT = typename Q0C_TYPE::T;

public:
    __aicore__ inline E2eCase() {}
    __aicore__ inline void Init(int32_t m, int32_t n, int32_t k, int32_t enableBias, __gm__ SrcT* a, __gm__ Src1T* b, __gm__ DstT* c)
    {
        gmA = a;
        gmB = b;
        gmC = c;
        mLength = m;
        nLength = n;
        kLength = k;
        enBias = enableBias;
        a1Addr = reinterpret_cast<__cbuf__ SrcT*>(0);
        b1Addr = reinterpret_cast<__cbuf__ Src1T*>(m * k * sizeof(SrcT));
        qAddr = reinterpret_cast<__cbuf__ Q0cT*>(m * k * sizeof(SrcT) + k * n * sizeof(Src1T));
        l0aAddr = reinterpret_cast<__ca__ SrcT*>(0);
        l0bAddr = reinterpret_cast<__cb__ Src1T*>(0);
        l0cAddr = reinterpret_cast<__cc__ L0cT*>(0);

    }

    __aicore__ inline void CopyGmToA1()
    {
        using namespace AscendC::Te;

        auto gmIterator = MakeGMmemPtr(gmA);
        auto gmMatrixLayout = MakeNzLayout<SrcT>(mLength, kLength);
        auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

        auto aIterator = MakeL1memPtr(a1Addr);
        auto aMatrixLayout = MakeNzLayout<SrcT>(mLength, kLength);
        auto aTensor = MakeTensor(aIterator, aMatrixLayout);
    }

    __aicore__ inline void CopyGmToB1()
    {
        using namespace AscendC::Te;

        auto gmIterator = MakeGMmemPtr(gmB);
        auto gmMatrixLayout = MakeNzLayout<Src1T>(mLength, kLength);
        auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

        auto bIterator = MakeL1memPtr(b1Addr);
        auto bMatrixLayout = MakeNzLayout<Src1T>(kLength, nLength);
        auto bTensor = MakeTensor(bIterator, bMatrixLayout);
    }

    __aicore__ inline void Load2DA1ToL0A()
    {
        using namespace AscendC::Te;

        auto a1Iterator = MakeL1memPtr(a1Addr);
        auto a1MatrixLayout = MakeNzLayout<SrcT>(mLength, kLength);
        auto a1Tensor = MakeTensor(a1Iterator, a1MatrixLayout);

        auto l0aIterator = MakeL0AmemPtr(l0aAddr);
        auto l0aMatrixLayout = MakeZzLayout<SrcT>(mLength, kLength);
        auto l0aTensor = MakeTensor(l0aIterator, l0aMatrixLayout); 
    }

    __aicore__ inline void Load2DA1ToL0B()
    {
        using namespace AscendC::Te;

        auto b1Iterator = MakeL1memPtr(b1Addr);
        auto b1MatrixLayout = MakeNzLayout<Src1T>(kLength, nLength);
        auto b1Tensor = MakeTensor(b1Iterator, b1MatrixLayout);

        auto l0bIterator = MakeL0BmemPtr(l0bAddr);
        auto l0bMatrixLayout = MakeZnLayout<Src1T>(kLength, nLength);
        auto l0bTensor = MakeTensor(l0bIterator, l0bMatrixLayout); 
    }
        
    __aicore__ inline void Compute()
    {
        using namespace AscendC::Te;
        using namespace AscendC::Std;

        auto l0aIterator = MakeL0AmemPtr(l0aAddr);
        auto l0aMatrixLayout = MakeZzLayout<SrcT>(mLength, kLength);
        auto l0aTensor = MakeTensor(l0aIterator, l0aMatrixLayout); 

        auto l0bIterator = MakeL0BmemPtr(l0bAddr);
        auto l0bMatrixLayout = MakeZnLayout<Src1T>(kLength, nLength);
        auto l0bTensor = MakeTensor(l0bIterator, l0bMatrixLayout); 

        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNzLayout<ignore_t>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 
    }

    __aicore__ inline void CopyL0CToGm()
    {
        using namespace AscendC::Te;
        using namespace AscendC::Std;

        constexpr static FixpipeTrait trait(static_cast<QuantMode_t>(NoQuant), false, false, 0, 0);
        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNzLayout<ignore_t>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 

        if constexpr (C_TYPE::format == CubeFormat::NZ) {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNzLayout<ignore_t>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            Fixpipe<trait>(gmTensor, l0cTensor);
            Fixpipe<trait>(gmTensor, gmTensor);
        } else {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNDLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            Fixpipe<trait>(gmTensor, l0cTensor);
            Fixpipe<trait>(gmTensor, gmTensor);
        }
    }

    __aicore__ inline void CopyCoordL0CToGm()
    {
        using namespace AscendC::Te;
        using namespace AscendC::Std;

        constexpr static FixpipeTrait trait(static_cast<QuantMode_t>(NoQuant), false, false, 0, 0);
        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNzLayout<ignore_t>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 

        if constexpr (C_TYPE::format == CubeFormat::NZ) {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNzLayout<ignore_t>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            auto coordM = Int<0>{};
            auto coordN = Int<0>{};
            auto coord = MakeCoord(coordM, coordN);

            Fixpipe<trait>(gmTensor, l0cTensor, coord);
        } else {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNDLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            auto coordM = Int<0>{};
            auto coordN = Int<0>{};
            auto coord = MakeCoord(coordM, coordN);

            Fixpipe<trait>(gmTensor, l0cTensor, coord);
        }
    }

    __aicore__ inline void CopyQuantL0CToGm()
    {
        using namespace AscendC::Te;
        using namespace AscendC::Std;

        constexpr static FixpipeTrait trait(static_cast<QuantMode_t>(F322BF16), false, false, 0, 0);
        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNzLayout<ignore_t>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 

        if constexpr (C_TYPE::format == CubeFormat::NZ) {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNzLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            Fixpipe<trait>(gmTensor, l0cTensor, static_cast<uint64_t>(0));
            Fixpipe<trait>(gmTensor, gmTensor, static_cast<uint64_t>(0));
        } else {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNDLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            Fixpipe<trait>(gmTensor, l0cTensor, static_cast<uint64_t>(0));
            Fixpipe<trait>(gmTensor, gmTensor, static_cast<uint64_t>(0));
        }
    }

    __aicore__ inline void CopyQuantCoordL0CToGm()
    {
        using namespace AscendC::Te;
        using namespace AscendC::Std;

        constexpr static FixpipeTrait trait(static_cast<QuantMode_t>(F322BF16), false, false, 0, 0);
        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNzLayout<ignore_t>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 

        if constexpr (C_TYPE::format == CubeFormat::NZ) {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNzLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            auto coordM = Int<0>{};
            auto coordN = Int<0>{};
            auto coord = MakeCoord(coordM, coordN);

            Fixpipe<trait>(gmTensor, l0cTensor, static_cast<uint64_t>(0), coord);
        } else {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNDLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            auto coordM = Int<0>{};
            auto coordN = Int<0>{};
            auto coord = MakeCoord(coordM, coordN);

            Fixpipe<trait>(gmTensor, l0cTensor, static_cast<uint64_t>(0), coord);
        }
    }

    __aicore__ inline void CopyQuantTensorL0CToGm()
    {
        using namespace AscendC::Te;
        using namespace AscendC::Std;
        
        auto qIterator = MakeL1memPtr(qAddr);
        auto qMatrixLayout = MakeNDLayout<Q0cT>(1, nLength);
        auto qTensor = MakeTensor(qIterator, qMatrixLayout);

        constexpr static FixpipeTrait trait(static_cast<QuantMode_t>(VDEQF16), false, false, 0, 0);
        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNzLayout<ignore_t>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 

        if constexpr (C_TYPE::format == CubeFormat::NZ) {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNzLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            Fixpipe<trait>(gmTensor, l0cTensor, qTensor);
        } else {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNDLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            Fixpipe<trait>(gmTensor, l0cTensor, qTensor);
        }
    }

    __aicore__ inline void CopyQuantTensorCoordL0CToGm()
    {
        using namespace AscendC::Te;
        using namespace AscendC::Std;

        auto qIterator = MakeL1memPtr(qAddr);
        auto qMatrixLayout = MakeNDLayout<Q0cT>(1, nLength);
        auto qTensor = MakeTensor(qIterator, qMatrixLayout);

        constexpr static FixpipeTrait trait(static_cast<QuantMode_t>(VDEQF16), false, false, 0, 0);
        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNzLayout<ignore_t>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 

        if constexpr (C_TYPE::format == CubeFormat::NZ) {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNzLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            auto coordM = Int<0>{};
            auto coordN = Int<0>{};
            auto coord = MakeCoord(coordM, coordN);

            Fixpipe<trait>(gmTensor, l0cTensor, qTensor, coord);
        } else {
            auto gmIterator = MakeGMmemPtr(gmC);
            auto gmMatrixLayout = MakeNDLayout<DstT>(mLength, nLength);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout); 

            auto coordM = Int<0>{};
            auto coordN = Int<0>{};
            auto coord = MakeCoord(coordM, coordN);

            Fixpipe<trait>(gmTensor, l0cTensor, qTensor, coord);
        }
    }

    __aicore__ inline void IterateAll()
    {        
        CopyGmToA1();
        CopyGmToB1();
        Load2DA1ToL0A();
        Load2DA1ToL0B();
        Compute();
        if constexpr(HAS_BIAS < 2) {
            if constexpr(HAS_BIAS % 2 == 1) {
                CopyL0CToGm();
            } else {
                CopyCoordL0CToGm();
            }
        } else if constexpr(HAS_BIAS > 1 && HAS_BIAS < 4) {
            if constexpr(HAS_BIAS % 2 == 1) {
                CopyQuantL0CToGm();
            } else {
                CopyQuantCoordL0CToGm();
            }
        } else if constexpr(HAS_BIAS > 3) {
            if constexpr(HAS_BIAS % 2 == 1) {
                CopyQuantTensorL0CToGm();
            } else {
                CopyQuantTensorCoordL0CToGm();
            }
        }
    }

private:
    int32_t mLength = 0;
    int32_t nLength = 0;
    int32_t kLength = 0;
    int32_t enBias = 0;

    __gm__ SrcT* gmA;
    __gm__ Src1T* gmB;
    __gm__ DstT* gmC;
    __cbuf__ SrcT* a1Addr;
    __cbuf__ Src1T* b1Addr;
    __cbuf__ Q0cT* qAddr;
    __ca__ SrcT* l0aAddr;
    __cb__ Src1T* l0bAddr;
    __cc__ L0cT* l0cAddr;

};

template <class A_TYPE, class B_TYPE, class L0CType, class Q0CType, class C_TYPE, class BIAS_TYPE, int HAS_BIAS>
__aicore__ inline void E2eKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, GM_ADDR biasGM, GM_ADDR quantGM, int32_t m, int32_t n, int32_t k,
    int32_t usedCoreNum, int hasBias)
{
    // cube core cases, ignore vector core
    if (g_coreType == AscendC::AIV) {
        return;
    }

    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using L0C_T = typename L0CType::T;
    using Q0C_T = typename Q0CType::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    if (block_idx >= usedCoreNum) {
        return;
    }

    auto gmA = reinterpret_cast<__gm__ A_T *>(aGM);
    auto gmB = reinterpret_cast<__gm__ B_T *>(bGM);
    auto gmC = reinterpret_cast<__gm__ C_T *>(cGM);

    bisheng::cce::set_atomic_none();
    E2eCase<A_TYPE, B_TYPE, L0CType, Q0CType, C_TYPE, BIAS_TYPE, HAS_BIAS> ins;
    ins.Init(m, n, k, hasBias, gmA, gmB, gmC);
    ins.IterateAll();
    bisheng::cce::set_atomic_none();
}

#define KERNEL_TENSOR_TILE_FIXPIPE_E2E(coreNum, M, N, K, A_Format, B_Format, C_Format, BIAS_Format, A_DType, B_DType, L0C_DType, C_DType, BIAS_DType, HAS_BIAS) \
    TEST_F(TEST_TENSOR_TILE_FIXPIPE, kernel_tensor_tile_fixpipe_##coreNum##_##M##_##N##_##K##_##A_Format##_##B_Format##_##C_Format##_##BIAS_Format##_##A_DType##_##B_DType##_##L0C_DType##_##C_DType##_##BIAS_DType##_##HAS_BIAS) \
    { \
        uint8_t aGM[M * K * sizeof(A_DType)] = {0}; \
        uint8_t bGM[K * M * sizeof(B_DType)] = {0}; \
        uint8_t cGM[M * N * sizeof(C_DType)] = {0}; \
        uint8_t biasGM[N * sizeof(BIAS_DType)] = {0}; \
        uint8_t quantGM[N * sizeof(uint64_t)] = {0}; \
        typedef InputInfo<CubeFormat::A_Format, A_DType> aType; \
        typedef InputInfo<CubeFormat::B_Format, B_DType> bType; \
        typedef InputInfo<CubeFormat::NZ, L0C_DType> l0cType; \
        typedef InputInfo<CubeFormat::ND, uint64_t> q0cType; \
        typedef InputInfo<CubeFormat::C_Format, C_DType> cType; \
        typedef InputInfo<CubeFormat::BIAS_Format, BIAS_DType> biasType; \
        E2eKernel<aType, bType, l0cType, q0cType, cType, biasType, HAS_BIAS>(aGM, bGM, cGM, biasGM, quantGM, M, N, K, coreNum, HAS_BIAS); \
        for (uint32_t i = 0; i < M * N; i++) { \
            EXPECT_EQ(cGM[i], 0x00); \
        } \
    }

KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, ND, ND, bfloat16_t, bfloat16_t, float, float, float, 0)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, NZ, ND, bfloat16_t, bfloat16_t, float, float, float, 0)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, ND, ND, bfloat16_t, bfloat16_t, float, float, float, 1)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, NZ, ND, bfloat16_t, bfloat16_t, float, float, float, 1)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, ND, ND, bfloat16_t, bfloat16_t, float, bfloat16_t, float, 2)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, NZ, ND, bfloat16_t, bfloat16_t, float, bfloat16_t, float, 2)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, ND, ND, bfloat16_t, bfloat16_t, float, bfloat16_t, float, 3)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, NZ, ND, bfloat16_t, bfloat16_t, float, bfloat16_t, float, 3)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, ND, ND, int8_t, int8_t, int32_t, half, float, 4)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, NZ, ND, int8_t, int8_t, int32_t, half, float, 4)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, ND, ND, int8_t, int8_t, int32_t, half, float, 5)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 16, 16, 16, ND, ND, NZ, ND, int8_t, int8_t, int32_t, half, float, 5)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 528, 528, 528, ND, ND, ND, ND, int8_t, int8_t, int32_t, half, float, 6)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 528, 528, 528, ND, ND, NZ, ND, int8_t, int8_t, int32_t, half, float, 6)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 528, 528, 528, ND, ND, ND, ND, int8_t, int8_t, int32_t, half, float, 7)
KERNEL_TENSOR_TILE_FIXPIPE_E2E(1, 528, 528, 528, ND, ND, NZ, ND, int8_t, int8_t, int32_t, half, float, 7)