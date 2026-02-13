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

class Tensor_Api_GEMM_Empty_Routing : public testing::Test {
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

template <class A_TYPE, class B_TYPE, class L0C_TYPE, class C_TYPE, class BIAS_TYPE, bool HAS_BIAS> class E2eCase {
    using SrcT = typename A_TYPE::T;
    using Src1T = typename B_TYPE::T;
    using DstT = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;
    using L0cT = typename L0C_TYPE::T;

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
        a1Addr = reinterpret_cast<__cbuf__ SrcT*>(A1);
        b1Addr = reinterpret_cast<__cbuf__ Src1T*>(B1);
        l0aAddr = reinterpret_cast<__ca__ SrcT*>(A2);
        l0bAddr = reinterpret_cast<__cb__ Src1T*>(B2);
        l0cAddr = reinterpret_cast<__cc__ DstT*>(C2);

        fbAddr = reinterpret_cast<__fbuf__ SrcT*>(FB);
        l1Addr = reinterpret_cast<__cbuf__ DstT*>(L1BT);
        btAddr = reinterpret_cast<__biasbuf__ DstT*>(BT);
    }

    __aicore__ inline void CopyGmToA1()
    {
        using namespace AscendC::Std;
        using namespace AscendC::Te;

        auto aIterator = MakeL1memPtr(a1Addr);
        auto aMatrixLayout = MakeNZLayout<SrcT>(mLength, kLength);
        auto aTensor = MakeTensor(aIterator, aMatrixLayout);

        auto coord = MakeCoord(
            MakeCoord(Int<0>{}, Int<0>{}),
            MakeCoord(Int<0>{}, Int<0>{})
        );

        DataCopy<DEFAULT_DATA_COPY_TRAIT>(aTensor, aTensor, coord);
    }

    __aicore__ inline void CopyGmToB1()
    {
        using namespace AscendC::Std;
        using namespace AscendC::Te;
        
        auto bIterator = MakeL1memPtr(b1Addr);
        auto bMatrixLayout = MakeNZLayout<Src1T>(kLength, nLength);
        auto bTensor = MakeTensor(bIterator, bMatrixLayout);

        auto coord = MakeCoord(
            MakeCoord(Int<0>{}, Int<0>{}),
            MakeCoord(Int<0>{}, Int<0>{})
        );

        DataCopy<DEFAULT_DATA_COPY_TRAIT>(bTensor, bTensor, coord);
    }

    __aicore__ inline void CopyA1ToFB()
    {
        using namespace AscendC::Std;
        using namespace AscendC::Te;

        auto fbIterator = MakeFixbufmemPtr(fbAddr);
        auto fbMatrixLayout = MakeRowMajorLayout<SrcT>(mLength, kLength);
        auto fbTensor = MakeTensor(fbIterator, fbMatrixLayout); 

        auto coord = MakeCoord(
            MakeCoord(Int<0>{}, Int<0>{}),
            MakeCoord(Int<0>{}, Int<0>{})
        );

        DataCopy<DEFAULT_DATA_COPY_TRAIT>(fbTensor, fbTensor, coord);
    }

    __aicore__ inline void CopyA1ToBT()
    {
        using namespace AscendC::Std;
        using namespace AscendC::Te;

        auto btIterator = MakeBiasmemPtr(btAddr);
        auto btMatrixLayout = MakeRowMajorLayout<DstT>(mLength, kLength);
        auto btTensor = MakeTensor(btIterator, btMatrixLayout); 

        auto coord = MakeCoord(
            MakeCoord(Int<0>{}, Int<0>{}),
            MakeCoord(Int<0>{}, Int<0>{})
        );

        DataCopy<DEFAULT_DATA_COPY_TRAIT>(btTensor, btTensor, coord);
    }

    __aicore__ inline void Load2DA1ToL0A()
    {
        using namespace AscendC::Std;
        using namespace AscendC::Te;

        auto l0aIterator = MakeL0AmemPtr(l0aAddr);
        auto l0aMatrixLayout = MakeZZLayout<SrcT>(mLength, kLength);
        auto l0aTensor = MakeTensor(l0aIterator, l0aMatrixLayout); 

        auto coord = MakeCoord(
            MakeCoord(Int<0>{}, Int<0>{}),
            MakeCoord(Int<0>{}, Int<0>{})
        );

        LoadData<DEFAULT_LOAD_DATA_TRAIT>(l0aTensor, l0aTensor, coord);
    }

    __aicore__ inline void Load2DA1ToL0B()
    {
        using namespace AscendC::Std;
        using namespace AscendC::Te;

        auto l0bIterator = MakeL0BmemPtr(l0bAddr);
        auto l0bMatrixLayout = MakeZNLayout<Src1T>(kLength, nLength);
        auto l0bTensor = MakeTensor(l0bIterator, l0bMatrixLayout); 

        auto coord = MakeCoord(
            MakeCoord(Int<0>{}, Int<0>{}),
            MakeCoord(Int<0>{}, Int<0>{})
        );

        LoadData<DEFAULT_LOAD_DATA_TRAIT>(l0bTensor, l0bTensor, coord);
    }
        
    __aicore__ inline void ComputeBt()
    {
        using namespace AscendC::Std;
        using namespace AscendC::Te;

        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNZLayout<ignore_t>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 
        
        auto biasIterator = MakeL0CmemPtr(btAddr);
        auto biasMatrixLayout = MakeRowMajorLayout<BiasT>(1, nLength);
        auto biasTensor = MakeTensor(biasIterator, biasMatrixLayout); 

        Mmad(l0cTensor, l0cTensor, l0cTensor, biasTensor, defaultMmadWithBiasParams);
    }
        
    __aicore__ inline void Compute()
    {
        using namespace AscendC::Std;
        using namespace AscendC::Te;

        auto l0bIterator = MakeL0BmemPtr(l0bAddr);
        auto l0bMatrixLayout = MakeZNLayout<Src1T>(kLength, nLength);
        auto l0bTensor = MakeTensor(l0bIterator, l0bMatrixLayout); 

        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNZLayout<ignore_t>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 
        Mmad(l0cTensor, l0cTensor, l0bTensor, defaultMmadParams);
    }

    __aicore__ inline void CopyL0CToGm()
    {
        using namespace AscendC::Te;

        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeNZLayout<DstT>(mLength, nLength);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout); 
    }

    __aicore__ inline void IterateAll()
    {
        CopyGmToA1();
        CopyGmToB1();
        CopyA1ToFB();
        CopyA1ToBT();
        Load2DA1ToL0A();
        Load2DA1ToL0B();
        if constexpr (HAS_BIAS) {
            ComputeBt();
        } else {
            Compute();
        }
        CopyL0CToGm();
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
    __ca__ SrcT* l0aAddr;
    __cb__ Src1T* l0bAddr;
    __cc__ DstT* l0cAddr;
    __fbuf__ SrcT* fbAddr;

    __cbuf__ DstT* l1Addr;
    __biasbuf__ DstT* btAddr;

    uint8_t A1[256 * 256 * sizeof(SrcT)] = {0};
    uint8_t B1[256 * 256 * sizeof(SrcT)] = {0};
    uint8_t A2[256 * 256 * sizeof(SrcT)] = {0};
    uint8_t B2[256 * 256 * sizeof(SrcT)] = {0};
    uint8_t C2[256 * 256 * sizeof(SrcT)] = {0};
    uint8_t FB[256 * 256 * sizeof(SrcT)] = {0};

    uint8_t L1BT[256 * 256 * sizeof(DstT)] = {0};
    uint8_t BT[256 * 256 * sizeof(DstT)] = {0};

};

template <class A_TYPE, class B_TYPE, class L0CType, class C_TYPE, class BIAS_TYPE, bool HAS_BIAS>
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
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    if (block_idx >= usedCoreNum) {
        return;
    }

    auto gmA = reinterpret_cast<__gm__ A_T *>(aGM);
    auto gmB = reinterpret_cast<__gm__ B_T *>(bGM);
    auto gmC = reinterpret_cast<__gm__ C_T *>(cGM);

    bisheng::cce::set_atomic_none();
    E2eCase<A_TYPE, B_TYPE, L0CType, C_TYPE, BIAS_TYPE, HAS_BIAS> ins;
    ins.Init(m, n, k, hasBias, gmA, gmB, gmC);
    ins.IterateAll();
    bisheng::cce::set_atomic_none();
}

#define KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(coreNum, M, N, K, A_Format, B_Format, C_Format, BIAS_Format, A_DType, B_DType, C_DType, BIAS_DType, HAS_BIAS) \
    TEST_F(Tensor_Api_GEMM_Empty_Routing, kernel_tensor_tile_gemm_empty_routing##coreNum##_##M##_##N##_##K##_##A_Format##_##B_Format##_##C_Format##_##BIAS_Format##_##A_DType##_##B_DType##_##C_DType##_##BIAS_DType##_##HAS_BIAS) \
    { \
        uint8_t aGM[M * K * sizeof(A_DType)] = {0}; \
        uint8_t bGM[K * N * sizeof(B_DType)] = {0}; \
        uint8_t cGM[M * N * sizeof(C_DType)] = {0}; \
        uint8_t biasGM[N * sizeof(BIAS_DType)] = {0}; \
        uint8_t quantGM[N * sizeof(C_DType)] = {0}; \
        typedef InputInfo<CubeFormat::A_Format, A_DType> aType; \
        typedef InputInfo<CubeFormat::B_Format, B_DType> bType; \
        typedef InputInfo<CubeFormat::C_Format, C_DType> l0cType; \
        typedef InputInfo<CubeFormat::C_Format, C_DType> cType; \
        typedef InputInfo<CubeFormat::BIAS_Format, BIAS_DType> biasType; \
        E2eKernel<aType, bType, l0cType, cType, biasType, HAS_BIAS>(aGM, bGM, cGM, biasGM, quantGM, M, N, K, coreNum, HAS_BIAS); \
        for (uint32_t i = 0; i < M * N; i++) { \
            EXPECT_EQ(cGM[i], 0x00); \
        } \
    }
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, DN, ND, bfloat16_t, bfloat16_t, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, ND, ND, bfloat16_t, bfloat16_t, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, NZ, NZ, NZ, ND, bfloat16_t, bfloat16_t, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, DN, ND, half, half, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, ND, ND, half, half, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, NZ, NZ, NZ, ND, half, half, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, ND, ND, float, float, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, DN, ND, float, float, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 32, 32, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 0) // LOAD L1 TO L0B 3Dv2 not support
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 128, 64, 128, ND, ND, ND, ND, half, half, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 32, ND, ND, ND, ND, half, half, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 128, 64, 128, ND, ND, ND, ND, float, float, float, float, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 128, 128, 128, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 0)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 32, 64, 64, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 0)

KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, DN, ND, bfloat16_t, bfloat16_t, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, ND, ND, bfloat16_t, bfloat16_t, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, NZ, NZ, NZ, ND, bfloat16_t, bfloat16_t, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, DN, ND, half, half, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, ND, ND, half, half, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, NZ, NZ, NZ, ND, half, half, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, ND, ND, float, float, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 16, ND, ND, DN, ND, float, float, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 32, 32, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 1) // LOAD L1 TO L0B 3Dv2 not support
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 128, 64, 128, ND, ND, ND, ND, half, half, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 16, 16, 32, ND, ND, ND, ND, half, half, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 128, 64, 128, ND, ND, ND, ND, float, float, float, float, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 128, 128, 128, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 1)
KERNEL_TENSOR_TILE_GEMM_E2E_EMPTY_ROUTING(1, 32, 64, 64, ND, ND, ND, ND, int8_t, int8_t, int32_t, int32_t, 1)