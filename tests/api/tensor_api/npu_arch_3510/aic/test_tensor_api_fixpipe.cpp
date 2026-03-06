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
#include <mockcpp/mockcpp.hpp>

class TEST_TENSOR_API_FIXPIPE : public testing::Test {
protected:
    void SetUp() override {
        AscendC::SetGCoreType(1);
        is_mock_copy_matrix_cc_to_gm = true;
    }
    
    void TearDown() override {
        AscendC::SetGCoreType(0);
        is_mock_copy_matrix_cc_to_gm = false;
    }
};

using namespace AscendC::Te;

enum class CubeFormat {
    ND = 0,
    NZ,
    DN,
};

template <CubeFormat FORMAT, typename TYPE>
struct InputInfo {
    constexpr static CubeFormat format = FORMAT;
    using T = TYPE;
};

template <class L0C_TYPE, class C_TYPE, QuantMode_t QUANT_MODE, bool HAS_COORD>
class TestCase {
    using DstT = typename C_TYPE::T;
    using L0cT = typename L0C_TYPE::T;

public:
    __aicore__ inline TestCase() {}
    __aicore__ inline void TestRun(int32_t m, int32_t n, __gm__ DstT* c)
    {
        gmC_ = c;
        mLength_ = m;
        nLength_ = n;
        qAddr = reinterpret_cast<__cbuf__ uint64_t*>(0);
        l0cAddr = reinterpret_cast<__cc__ L0cT*>(0);
        constexpr static FixpipeTrait trait(QUANT_MODE, false, false, 0, 0);
        auto l0cIterator = MakeL0CmemPtr(l0cAddr);
        auto l0cMatrixLayout = MakeL0CLayout(mLength_, nLength_);
        auto l0cTensor = MakeTensor(l0cIterator, l0cMatrixLayout);
        if constexpr (C_TYPE::format == CubeFormat::ND) {
            n_size_global = n;
            m_size_global = m;
            dst_stride_global = n;
            src_stride_global = C0_SIZE / sizeof(uint16_t) * CeilAlign(m, FRACTAL_FIXED) / FRACTAL_FIXED;
            NZ2ND_en_global = true;
            NZ2DN_en_global = false;
        } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
            n_size_global = CeilAlign(n, FRACTAL_FIXED);
            m_size_global =  CeilAlign(m, C0_SIZE / sizeof(uint16_t));
            dst_stride_global = C0_SIZE / sizeof(DstT) * CeilAlign(m, FRACTAL_FIXED);
            src_stride_global = C0_SIZE / sizeof(uint16_t) * CeilAlign(m, FRACTAL_FIXED) / FRACTAL_FIXED;
            NZ2ND_en_global = false;
            NZ2DN_en_global = false;
        } else {
            n_size_global = n;
            m_size_global = m;
            dst_stride_global = m;
            src_stride_global = C0_SIZE / sizeof(uint16_t) * CeilAlign(m, FRACTAL_FIXED) / FRACTAL_FIXED;
            NZ2ND_en_global = false;
            NZ2DN_en_global = true;
        }

        auto gmTensor = MakeGMTensor();

        if constexpr (QUANT_MODE == QuantMode_t::F322F16) {
            Fixpipe<trait>(gmTensor, l0cTensor, (uint64_t)0);
        } else if constexpr (QUANT_MODE == QuantMode_t::NoQuant) {
            Fixpipe<trait>(gmTensor, l0cTensor);
        } else {
            auto qIterator = MakeL1memPtr(qAddr);
            auto qMatrixLayout = MakeNDLayout<uint64_t>(1, nLength_);
            auto qTensor = MakeTensor(qIterator, qMatrixLayout);
            Fixpipe<trait>(gmTensor, l0cIterator, qTensor);
        }
    }


private:
    int32_t mLength_ = 0;
    int32_t nLength_ = 0;

    __gm__ DstT* gmC_;
    __cbuf__ uint64_t* qAddr;
    __cc__ L0cT* l0cAddr;

    __aicore__ inline constexpr auto MakeGMTensor()
    {
        auto gmIterator = MakeGMmemPtr(gmC_);
        if constexpr (C_TYPE::format == CubeFormat::NZ) {
            auto gmMatrixLayout = MakeNzLayout<DstT>(mLength_, nLength_);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout);
            return gmTensor;
        } else if constexpr (C_TYPE::format == CubeFormat::DN) {
            auto gmMatrixLayout = MakeDNLayout<DstT>(mLength_, nLength_);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout);
            return gmTensor;
        } else {
            auto gmMatrixLayout = MakeNDLayout<DstT>(mLength_, nLength_);
            auto gmTensor = MakeTensor(gmIterator, gmMatrixLayout);
            return gmTensor;
        }
    }

};

template <class L0C_TYPE, class C_TYPE, QuantMode_t QUANT_MODE, bool HAS_COORD>
__aicore__ inline void TestFixpipe(GM_ADDR cGM, int32_t m, int32_t n, int32_t usedCoreNum)
{
    // cube core cases, ignore vector core
    if (g_coreType == AscendC::AIV) {
        return;
    }

    using L0C_T = typename L0C_TYPE::T;
    using C_T = typename C_TYPE::T;

    if (block_idx >= usedCoreNum) {
        return;
    }

    auto gmC = reinterpret_cast<__gm__ C_T *>(cGM);

    TestCase<L0C_TYPE, C_TYPE, QUANT_MODE, HAS_COORD> ins;
    ins.TestRun(m, n, gmC);
}

#define KERNEL_TENSOR_API_FIXPIPE_E2E(coreNum, M, N, C_Format, L0C_DType, C_DType, Quant_Mode, Has_Coord) \
    TEST_F(TEST_TENSOR_API_FIXPIPE, kernel_tensor_api_fixpipe_##coreNum##_##M##_##N##_##C_Format##_##L0C_DType##_##C_DType##_##Quant_Mode##_##Has_Coord) \
    { \
        uint8_t cGM[M * N * sizeof(C_DType)] = {0}; \
        typedef InputInfo<CubeFormat::NZ, L0C_DType> l0cType; \
        typedef InputInfo<CubeFormat::C_Format, C_DType> cType; \
        TestFixpipe<l0cType, cType, QuantMode_t::Quant_Mode, Has_Coord>(cGM, M, N, coreNum); \
        for (uint32_t i = 0; i < M * N; i++) { \
            EXPECT_EQ(cGM[i], 0x00); \
        } \
    }

KERNEL_TENSOR_API_FIXPIPE_E2E(1, 16, 16, ND, float, float, NoQuant, false)
KERNEL_TENSOR_API_FIXPIPE_E2E(1, 16, 16, NZ, float, float, NoQuant, false)
// KERNEL_TENSOR_API_FIXPIPE_E2E(1, 16, 16, DN, float, float, NoQuant, false)
KERNEL_TENSOR_API_FIXPIPE_E2E(1, 128, 64, ND, float, float, NoQuant, false)
KERNEL_TENSOR_API_FIXPIPE_E2E(1, 128, 64, NZ, float, float, NoQuant, false)
KERNEL_TENSOR_API_FIXPIPE_E2E(1, 16, 16, ND, float, half, F322F16, false)
