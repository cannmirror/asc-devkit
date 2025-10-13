/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @brief mock modules for scheduler
 *
 */

#include <gtest/gtest.h>
#include "kernel_operator.h"
#include "include/adv_api/matmul/tiling.h"
#include "impl/adv_api/detail/matmul/utils/matmul_param.h"
#include "fake_modules.h"
using namespace std;
using namespace AscendC;

template <typename IMPL, typename C_T, typename A_TYPE, typename B_T, const auto& MM_CFG>
class CustomMxMmadCompute {
using A_T = typename Conditional<HasScalePosition<A_TYPE>::value, \
            typename GetL0DataType<typename A_TYPE::T, true>::Type, \
            typename GetL0DataType<typename A_TYPE::T, false>::Type>::type;
public:
    inline __aicore__ void Compute(const LocalTensor<C_T>& cMatrix, const LocalTensor<A_T>& l0A,
        const LocalTensor<B_T>& l0B, uint16_t mmadM, uint16_t mmadK, uint16_t mmadN, bool isATrans, bool isBTrans,
        uint8_t unitFlag = 0, bool cmatrixSource = false, bool cmatrixInitVal = true, bool isBias = false) {}
};

template <typename IMPL, typename TYPE, const auto& MM_CFG, typename = void>
class CustomMxLoadToL0A
{
    using A_T = typename TYPE::T;
    using SrcA2T = typename GetL0DataType<typename TYPE::T, true>::Type;
    using AuxDtype = decltype(AscendC::Impl::Detail::GetAuxDataType<TYPE>());
public:

    __aicore__ inline void Prepare(bool isATranspose, uint16_t aL1K, uint16_t aL1M) const {};

    __aicore__ inline void Load(const LocalTensor<SrcA2T> &dst, const LocalTensor<A_T> &aMatrix,
     uint16_t aL1M, uint16_t aL1K, uint16_t madM, uint16_t madK, uint16_t aL1MOffset, uint16_t aL1KOffset,
     bool isATranspose, const LocalTensor<AuxDtype> &l1AAuxMatrix = {}, uint16_t aAuxL1K = 0,
     uint16_t aAuxL1KOffset = 0, uint16_t aAuxL1MOffset = 0) const {}
};

template <typename IMPL, typename TYPE, const auto& MM_CFG, typename = void>
class CustomMxLoadToL0B
{
    using B_T = typename TYPE::T;
    using SrcB2T = typename GetL0DataType<typename TYPE::T, true>::Type;
    using AuxDtype = decltype(AscendC::Impl::Detail::GetAuxDataType<TYPE>());
public:

    __aicore__ inline void Prepare(bool isBTranspose, uint16_t bL1K) const {};

    __aicore__ inline void Load(const LocalTensor<SrcB2T> &dst, const LocalTensor<B_T> &bMatrix,
     uint16_t bL1N, uint16_t bL1K, uint16_t madN, uint16_t madK, uint16_t bL1NOffset, uint16_t bL1KOffset,
     bool isBTranspose, const LocalTensor<AuxDtype> &l1BAuxMatrix = {}, uint16_t bAuxL1K = 0,
     uint16_t bAuxL1KOffset = 0, uint16_t bAuxL1NOffset = 0) const {}
};
