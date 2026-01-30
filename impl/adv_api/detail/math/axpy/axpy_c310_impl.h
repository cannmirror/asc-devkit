/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file axpy_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_AXPY_AXPY_C310_IMPL_H
#define IMPL_MATH_AXPY_AXPY_C310_IMPL_H
#include "kernel_tensor.h"
#include "kernel_basic_intf.h"
#include "../../common/check.h"
#ifdef ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_check/math/axpy/axpy_check.h"
#endif // ASCENDC_CPU_DEBUG
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
namespace AxpyAPI {
constexpr MicroAPI::CastTrait castTraitF162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
template<typename T, typename U>
__simd_vf__ inline void AxpyCompute(__ubuf__ T* dst, __ubuf__ U* src, U scalarValue, uint32_t calCount,
                                   uint16_t repeatTimes, uint16_t oneRepSize, uint32_t mainBlockCount,
                                   uint32_t tailCount, uint16_t offset, uint16_t singleMainBlockCtrl)
{
    MicroAPI::MaskReg mask, maskTail;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::RegTensor<U> srcVreg;
    mask = MicroAPI::UpdateMask<T>(mainBlockCount);
    maskTail = MicroAPI::UpdateMask<T>(tailCount);
    if constexpr (IsSameType<U, half>::value && IsSameType<T, float>::value) {
        MicroAPI::RegTensor<float> tempSrcVreg;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            MicroAPI::LoadAlign<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, src + i * oneRepSize);
            MicroAPI::Cast<float, U, castTraitF162F32>(tempSrcVreg, srcVreg, mask);
            MicroAPI::LoadAlign(dstVreg, dst + i * oneRepSize);
            MicroAPI::Axpy(dstVreg, tempSrcVreg, scalarValue, mask);
            MicroAPI::StoreAlign(dst + i * oneRepSize, dstVreg, mask);
            // unroll
            MicroAPI::LoadAlign<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, src + i * oneRepSize + offset);
            MicroAPI::Cast<float, U, castTraitF162F32>(tempSrcVreg, srcVreg, mask);
            MicroAPI::LoadAlign(dstVreg, dst + i * oneRepSize + offset);
            MicroAPI::Axpy(dstVreg, tempSrcVreg, scalarValue, mask);
            MicroAPI::StoreAlign(dst + i * oneRepSize + offset, dstVreg, mask);
        }
        for (uint16_t j = 0; j < singleMainBlockCtrl; ++j) {
            MicroAPI::LoadAlign<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, src + repeatTimes * oneRepSize * 2);
            MicroAPI::Cast<float, U, castTraitF162F32>(tempSrcVreg, srcVreg, mask);
            MicroAPI::LoadAlign(dstVreg, dst + repeatTimes * oneRepSize * 2);
            MicroAPI::Axpy(dstVreg, tempSrcVreg, scalarValue, mask);
            MicroAPI::StoreAlign(dst + repeatTimes * oneRepSize * 2, dstVreg, mask);
        }
        MicroAPI::LoadAlign<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(
            srcVreg, src + repeatTimes * oneRepSize * 2 + singleMainBlockCtrl * oneRepSize);
        MicroAPI::Cast<float, U, castTraitF162F32>(tempSrcVreg, srcVreg, maskTail);
        MicroAPI::LoadAlign(dstVreg, dst + repeatTimes * oneRepSize * 2 + singleMainBlockCtrl * oneRepSize);
        MicroAPI::Axpy(dstVreg, tempSrcVreg, scalarValue, maskTail);
        MicroAPI::StoreAlign(dst + repeatTimes * oneRepSize * 2 + singleMainBlockCtrl * oneRepSize, dstVreg, maskTail);
    } else {
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            MicroAPI::LoadAlign(srcVreg, src + i * oneRepSize);
            MicroAPI::LoadAlign(dstVreg, dst + i * oneRepSize);
            MicroAPI::Axpy(dstVreg, srcVreg, scalarValue, mask);
            MicroAPI::StoreAlign(dst + i * oneRepSize, dstVreg, mask);
            // unroll
            MicroAPI::LoadAlign(srcVreg, src + i * oneRepSize + offset);
            MicroAPI::LoadAlign(dstVreg, dst + i * oneRepSize + offset);
            MicroAPI::Axpy(dstVreg, srcVreg, scalarValue, mask);
            MicroAPI::StoreAlign(dst + i * oneRepSize + offset, dstVreg, mask);
        }
        for (uint16_t j = 0; j < singleMainBlockCtrl; ++j) {
            MicroAPI::LoadAlign(srcVreg, src + repeatTimes * oneRepSize * 2);
            MicroAPI::LoadAlign(dstVreg, dst + repeatTimes * oneRepSize * 2);
            MicroAPI::Axpy(dstVreg, srcVreg, scalarValue, mask);
            MicroAPI::StoreAlign(dst + repeatTimes * oneRepSize * 2, dstVreg, mask);
        }
        MicroAPI::LoadAlign(srcVreg, src + repeatTimes * oneRepSize * 2 + singleMainBlockCtrl * oneRepSize);
        MicroAPI::LoadAlign(dstVreg, dst + repeatTimes * oneRepSize * 2 + singleMainBlockCtrl * oneRepSize);
        MicroAPI::Axpy(dstVreg, srcVreg, scalarValue, maskTail);
        MicroAPI::StoreAlign(dst + repeatTimes * oneRepSize * 2 + singleMainBlockCtrl * oneRepSize, dstVreg, maskTail);
    }
}
}//namespace AxpyAPI
template <typename T, typename U, bool isReuseSource>
__aicore__ inline void AxpyImpl(const LocalTensor<T> &dstLocal, const LocalTensor<U> &srcLocal, const U scalarValue, 
                                const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    CHECK_FUNC_HIGHLEVEL_API(Axpy, (T, U, isReuseSource), (dstLocal, srcLocal, scalarValue, sharedTmpBuffer, calCount));
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(dstLocal, "dstLocal", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcLocal, "srcLocal", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", dstLocal, "dstLocal", "Axpy");
    CheckCalCount(calCount, "calCount", srcLocal, "srcLocal", "Axpy");
    static_assert(SupportType<T, half, float>(), "Axpy current dst data type is not supported on current device!");
    static_assert(SupportType<U, half, float>(), "Axpy current src data type is not supported on current device!");
    __ubuf__ T *dst = (__ubuf__ T *)dstLocal.GetPhyAddr();
    __ubuf__ U *src = (__ubuf__ U *)srcLocal.GetPhyAddr();
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    const uint32_t mainBlockCount = oneRepSize;
    uint32_t tailCount = calCount % oneRepSize;
    uint16_t repeatTimes = calCount / oneRepSize;
    if (tailCount == 0 && repeatTimes > 0) {
        repeatTimes--;
        tailCount += oneRepSize;
    }
    uint16_t repeatTimesUnRoll = repeatTimes / 2;
    uint16_t singleMainBlockCtrl = repeatTimes % 2;
    uint16_t offset = repeatTimesUnRoll * oneRepSize;
    AxpyAPI::AxpyCompute<T, U>(dst, src, scalarValue, calCount, repeatTimesUnRoll, oneRepSize, mainBlockCount,
                                        tailCount, offset, singleMainBlockCtrl);
}
} // namespace AscendC
#endif // IMPL_MATH_AXPY_AXPY_C310_IMPL_H