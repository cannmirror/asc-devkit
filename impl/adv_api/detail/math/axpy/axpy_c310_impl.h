/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file axpy_c310_impl.h
 * \brief
 */
#ifndef IMPL_MATH_AXPY_AXPY_C310_IMPL_H
#define IMPL_MATH_AXPY_AXPY_C310_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"
#include "../../api_check/kernel_api_check.h"

namespace AscendC {
namespace AxpyAPI {
constexpr MicroAPI::CastTrait castTraitF162F32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN, MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
template<typename T, typename U>
__simd_vf__ inline void AxpyCompute(__local_mem__ T* dst, __local_mem__ U* src, U scalarValue, uint32_t calCount, uint16_t repeatTimes, uint16_t oneRepSize)
{
    MicroAPI::MaskReg mask;
    MicroAPI::RegTensor<T> dstVreg;
    MicroAPI::RegTensor<U> srcVreg;
    if constexpr (IsSameType<U, half>::value && IsSameType<T, half>::value) {
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = MicroAPI::UpdateMask<T>(calCount);
            MicroAPI::DataCopy(srcVreg, src + i * oneRepSize);
            MicroAPI::DataCopy(dstVreg, dst + i * oneRepSize);
            MicroAPI::Axpy(dstVreg, srcVreg, scalarValue, mask);
            MicroAPI::DataCopy(dst + i * oneRepSize, dstVreg, mask);
        }
    } else if constexpr (IsSameType<U, half>::value && IsSameType<T, float>::value) {
        MicroAPI::RegTensor<float> tempSrcVreg;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = MicroAPI::UpdateMask<T>(calCount);
            MicroAPI::DataCopy<U, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcVreg, src + i * oneRepSize);
            MicroAPI::Cast<float, U, castTraitF162F32>(tempSrcVreg, srcVreg, mask);
            MicroAPI::DataCopy(dstVreg, dst + i * oneRepSize);
            MicroAPI::Axpy(dstVreg, tempSrcVreg, scalarValue, mask);
            MicroAPI::DataCopy(dst + i * oneRepSize, dstVreg, mask);
        }
    } else if constexpr (IsSameType<U, float>::value && IsSameType<T, float>::value) {
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            mask = MicroAPI::UpdateMask<T>(calCount);
            MicroAPI::DataCopy(srcVreg, src + i * oneRepSize);
            MicroAPI::DataCopy(dstVreg, dst + i * oneRepSize);
            MicroAPI::Axpy(dstVreg, srcVreg, scalarValue, mask);
            MicroAPI::DataCopy(dst + i * oneRepSize, dstVreg, mask);
        }
    }
}
}//namespace AxpyAPI
template <typename T, typename U, bool isReuseSource>
__aicore__ inline void AxpyImpl(const LocalTensor<T>& dstLocal, const LocalTensor<U>& srcLocal,
const U scalarValue , const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    CHECK_FUNC_HIGHLEVEL_API(Axpy, (T, U, isReuseSource), (dstLocal, srcLocal, scalarValue, sharedTmpBuffer, calCount));
    CheckTensorPosition(sharedTmpBuffer, "sharedTmpBuffer", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(dstLocal, "dstLocal", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcLocal, "srcLocal", "VECIN, VECOUT, VECCALC");
    CheckCalCount(calCount, "calCount", dstLocal, "dstLocal", "Axpy");
    CheckCalCount(calCount, "calCount", srcLocal, "srcLocal", "Axpy");
    static_assert(SupportType<T, half, float>(), "Axpy current dst data type is not supported on current device!");
    static_assert(SupportType<U, half, float>(), "Axpy current src data type is not supported on current device!");
    __local_mem__ T *dst = (__local_mem__ T *)dstLocal.GetPhyAddr();
    __local_mem__ U *src = (__local_mem__ U *)srcLocal.GetPhyAddr();
    constexpr uint16_t oneRepSize = GetVecLen() / sizeof(T);
    uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);
    AxpyAPI::AxpyCompute<T, U>(dst, src, scalarValue, calCount, repeatTimes, oneRepSize);
}
} // namespace AscendC
#endif // IMPL_MATH_AXPY_AXPY_C310_IMPL_H
