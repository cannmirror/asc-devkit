/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file fmod_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_MATH_FMOD_FMOD_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_MATH_FMOD_FMOD_C310_IMPL_H
#include "kernel_tensor.h"
#include "../../common/check.h"

namespace AscendC {
template <typename T>
__aicore__ inline void FmodCompute(__local_mem__ T* dstTensor, __local_mem__ T* src0Tensor, __local_mem__ T* src1Tensor,
    uint16_t oneRepSize, uint16_t repeatTimes, uint32_t calCount)
{
    NotNumUnion inf;
    inf.i = F32_INF;
    NotNumUnion negInf;
    negInf.i = F32_NEG_INF;
    NotNumUnion nan;
    nan.i = F32_NAN;
    static constexpr MicroAPI::DivSpecificMode mode = {MicroAPI::MaskMergeMode::ZEROING, true};
    __VEC_SCOPE__
    {
        uint32_t count = calCount;
        MicroAPI::RegTensor<float> src0Reg;
        MicroAPI::RegTensor<float> src1Reg;
        MicroAPI::RegTensor<float> negReg;
        MicroAPI::RegTensor<float> divReg;
        MicroAPI::RegTensor<float> dstReg;
        MicroAPI::RegTensor<float> nanReg;

        MicroAPI::MaskReg src0Is0CmpReg;
        MicroAPI::MaskReg src0IsNeg0CmpReg;
        MicroAPI::MaskReg src0InfCmpReg;
        MicroAPI::MaskReg src0NegInfCmpReg;
        MicroAPI::MaskReg src1InfCmpReg;
        MicroAPI::MaskReg src1NegInfCmpReg;
        MicroAPI::MaskReg srcBothInfCmpReg;
        MicroAPI::MaskReg src1Not0CmpReg;
        MicroAPI::MaskReg src1NotNeg0CmpReg;
        MicroAPI::MaskReg src1NotNanCmpReg;

        MicroAPI::MaskReg maskReg;
        MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::Duplicate(nanReg, nan.f, maskFull);

        for (uint16_t i = 0; i < repeatTimes; i++) {
            maskReg = MicroAPI::UpdateMask<float>(count);
            if constexpr (IsSameType<T, half>::value) {
                MicroAPI::RegTensor<T> src0Origin;
                MicroAPI::RegTensor<T> src1Origin;
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(src0Origin, src0Tensor + i * oneRepSize);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(src1Origin, src1Tensor + i * oneRepSize);
                MicroAPI::Cast<float, T, layoutZMrgZ>(src0Reg, src0Origin, maskReg);
                MicroAPI::Cast<float, T, layoutZMrgZ>(src1Reg, src1Origin, maskReg);
            } else {
                MicroAPI::DataCopy(src0Reg, src0Tensor + i * oneRepSize);
                MicroAPI::DataCopy(src1Reg, src1Tensor + i * oneRepSize);
            }
            MicroAPI::Div<float, &mode>(divReg, src0Reg, src1Reg, maskReg);
            MicroAPI::Truncate<float, RoundMode::CAST_TRUNC, MicroAPI::MaskMergeMode::ZEROING>(dstReg, divReg, maskReg);
            MicroAPI::Neg(negReg, src1Reg, maskReg);
            MicroAPI::FusedMulDstAdd(dstReg, negReg, src0Reg, maskReg);

            // if src1Tensor is inf return src0Reg
            MicroAPI::CompareScalar(src1InfCmpReg, src1Reg, inf.f, maskReg);
            MicroAPI::Select(dstReg, src0Reg, dstReg, src1InfCmpReg);
            // if src1Tensor is -inf return src0Reg
            MicroAPI::CompareScalar(src1NegInfCmpReg, src1Reg, negInf.f, maskReg);
            MicroAPI::Select(dstReg, src0Reg, dstReg, src1NegInfCmpReg);
            // if src0Tensor is inf
            MicroAPI::CompareScalar(src0InfCmpReg, src0Reg, inf.f, maskReg);
            // if src0Tensor is -inf
            MicroAPI::CompareScalar(src0NegInfCmpReg, src0Reg, negInf.f, maskReg);
            // if src0Tensor and src1Tensor both inf return inf
            MicroAPI::MaskOr(src0InfCmpReg, src0InfCmpReg, src0NegInfCmpReg, maskReg);
            MicroAPI::MaskOr(src1InfCmpReg, src1InfCmpReg, src1NegInfCmpReg, maskReg);
            MicroAPI::MaskAnd(srcBothInfCmpReg, src0InfCmpReg, src1InfCmpReg, maskReg);
            MicroAPI::Select(dstReg, nanReg, dstReg, srcBothInfCmpReg);
            // if src0Tensor is ±0 and src1Tensor is not ±0 and not nan, return src0Tensor
            MicroAPI::CompareScalar(src0Is0CmpReg, src0Reg, float(0), maskReg);
            MicroAPI::CompareScalar(src0IsNeg0CmpReg, src0Reg, float(-0), maskReg);
            MicroAPI::MaskOr(src0Is0CmpReg, src0Is0CmpReg, src0IsNeg0CmpReg, maskReg);

            MicroAPI::CompareScalar<float, CMPMODE::NE>(src1Not0CmpReg, src1Reg, float(0), maskReg);
            MicroAPI::CompareScalar<float, CMPMODE::NE>(src1NotNeg0CmpReg, src1Reg, float(-0), maskReg);
            MicroAPI::Compare<float, CMPMODE::NE>(src1NotNanCmpReg, src1Reg, src1Reg, maskReg);
            MicroAPI::MaskNot(src1NotNanCmpReg, src1NotNanCmpReg, maskReg);

            MicroAPI::MaskOr(src1Not0CmpReg, src1Not0CmpReg, src1NotNeg0CmpReg, maskReg);
            MicroAPI::MaskAnd(src1Not0CmpReg, src1Not0CmpReg, src1NotNanCmpReg, maskReg);

            MicroAPI::MaskAnd(src0Is0CmpReg, src0Is0CmpReg, src1Not0CmpReg, maskReg);
            MicroAPI::Select(dstReg, src0Reg, dstReg, src0Is0CmpReg);

            if constexpr (IsSameType<T, half>::value) {
                MicroAPI::RegTensor<T> regT;
                MicroAPI::Cast<T, float, LayoutZMrgZRndRSatNS>(regT, dstReg, maskReg);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstTensor + i * oneRepSize, regT, maskReg);
            } else {
                MicroAPI::DataCopy(dstTensor + i * oneRepSize, dstReg, maskReg);
            }
        }
    }
}

__aicore__ inline void FmodCompute(const LocalTensor<float>& dstTensor, const LocalTensor<float>& src0Tensor,
    const LocalTensor<float>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t stackSize,
    const uint32_t calCount)
{
    __local_mem__ float* src0 = (__local_mem__ float*)src0Tensor.GetPhyAddr();
    __local_mem__ float* src1 = (__local_mem__ float*)src1Tensor.GetPhyAddr();
    __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
    constexpr uint16_t oneRepSize = static_cast<uint16_t>(GetVecLen() / sizeof(float));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, oneRepSize));

    FmodCompute(dst, src0, src1, oneRepSize, repeatTimes, calCount);
}

__aicore__ inline void FmodCompute(const LocalTensor<half>& dstTensor, const LocalTensor<half>& src0Tensor,
    const LocalTensor<half>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t stackSize,
    const uint32_t calCount)
{
    __local_mem__ half* src0 = (__local_mem__ half*)src0Tensor.GetPhyAddr();
    __local_mem__ half* src1 = (__local_mem__ half*)src1Tensor.GetPhyAddr();
    __local_mem__ half* dst = (__local_mem__ half*)dstTensor.GetPhyAddr();
    constexpr uint16_t oneRepSize = static_cast<uint16_t>(GetVecLen() / sizeof(float));
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, oneRepSize));

    FmodCompute(dst, src0, src1, oneRepSize, repeatTimes, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void FmodImpl(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor,
    const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
{
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPos(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECOUT / VECCALC", "Fmod");
    CheckTensorPos(src0Tensor, Hardware::UB, "src0Tensor", "VECIN / VECOUT / VECCALC", "Fmod");
    CheckTensorPos(src1Tensor, Hardware::UB, "src1Tensor", "VECIN / VECOUT / VECCALC", "Fmod");
    CheckTensorPos(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECOUT / VECCALC", "Fmod");

    CheckCalCount(calCount, "calCount", src0Tensor, "src0Tensor", "Fmod");
    CheckCalCount(calCount, "calCount", src1Tensor, "src1Tensor", "Fmod");
    CheckCalCount(calCount, "calCount", dstTensor, "dstTensor", "Fmod");

    ASCENDC_ASSERT((std::is_same<T, float>::value || std::is_same<T, half>::value), {
        KERNEL_LOG(KERNEL_ERROR, "Failed to check the data types, current api support data types are half/float.");
    });

    ASCENDC_ASSERT((src0Tensor.GetSize() == src1Tensor.GetSize()),
        { KERNEL_LOG(KERNEL_ERROR, "Input params.GetSize must be equal with each other!"); });

    FmodCompute(dstTensor, src0Tensor, src1Tensor, sharedTmpBuffer, src0Tensor.GetSize(), calCount);
}
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATH_FMOD_FMOD_C310_IMPL_H