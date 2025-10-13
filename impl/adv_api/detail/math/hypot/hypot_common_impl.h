/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LIB_MATH_HYPOT_IMPL_H
#define LIB_MATH_HYPOT_IMPL_H
#if defined(__DAV_C310__) || defined(__DAV_310R6__) || (__NPU_ARCH__ == 5102)
#include "kernel_tensor.h"

// Implementation Process
// 1. Use vcmp_ne for comparison to find nan.
// 2. Use vcmps_eq for comparison to find ±inf.
// 3. Use Mask Operator to get final mask.

namespace AscendC {
namespace HypotInternal {
constexpr uint16_t B_HALF_ONE = 0x3f80;
constexpr uint32_t INF = 0x7f800000;
constexpr uint32_t NEG_INF = 0xff800000;
constexpr uint16_t HALF_INF = 0x7c00;
constexpr uint16_t HALF_NEG_INF = 0xfc00;
constexpr uint16_t B_HALF_INF = 0x7f80;
constexpr uint16_t B_HALF_NEG_INF = 0xff80;

constexpr uint32_t AND_OPERATOR = 0xfe000000;
constexpr uint32_t ADD_OPERATOR = 0x7e800000;
constexpr uint32_t OR_OPERATOR = 0x800000;

constexpr MicroAPI::CastTrait HYPOT_CAST_TRAIT_RINT = { MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
    MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT };
}

template <typename T, typename U>
__aicore__ inline void CompareScalar(MicroAPI::MaskReg &cmpMaskZero, MicroAPI::MaskReg &cmpMaskSrcInf,
    MicroAPI::RegTensor<T> &vSrcTmpReg0, MicroAPI::RegTensor<T> &vSrcTmpReg1, MicroAPI::MaskReg &cmpMaskSrc0NAN,
    MicroAPI::MaskReg &cmpMaskSrc1NAN, MicroAPI::RegTensor<T> &vRegOne, MicroAPI::RegTensor<T> &vSrcReg0,
    MicroAPI::RegTensor<T> &vSrcReg1, const U INF, const U NEG_INF, MicroAPI::MaskReg maskReg)
{
    MicroAPI::MaskReg cmpMaskSrc0INF;
    MicroAPI::MaskReg cmpMaskSrc1INF;
    MicroAPI::MaskReg cmpMaskSrc0NINF;
    MicroAPI::MaskReg cmpMaskSrc1NINF;
    MicroAPI::MaskReg cmpMaskSrc0Zero;
    MicroAPI::MaskReg cmpMaskSrc1Zero;
    MicroAPI::MaskReg cmpMaskSrc0Inf;
    MicroAPI::MaskReg cmpMaskSrc1Inf;

    MicroAPI::CompareScalar<U>(cmpMaskSrc0INF, (MicroAPI::RegTensor<U> &)vSrcReg0, INF, maskReg);
    MicroAPI::CompareScalar<U>(cmpMaskSrc1INF, (MicroAPI::RegTensor<U> &)vSrcReg1, INF, maskReg);
    MicroAPI::CompareScalar<U>(cmpMaskSrc0NINF, (MicroAPI::RegTensor<U> &)vSrcReg0, NEG_INF, maskReg);
    MicroAPI::CompareScalar<U>(cmpMaskSrc1NINF, (MicroAPI::RegTensor<U> &)vSrcReg1, NEG_INF, maskReg);
    MicroAPI::CompareScalar<U>(cmpMaskSrc0Zero, (MicroAPI::RegTensor<U> &)vSrcReg0, 0, maskReg);
    MicroAPI::CompareScalar<U>(cmpMaskSrc1Zero, (MicroAPI::RegTensor<U> &)vSrcReg1, 0, maskReg);

    MicroAPI::MaskAnd(cmpMaskZero, cmpMaskSrc0Zero, cmpMaskSrc1Zero, maskReg);
    MicroAPI::MaskOr(cmpMaskSrc0Inf, cmpMaskSrc0INF, cmpMaskSrc0NINF, maskReg);
    MicroAPI::MaskOr(cmpMaskSrc1Inf, cmpMaskSrc1INF, cmpMaskSrc1NINF, maskReg);
    MicroAPI::MaskOr(cmpMaskSrcInf, cmpMaskSrc0Inf, cmpMaskSrc1Inf, maskReg);
    MicroAPI::Select(vSrcTmpReg0, vRegOne, vSrcReg0, cmpMaskSrc0NAN);
    MicroAPI::Select(vSrcTmpReg1, vRegOne, vSrcReg1, cmpMaskSrc1NAN);
    MicroAPI::Select(vSrcTmpReg0, vRegOne, vSrcReg0, cmpMaskSrcInf);
    MicroAPI::Select(vSrcTmpReg1, vRegOne, vSrcReg1, cmpMaskSrcInf);
}

__aicore__ inline void HypotCommonProcess(MicroAPI::RegTensor<float> &vSrcTmpReg0, MicroAPI::RegTensor<float> &vSrcTmpReg1,
    MicroAPI::RegTensor<float> &vDstReg0, MicroAPI::MaskReg maskReg)
{
    MicroAPI::RegTensor<float> vTmpReg0, vTmpReg1, vTmpReg2, vTmpReg3;
    MicroAPI::RegTensor<int32_t> vAndReg, vNegReg, vAddsReg, vOrReg, vMinReg, vMaxReg;
    MicroAPI::RegTensor<int32_t> vConstReg0, vConstReg1;
    MicroAPI::Abs(vSrcTmpReg0, vSrcTmpReg0, maskReg);
    MicroAPI::Abs(vSrcTmpReg1, vSrcTmpReg1, maskReg);
    MicroAPI::Min(vMinReg, (MicroAPI::RegTensor<int32_t>&)vSrcTmpReg0, (MicroAPI::RegTensor<int32_t>&)vSrcTmpReg1, maskReg);
    MicroAPI::Max(vMaxReg, (MicroAPI::RegTensor<int32_t>&)vSrcTmpReg0, (MicroAPI::RegTensor<int32_t>&)vSrcTmpReg1, maskReg);
    MicroAPI::Duplicate(vConstReg0, HypotInternal::AND_OPERATOR, maskReg);
    MicroAPI::And(vAndReg, vMaxReg, vConstReg0, maskReg);
    MicroAPI::Neg(vNegReg, vAndReg, maskReg);
    MicroAPI::Adds(vAddsReg, vNegReg, HypotInternal::ADD_OPERATOR, maskReg);
    MicroAPI::Mul(vTmpReg0, (MicroAPI::RegTensor<float>&)vMinReg, (MicroAPI::RegTensor<float>&)vAddsReg, maskReg);
    MicroAPI::Mul(vTmpReg1, (MicroAPI::RegTensor<float>&)vMaxReg, (MicroAPI::RegTensor<float>&)vAddsReg, maskReg);
    MicroAPI::Mul(vTmpReg2, vTmpReg0, vTmpReg0, maskReg);
    MicroAPI::MulAddDst(vTmpReg2, vTmpReg1, vTmpReg1, maskReg);
    MicroAPI::Sqrt(vTmpReg3, vTmpReg2, maskReg);
    MicroAPI::Duplicate(vConstReg1, HypotInternal::OR_OPERATOR, maskReg);
    MicroAPI::Or(vOrReg, vAndReg, vConstReg1, maskReg);
    MicroAPI::Mul(vDstReg0, vTmpReg3, (MicroAPI::RegTensor<float>&)vOrReg, maskReg);
}

template <typename T>
__aicore__ inline void HypotCompute(MicroAPI::RegTensor<T> &vSrcTmpReg0, MicroAPI::RegTensor<T> &vSrcTmpReg1,
    MicroAPI::RegTensor<T> &vDstReg0, MicroAPI::MaskReg maskReg)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::RegTensor<float> vDstF, vSrc0, vSrc1;
        MicroAPI::Cast<float, bfloat16_t, HypotInternal::HYPOT_CAST_TRAIT_RINT>(vSrc0, vSrcTmpReg0, maskReg);
        MicroAPI::Cast<float, bfloat16_t, HypotInternal::HYPOT_CAST_TRAIT_RINT>(vSrc1, vSrcTmpReg1, maskReg);
        
        HypotCommonProcess(vSrc0, vSrc1, vDstF, maskReg);

        MicroAPI::Cast<bfloat16_t, float, HypotInternal::HYPOT_CAST_TRAIT_RINT>(vDstReg0, vDstF, maskReg);
    } else if constexpr (IsSameType<T, half>::value) {
        MicroAPI::RegTensor<float> vDstF, vSrc0, vSrc1;
        MicroAPI::Cast<float, half, HypotInternal::HYPOT_CAST_TRAIT_RINT>(vSrc0, vSrcTmpReg0, maskReg);
        MicroAPI::Cast<float, half, HypotInternal::HYPOT_CAST_TRAIT_RINT>(vSrc1, vSrcTmpReg1, maskReg);

        HypotCommonProcess(vSrc0, vSrc1, vDstF, maskReg);
        
        MicroAPI::Cast<half, float, HypotInternal::HYPOT_CAST_TRAIT_RINT>(vDstReg0, vDstF, maskReg);
    } else {
        HypotCommonProcess(vSrcTmpReg0, vSrcTmpReg1, vDstReg0, maskReg);
    }
}

template <typename T>
__aicore__ inline void VfHypotImpl(__local_mem__ T *dstUb, __local_mem__ T *src0Ub, __local_mem__ T *src1Ub,
    const uint32_t calCount)
{
    MicroAPI::RegTensor<T> vSrcReg0;
    MicroAPI::RegTensor<T> vSrcReg1;
    MicroAPI::RegTensor<T> vDstReg0;
    MicroAPI::RegTensor<T> vRegZero;
    MicroAPI::RegTensor<T> vRegOne;
    MicroAPI::RegTensor<T> vRegInf;
    MicroAPI::RegTensor<T> vSrcTmpReg0;
    MicroAPI::RegTensor<T> vSrcTmpReg1;

    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg cmpMaskZero;
    MicroAPI::MaskReg cmpMaskSrcInf;
    MicroAPI::MaskReg cmpMaskSrc0NAN;
    MicroAPI::MaskReg cmpMaskSrc1NAN;

    uint32_t sreg = static_cast<uint32_t>(calCount);
    uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    if constexpr ((IsSameType<T, bfloat16_t>::value) || (IsSameType<T, half>::value)) {
        sregLower = sregLower >> 1;
        sreg = sreg * 2;
    }
    uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, sregLower));

    if constexpr (IsSameType<T, float>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint32_t> &)vRegInf, HypotInternal::INF);
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint32_t> &)vRegOne, 1.0f);
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint32_t> &)vRegZero, 0);
    } else if constexpr (IsSameType<T, half>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t> &)vRegInf, HypotInternal::HALF_INF);
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t> &)vRegOne, 1.0f);
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t> &)vRegZero, 0);
    } else if constexpr (IsSameType<T, bfloat16_t>::value) {
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t> &)vRegInf, HypotInternal::B_HALF_INF);
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t> &)vRegOne, HypotInternal::B_HALF_ONE);
        MicroAPI::Duplicate((MicroAPI::RegTensor<uint16_t> &)vRegZero, 0);
    }

    for (uint16_t i = 0; i < repeatTimes; ++i) {
        maskReg = MicroAPI::UpdateMask<T>(sreg);
        if constexpr ((IsSameType<T, bfloat16_t>::value) || (IsSameType<T, half>::value)) {
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vSrcReg0, src0Ub + i * sregLower);
            MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vSrcReg1, src1Ub + i * sregLower);
            MicroAPI::MaskUnPack(maskReg, maskReg);
        } else {
            MicroAPI::DataCopy<T>(vSrcReg0, src0Ub + i * sregLower);
            MicroAPI::DataCopy<T>(vSrcReg1, src1Ub + i * sregLower);
        }

        MicroAPI::Compare<T, CMPMODE::NE>(cmpMaskSrc0NAN, vSrcReg0, vSrcReg0, maskReg);
        MicroAPI::Compare<T, CMPMODE::NE>(cmpMaskSrc1NAN, vSrcReg1, vSrcReg1, maskReg);
        if constexpr (IsSameType<T, float>::value) {
            CompareScalar<T, uint32_t>(cmpMaskZero, cmpMaskSrcInf, vSrcTmpReg0, vSrcTmpReg1, cmpMaskSrc0NAN,
                cmpMaskSrc1NAN, vRegOne, vSrcReg0, vSrcReg1, HypotInternal::INF, HypotInternal::NEG_INF, maskReg);
        } else if constexpr (IsSameType<T, half>::value) {
            CompareScalar<T, uint16_t>(cmpMaskZero, cmpMaskSrcInf, vSrcTmpReg0, vSrcTmpReg1, cmpMaskSrc0NAN,
                cmpMaskSrc1NAN, vRegOne, vSrcReg0, vSrcReg1, HypotInternal::HALF_INF, HypotInternal::HALF_NEG_INF,
                maskReg);
        } else if constexpr (IsSameType<T, bfloat16_t>::value) {
            CompareScalar<T, uint16_t>(cmpMaskZero, cmpMaskSrcInf, vSrcTmpReg0, vSrcTmpReg1, cmpMaskSrc0NAN,
                cmpMaskSrc1NAN, vRegOne, vSrcReg0, vSrcReg1, HypotInternal::B_HALF_INF, HypotInternal::B_HALF_NEG_INF,
                maskReg);
        }

        HypotCompute<T>(vSrcTmpReg0, vSrcTmpReg1, vDstReg0, maskReg);

        MicroAPI::Select(vDstReg0, vRegZero, vDstReg0, cmpMaskZero);
        MicroAPI::Select(vDstReg0, vSrcReg0, vDstReg0, cmpMaskSrc0NAN);
        MicroAPI::Select(vDstReg0, vSrcReg1, vDstReg0, cmpMaskSrc1NAN);
        MicroAPI::Select(vDstReg0, vRegInf, vDstReg0, cmpMaskSrcInf);
        if constexpr ((IsSameType<T, bfloat16_t>::value) || (IsSameType<T, half>::value)) {
            MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dstUb + i * sregLower, vDstReg0, maskReg);
        } else {
            MicroAPI::DataCopy<T>(dstUb + i * sregLower, vDstReg0, maskReg);
        }
    }
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void HypotImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor,
    const LocalTensor<T> &src1Tensor, const uint32_t calCount)
{
    static_assert((SupportType<T, half, bfloat16_t, float>(),
        "Hypot only support half/bfloat16_t/float data type on current device!"));
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }

    CheckTensorPos<T>(dstTensor, Hardware::UB, "dstTensor", "VECIN / VECCALC / VECOUT", "Hypot");
    CheckTensorPos<T>(src0Tensor, Hardware::UB, "src0Tensor", "VECIN / VECCALC / VECOUT", "Hypot");
    CheckTensorPos<T>(src1Tensor, Hardware::UB, "src1Tensor", "VECIN / VECCALC / VECOUT", "Hypot");
    ASCENDC_ASSERT((calCount <= src0Tensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not be larger than src0Tensor length %u", calCount,
            src0Tensor.GetSize());
    });
    ASCENDC_ASSERT((calCount <= src1Tensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not be larger than src1Tensor length %u", calCount,
            src1Tensor.GetSize());
    });
    ASCENDC_ASSERT((calCount <= dstTensor.GetSize()), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should not be larger than dstTensor length %u", calCount,
            dstTensor.GetSize());
    });

    __local_mem__ T *src0Ub = (__local_mem__ T *)src0Tensor.GetPhyAddr();
    __local_mem__ T *src1Ub = (__local_mem__ T *)src1Tensor.GetPhyAddr();
    __local_mem__ T *dstUb = (__local_mem__ T *)dstTensor.GetPhyAddr();

    VF_CALL<VfHypotImpl<T>>(dstUb, src0Ub, src1Ub, calCount);
}

template <typename T, bool isReuseSource = false>
__aicore__ inline void HypotImpl(const LocalTensor<T> &dstTensor, const LocalTensor<T> &src0Tensor,
    const LocalTensor<T> &src1Tensor, const LocalTensor<uint8_t> &sharedTmpBuffer, const uint32_t calCount)
{
    // Only for AI Vector Core.
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPos<uint8_t>(sharedTmpBuffer, Hardware::UB, "sharedTmpBuffer", "VECIN / VECCALC / VECOUT", "Hypot");

    HypotImpl(dstTensor, src0Tensor, src1Tensor, calCount);
}
} // namespace AscendC
#endif
#endif // IMPL_MATH_HYPOT_HYPOT_COMMON_IMPL_H
