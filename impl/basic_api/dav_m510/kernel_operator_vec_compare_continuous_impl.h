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
 * \file kernel_operator_vec_compare_continuous_impl.h
 * \brief
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_COMPARE_CONTINUOUS_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_COMPARE_CONTINUOUS_IMPL_H

#include "kernel_utils.h"

namespace AscendC {

// Compare::Level 2 - counter mode

template <typename T, typename U, CMPMODE cmpMode>
__aicore__ inline void CompareLevel2(__ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint32_t calCount)
{
    __VEC_SCOPE__
    {
        uint32_t repeatElm = GetVecLen() / sizeof(T);
        uint16_t repeatTime = CeilDivision(calCount, repeatElm);
        uint32_t sreg = calCount;
        MicroAPI::MaskReg dstReg, mask;
        MicroAPI::UnalignReg uReg;
        if constexpr (sizeof(T) == 8) {
            repeatElm = repeatElm * 2;
            repeatTime = CeilDivision(calCount, repeatElm);
            __ubuf__ uint32_t *dstT = reinterpret_cast<__ubuf__ uint32_t*>(dst);
            MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> src0Reg, src1Reg;
            for (uint16_t i = 0; i < repeatTime; ++i) {
                mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
                MicroAPI::LoadAlign(src0Reg, src0 + i * repeatElm);
                MicroAPI::LoadAlign(src1Reg, src1 + i * repeatElm);
                MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, mask);
                MicroAPI::StoreUnAlign(dstT, dstReg, uReg);
            }
            MicroAPI::StoreUnAlignPost<uint32_t, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dstT, uReg, 0);
        } else {
            MicroAPI::RegTensor<T> src0Reg, src1Reg;
            uint32_t offset = GetVecLen() / sizeof(T) / 8;
            __ubuf__ T *dstT = reinterpret_cast<__ubuf__ T*>(dst);
            for (uint16_t i = 0; i < repeatTime; ++i) {
                mask = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::LoadAlign(src0Reg, src0 + i * repeatElm);
                MicroAPI::LoadAlign(src1Reg, src1 + i * repeatElm);
                MicroAPI::Compare<T, cmpMode>(dstReg, src0Reg, src1Reg, mask);
                if constexpr (sizeof(T) == 1) {
                    MicroAPI::StoreAlign(dst + i * offset, dstReg);
                } else {
                    MicroAPI::StoreUnAlign(dstT, dstReg, uReg);
                }
            }
            if constexpr (sizeof(T) > 1) {
                MicroAPI::StoreUnAlignPost<T, MicroAPI::PostLiteral::POST_MODE_NORMAL>(dstT, uReg, 0);
            }
        }
    }
}

template <typename T, typename U, bool isSetMask = true>
__aicore__ inline void VcmpvImpl(
    __ubuf__ U *dst, __ubuf__ T *src0, __ubuf__ T *src1, CMPMODE cmpMode, const uint32_t calCount)
{
    static_assert(SupportType<T, half, int16_t, uint16_t, int32_t, uint32_t, float, uint8_t, int8_t, bfloat16_t,
        uint64_t, int64_t>(), "current data type is not supported!");
    static_assert(SupportType<U, uint8_t, int8_t>(), "current data type is not supported!");
    switch (cmpMode) {
        case CMPMODE::LT: {
            CompareLevel2<T, U, CMPMODE::LT>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GT: {
            CompareLevel2<T, U, CMPMODE::GT>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::EQ: {
            CompareLevel2<T, U, CMPMODE::EQ>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::LE: {
            CompareLevel2<T, U, CMPMODE::LE>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::GE: {
            CompareLevel2<T, U, CMPMODE::GE>(dst, src0, src1, calCount);
            break;
        }
        case CMPMODE::NE: {
            CompareLevel2<T, U, CMPMODE::NE>(dst, src0, src1, calCount);
            break;
        }
        default:
            break;
    }
}


} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_COMPARE_CONTINUOUS_IMPL_H
