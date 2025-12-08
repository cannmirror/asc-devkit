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
 * \file kernel_operator_vec_vconv_impl.h
 * \brief AscendC l300 support vector cast api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_struct_unary.h"

namespace AscendC {

constexpr MicroAPI::CastTrait layoutZMrgZ = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr MicroAPI::CastTrait layoutZSatSMrgZ = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr MicroAPI::CastTrait layoutZSatSMrgZRndA = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                     MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

constexpr MicroAPI::CastTrait layoutZSatSMrgZRndR = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                     MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait layoutZMrgZRndR = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait layoutZMrgZRndA = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

constexpr MicroAPI::CastTrait layoutZMrgZRndC = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_CEIL};

constexpr MicroAPI::CastTrait layoutZMrgZRndF = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};

constexpr MicroAPI::CastTrait layoutZMrgZRndZ = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                 MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

constexpr MicroAPI::CastTrait MrgZRndR = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                          MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait MrgZRndA = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                          MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

constexpr MicroAPI::CastTrait MrgZRndF = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                          MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_CEIL};

constexpr MicroAPI::CastTrait MrgZRndC = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                          MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};

constexpr MicroAPI::CastTrait MrgZRndZ = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::UNKNOWN,
                                          MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

constexpr MicroAPI::CastTrait MrgZRndRSatS = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait MrgZRndASatS = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

constexpr MicroAPI::CastTrait MrgZRndFSatS = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_CEIL};

constexpr MicroAPI::CastTrait MrgZRndCSatS = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_FLOOR};

constexpr MicroAPI::CastTrait MrgZRndZSatS = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_TRUNC};

constexpr MicroAPI::CastTrait LayoutZMrgZRndRSatS = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                     MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait LayoutZMrgZRndASatS = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                     MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

constexpr MicroAPI::CastTrait LayoutZMrgZRndRSatNS = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr MicroAPI::CastTrait LayoutZMrgZRndASatNS = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                      MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

constexpr MicroAPI::CastTrait MrgZRndRSatNS = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::NO_SAT,
                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

namespace CastParam {
constexpr MicroAPI::CastTrait AddReluCastTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait SubReluCastTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait s162halfTrait = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                               MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr MicroAPI::CastTrait s162f32CastTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait f322s16CastTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                  MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
constexpr MicroAPI::CastTrait TrueHalfBlockCastTrait = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::SAT,
                                                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait FalseHalfBlockCastTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                         MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait TrueHalfBlockCastNoSatTrait = {MicroAPI::RegLayout::ONE, MicroAPI::SatMode::NO_SAT,
                                                             MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr MicroAPI::CastTrait FalseHalfBlockCastNoSatTrait = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::NO_SAT,
                                                              MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
}  // namespace CastParam

// micro adaptor
template <typename T, typename U, RoundMode roundMode, Mode mode, SatMode satMode, PartMode partMode>
__aicore__ inline void CastAdaptor(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    if constexpr (roundMode != RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN && partMode != PartMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode, satMode, partMode>(dstReg, srcReg, mask);
    } else if constexpr (roundMode != RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN &&
                         partMode == PartMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode, satMode>(dstReg, srcReg, mask);
    } else if constexpr (roundMode != RoundMode::CAST_NONE && satMode == SatMode::UNKNOWN &&
                         partMode != PartMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode, partMode>(dstReg, srcReg, mask);
    } else if constexpr (roundMode == RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN &&
                         partMode != PartMode::UNKNOWN) {
        return Cast<T, U, mode, satMode, partMode>(dstReg, srcReg, mask);
    } else if constexpr (roundMode == RoundMode::CAST_NONE && satMode == SatMode::UNKNOWN &&
                         partMode != PartMode::UNKNOWN) {
        return Cast<T, U, mode, partMode>(dstReg, srcReg, mask);
    } else if constexpr (roundMode != RoundMode::CAST_NONE && satMode == SatMode::UNKNOWN &&
                         partMode == PartMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode>(dstReg, srcReg, mask);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport pattern of CastAdaptor"); });
    }
}

template <typename T, typename U, RoundMode roundMode, Mode mode, SatMode satMode, PPMode ppMode>
__aicore__ inline void CastAdaptor(RegTensor<T> &dstReg, RegTensor<U> &srcReg, MaskReg &mask)
{
    if constexpr (roundMode != RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN && ppMode != PPMode::UNKNOWN) {
        return Cast<T, U, roundMode, mode, satMode, ppMode>(dstReg, srcReg, mask);
    }
    if constexpr (roundMode == RoundMode::CAST_NONE && satMode != SatMode::UNKNOWN && ppMode != PPMode::UNKNOWN) {
        return Cast<T, U, mode, satMode, ppMode>(dstReg, srcReg, mask);
    } else if constexpr (roundMode == RoundMode::CAST_NONE && satMode == SatMode::UNKNOWN &&
                         ppMode != PPMode::UNKNOWN) {
        return Cast<T, U, mode, ppMode>(dstReg, srcReg, mask);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport pattern of CastAdaptor"); });
    }
}

// For Cast L2
#define CAST_LOWER_HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)              \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType *dst, __ubuf__ srcType *src,      \
                                                      const uint32_t calCount)                           \
    {                                                                                                    \
        __VEC_SCOPE__                                                                                    \
        {                                                                                                \
            uint32_t len = calCount;                                                                     \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(dstType));                         \
            uint16_t repeatTimes = CeilDivision(calCount, sregLower);                                    \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                       \
                RegTensor<srcType> input0;                                                               \
                RegTensor<srcType> input1;                                                               \
                RegTensor<dstType> mid0;                                                                 \
                RegTensor<dstType> mid1;                                                                 \
                RegTensor<dstType> output;                                                               \
                MaskReg p0 = CreatePredicate<dstType>(len);                                              \
                DataCopy(input0, src, i *sregLower);                                                     \
                DataCopy(input1, src + (sregLower >> 1), i * sregLower);                                 \
                DeInterleave(input0, input1, input0, input1);                                            \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::EVEN>(mid0, input0, p0); \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::ODD>(mid1, input1, p0);  \
                Or(output, mid0, mid1, p0);                                                              \
                DataCopy(dst, output, i *sregLower, p0);                                                 \
            }                                                                                            \
        }                                                                                                \
    }

// For Cast L2
#define CAST_UPPER_HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)                             \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType* dst,                                            \
        __ubuf__ srcType* src, const uint32_t calCount)                                                                 \
    {                                                                                                                   \
        __VEC_SCOPE__                                                                                                   \
        {                                                                                                               \
            uint32_t len = calCount;                                                                                    \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(srcType));                                        \
            uint16_t repeatTimes = CeilDivision(calCount, sregLower);                                                   \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                                      \
                RegTensor<srcType> input;                                                                               \
                RegTensor<dstType> output0;                                                                             \
                RegTensor<dstType> output1;                                                                             \
                RegTensor<dstType> output_even;                                                                         \
                RegTensor<dstType> output_odd;                                                                          \
                MaskReg p0 = CreatePredicate<srcType>(len);                                                             \
                MaskReg p1;                                                                                             \
                MaskReg p2;                                                                                             \
                DataCopy(input, src, i * sregLower);                                                                    \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::EVEN>(output_even, input, p0);          \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::ODD>(output_odd, input, p0);            \
                Interleave(output0, output1, output_even, output_odd);                                                  \
                PredicateUnPack<HiloPart::Lower>(p1, p0);                                                               \
                PredicateUnPack<HiloPart::Higher>(p2, p0);                                                              \
                DataCopy(dst, output0, i * sregLower, p1);                                                              \
                DataCopy(dst + (sregLower >> 1), output1, i * sregLower, p2);                                           \
            }                                                                                                           \
        }                                                                                                               \
    }                                                                                                                   \

// For Cast L2 s322half(s32->float->half)
#define CAST_S322HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)                              \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(                                                                 \
        __ubuf__ dstType *dst, __ubuf__ srcType *src, const uint32_t calCount)                                         \
    {                                                                                                                  \
        float deqValueTmp = static_cast<float>(g_deqValue);                                                            \
        __VEC_SCOPE__                                                                                                  \
        {                                                                                                              \
            uint32_t len = calCount;                                                                                   \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(dstType));                                       \
            uint16_t repeatTimes = CeilDivision(calCount, sregLower);                                                  \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                                     \
                RegTensor<srcType> input1, input2, input_even, input_odd;                                              \
                RegTensor<float> intermediate_even, intermediate_odd;                                                  \
                RegTensor<dstType> output1, output2, output;                                                           \
                MaskReg pg = CreatePredicate<dstType>(len);                                                            \
                MaskReg pg_all = CreatePredicate<float>();                                                             \
                DataCopy(input1, src, i *sregLower);                                                                   \
                DataCopy(input2, src, (sregLower >> 1) + i * sregLower);                                               \
                DeInterleave(input_even, input_odd, input1, input2);                                                   \
                CastAdaptor<float, srcType, rndMode, mode, SatMode::UNKNOWN, PartMode::UNKNOWN>(                       \
                    intermediate_even, input_even, pg_all);                                                            \
                CastAdaptor<float, srcType, rndMode, mode, SatMode::UNKNOWN, PartMode::UNKNOWN>(                       \
                    intermediate_odd, input_odd, pg_all);                                                              \
                Muls(intermediate_even, intermediate_even, deqValueTmp, pg_all);                                       \
                Muls(intermediate_odd, intermediate_odd, deqValueTmp, pg_all);                                         \
                CastAdaptor<dstType, float, rndMode, mode, satMode, PartMode::EVEN>(                                   \
                    output1, intermediate_even, pg_all);                                                               \
                CastAdaptor<dstType, float, rndMode, mode, satMode, PartMode::ODD>(output2, intermediate_odd, pg_all); \
                Or((RegTensor<uint16_t> &)output, (RegTensor<uint16_t> &)output1, (RegTensor<uint16_t> &)output2, pg); \
                DataCopy(dst, output, i *sregLower, pg);                                                               \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \

// For Cast L2 half2s4(half->int4b_t)
#define CAST_HALF2S4(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)                         \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(                                                           \
        __ubuf__ dstType *dst, __ubuf__ srcType *src, const uint32_t calCount)                                   \
    {                                                                                                            \
        __VEC_SCOPE__                                                                                            \
        {                                                                                                        \
            uint16_t oneRepSize = (uint32_t)(VECTOR_REG_WIDTH / sizeof(srcType));                                \
            uint16_t repeatTimes = CeilDivision(calCount, oneRepSize);                                           \
            uint32_t sreg = static_cast<uint32_t>(calCount);                                                     \
            static constexpr MicroAPI::CastTrait castTrait = {                                                   \
                MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, rndMode};   \
            MicroAPI::MaskReg preg, pregDst, pregTemp;                                                           \
            MicroAPI::RegTensor<srcType> srcVreg;                                                                \
            MicroAPI::RegTensor<int4x2_t> dstVreg;                                                               \
            MicroAPI::RegTensor<uint8_t> tmpVreg;                                                                \
            for (uint16_t i = 0; i < repeatTimes; ++i) {                                                         \
                preg = MicroAPI::UpdateMask<srcType>(sreg);                                                      \
                MicroAPI::MaskPack(pregTemp, preg);                                                              \
                MicroAPI::MaskPack(pregDst, pregTemp);                                                           \
                MicroAPI::DataCopy(srcVreg, src + i * oneRepSize);                                               \
                MicroAPI::Cast<int4x2_t, srcType, castTrait>(dstVreg, srcVreg, preg);                            \
                MicroAPI::DeInterleave((MicroAPI::RegTensor<uint8_t> &)dstVreg, tmpVreg,                         \
                                       (MicroAPI::RegTensor<uint8_t> &)dstVreg, tmpVreg);                        \
                MicroAPI::DeInterleave((MicroAPI::RegTensor<uint8_t> &)dstVreg, tmpVreg,                         \
                                       (MicroAPI::RegTensor<uint8_t> &)dstVreg, tmpVreg);                        \
                MicroAPI::DataCopy( (__ubuf__ uint8_t *)dst + (i * oneRepSize) / 2,                         \
                    (MicroAPI::RegTensor<uint8_t> &)dstVreg, pregDst);                                           \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \

// For Cast L2
#define CAST_LOWER_QUATER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)           \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType *dst, __ubuf__ srcType *src,     \
                                                      const uint32_t calCount)                          \
    {                                                                                                   \
        __VEC_SCOPE__                                                                                   \
        {                                                                                               \
            uint32_t len = calCount;                                                                    \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(dstType));                        \
            uint16_t repeatTimes = CeilDivision(calCount, sregLower);                                   \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                      \
                RegTensor<srcType> input0;                                                              \
                RegTensor<srcType> input1;                                                              \
                RegTensor<srcType> input2;                                                              \
                RegTensor<srcType> input3;                                                              \
                RegTensor<dstType> output;                                                              \
                RegTensor<dstType> mid0;                                                                \
                RegTensor<dstType> mid1;                                                                \
                RegTensor<dstType> mid2;                                                                \
                RegTensor<dstType> mid3;                                                                \
                MaskReg p0 = CreatePredicate<dstType>(len);                                             \
                DataCopy(input0, src, i *sregLower);                                                    \
                DataCopy(input1, src + (sregLower >> 2) * 1, i * sregLower);                            \
                DataCopy(input2, src + (sregLower >> 2) * 2, i * sregLower);                            \
                DataCopy(input3, src + (sregLower >> 2) * 3, i * sregLower);                            \
                DeInterleave(input0, input1, input0, input1);                                           \
                DeInterleave(input2, input3, input2, input3);                                           \
                DeInterleave(input0, input2, input0, input2);                                           \
                DeInterleave(input1, input3, input1, input3);                                           \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ZERO>(mid0, input0, p0);  \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ONE>(mid1, input1, p0);   \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::TWO>(mid2, input2, p0);   \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::THREE>(mid3, input3, p0); \
                Or(mid0, mid0, mid1, p0);                                                               \
                Or(mid2, mid2, mid3, p0);                                                               \
                Or(output, mid0, mid2, p0);                                                             \
                DataCopy(dst, output, i *sregLower, p0);                                                \
            }                                                                                           \
        }                                                                                               \
    }

// For Cast L2
#define CAST_UPPER_QUATER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)          \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType *dst, __ubuf__ srcType *src,    \
                                                      const uint32_t calCount)                         \
    {                                                                                                  \
        __VEC_SCOPE__                                                                                  \
        {                                                                                              \
            uint32_t len = calCount;                                                                   \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(srcType));                       \
            uint16_t repeatTimes = CeilDivision(calCount, sregLower);                                  \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                     \
                RegTensor<srcType> input;                                                              \
                RegTensor<dstType> output0;                                                            \
                RegTensor<dstType> output1;                                                            \
                RegTensor<dstType> output2;                                                            \
                RegTensor<dstType> output3;                                                            \
                RegTensor<dstType> mid0;                                                               \
                RegTensor<dstType> mid1;                                                               \
                RegTensor<dstType> mid2;                                                               \
                RegTensor<dstType> mid3;                                                               \
                RegTensor<dstType> mid4;                                                               \
                RegTensor<dstType> mid5;                                                               \
                RegTensor<dstType> mid6;                                                               \
                RegTensor<dstType> mid7;                                                               \
                MaskReg p0 = CreatePredicate<srcType>(len);                                            \
                MaskReg p1;                                                                            \
                MaskReg p2;                                                                            \
                MaskReg p11;                                                                           \
                MaskReg p12;                                                                           \
                MaskReg p21;                                                                           \
                MaskReg p22;                                                                           \
                DataCopy(input, src, i *sregLower);                                                    \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ZERO>(mid0, input, p0);  \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ONE>(mid1, input, p0);   \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::TWO>(mid2, input, p0);   \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::THREE>(mid3, input, p0); \
                Interleave(mid4, mid5, mid0, mid2);                                                    \
                Interleave(mid6, mid7, mid1, mid3);                                                    \
                Interleave(output0, output1, mid4, mid6);                                              \
                Interleave(output2, output3, mid5, mid7);                                              \
                PredicateUnPack<HiloPart::Lower>(p1, p0);                                              \
                PredicateUnPack<HiloPart::Lower>(p11, p1);                                             \
                PredicateUnPack<HiloPart::Higher>(p12, p1);                                            \
                PredicateUnPack<HiloPart::Higher>(p2, p0);                                             \
                PredicateUnPack<HiloPart::Lower>(p21, p2);                                             \
                PredicateUnPack<HiloPart::Higher>(p22, p2);                                            \
                DataCopy(dst, output0, i *sregLower, p11);                                             \
                DataCopy(dst + (sregLower >> 2) * 1, output1, i * sregLower, p12);                     \
                DataCopy(dst + (sregLower >> 2) * 2, output2, i * sregLower, p21);                     \
                DataCopy(dst + (sregLower >> 2) * 3, output3, i * sregLower, p22);                     \
            }                                                                                          \
        }                                                                                              \
    }

// For Cast L2
#define CAST_EQUAL(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)                       \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType *dst, __ubuf__ srcType *src,          \
                                                      const uint32_t calCount)                               \
    {                                                                                                        \
        __VEC_SCOPE__                                                                                        \
        {                                                                                                    \
            uint32_t len = calCount;                                                                         \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(srcType));                             \
            uint16_t repeatTimes = CeilDivision(calCount, sregLower);                                        \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                                           \
                RegTensor<srcType> input;                                                                    \
                RegTensor<dstType> output;                                                                   \
                MaskReg p0 = CreatePredicate<dstType>(len);                                                  \
                DataCopy(input, src, i *sregLower);                                                          \
                CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::UNKNOWN>(output, input, p0); \
                DataCopy(dst, output, i *sregLower, p0);                                                     \
            }                                                                                                \
        }                                                                                                    \
    }

// For Truncate L2
#define CAST_TRUNCATE(dType, rndStr, rndMode, mode)                                             \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dType *dst, __ubuf__ dType *src, \
                                                      const uint32_t calCount)                  \
    {                                                                                           \
        __VEC_SCOPE__                                                                           \
        {                                                                                       \
            uint32_t len = calCount;                                                            \
            uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(dType));                  \
            uint16_t repeatTimes = CeilDivision(calCount, sregLower);                           \
            for (uint16_t i = 0; i < (uint16_t)repeatTimes; i++) {                              \
                RegTensor<dType> input;                                                         \
                RegTensor<dType> output;                                                        \
                MaskReg p0 = CreatePredicate<dType>(len);                                       \
                DataCopy(input, src, i *sregLower);                                             \
                Truncate<dType, rndMode, mode>(output, input, p0);                              \
                DataCopy(dst, output, i *sregLower, p0);                                        \
            }                                                                                   \
        }                                                                                       \
    }

#define CAST_TO_EQUAL(dstType, srcType, rndMode, satMode, mode)                                     \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::UNKNOWN>(vreg1, vreg0, preg)

#define LOWER_TO_HALF(dstType, srcType, rndMode, satMode, mode) \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::EVEN>(vreg1, vreg0, preg)

#define UPPER_TO_HALF(dstType, srcType, rndMode, satMode, mode) \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PartMode::EVEN>(vreg1, vreg0, preg_dst)

#define LOWER_TO_QUATER(dstType, srcType, rndMode, satMode, mode) \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ZERO>(vreg1, vreg0, preg)

#define UPPER_TO_QUATER(dstType, srcType, rndMode, satMode, mode) \
    CastAdaptor<dstType, srcType, rndMode, mode, satMode, PPMode::ZERO>(vreg1, vreg0, preg_dst)

#define TRUNCATE_ROUND(dstType, srcType, rndMode, satMode, mode)                                    \
    Truncate<dstType, rndMode, mode>(vreg1, vreg0, preg)

#define REGISTER_CAST_LOWER_HALF(rndStr, rndMode, srcType, dstType,                                     \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_LOWER_HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)

#define REGISTER_CAST_UPPER_HALF(rndStr, rndMode, srcType, dstType,                                     \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_UPPER_HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)

#define REGISTER_CAST_LOWER_QUATER(rndStr, rndMode, srcType, dstType,                                   \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_LOWER_QUATER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)
 
#define REGISTER_CAST_S322HALF(rndStr, rndMode, srcType, dstType, srcBits, dstBits, satMode, mode) \
    CAST_S322HALF(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)
 
#define REGISTER_CAST_HALF2S4(rndStr, rndMode, srcType, dstType, srcBits, dstBits, satMode, mode) \
    CAST_HALF2S4(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)

#define REGISTER_CAST_UPPER_QUATER(rndStr, rndMode, srcType, dstType,                                   \
        srcBits, dstBits, satMode, mode)                                                                 \
    CAST_UPPER_QUATER(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)

#define REGISTER_CAST_EQUAL(rndStr, rndMode, srcType, dstType,                                          \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_EQUAL(srcType, dstType, srcBits, dstBits, rndStr, rndMode, satMode, mode)

#define REGISTER_CAST_TRUNCATE(rndStr, rndMode, srcType, dstType,                                       \
        srcBits, dstBits, satMode, mode)                                                                \
    CAST_TRUNCATE(srcType, rndStr, rndMode, mode)

#define REGISTER_CAST_LV2_NOT_SUPPORTED(rndStr, srcType, dstType)                                                \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ dstType *dst, __ubuf__ srcType *src,              \
                                                      const uint32_t calCount)                                   \
    {                                                                                                            \
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "rndStr from srcType to dstType not supported!"); }); \
    }

#define REGISTER_CAST_NOT_SUPPORTED(rndStr, srcType, dstType)                                          \
    REGISTER_CAST_LV2_NOT_SUPPORTED(rndStr, srcType, dstType)

// ROUND GROUP 0
// support CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC, CAST_ODD
#define REGISTER_CAST_ROUND_GROUP0(sizeMode, srcType, dstType, srcBits, dstBits, satMode, mode)                    \
    REGISTER_CAST_##sizeMode(CastRound, RoundMode::CAST_ROUND, srcType, dstType, srcBits, dstBits, satMode, mode); \
    REGISTER_CAST_##sizeMode(CastRint, RoundMode::CAST_RINT, srcType, dstType, srcBits, dstBits, satMode, mode);   \
    REGISTER_CAST_##sizeMode(CastFloor, RoundMode::CAST_FLOOR, srcType, dstType, srcBits, dstBits, satMode, mode); \
    REGISTER_CAST_##sizeMode(CastTrunc, RoundMode::CAST_TRUNC, srcType, dstType, srcBits, dstBits, satMode, mode); \
    REGISTER_CAST_##sizeMode(CastCeil, RoundMode::CAST_CEIL, srcType, dstType, srcBits, dstBits, satMode, mode);   \
    REGISTER_CAST_##sizeMode(CastNone, RoundMode::CAST_RINT, srcType, dstType, srcBits, dstBits, satMode, mode);   \
    REGISTER_CAST_##sizeMode(CastOdd, RoundMode::CAST_ODD, srcType, dstType, srcBits, dstBits, satMode, mode)

// ROUND GROUP 1
// support CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC
// not support CAST_ODD
#define REGISTER_CAST_ROUND_GROUP1(sizeMode, srcType, dstType, srcBits, dstBits, satMode, mode)                    \
    REGISTER_CAST_##sizeMode(CastRound, RoundMode::CAST_ROUND, srcType, dstType, srcBits, dstBits, satMode, mode); \
    REGISTER_CAST_##sizeMode(CastRint, RoundMode::CAST_RINT, srcType, dstType, srcBits, dstBits, satMode, mode);   \
    REGISTER_CAST_##sizeMode(CastFloor, RoundMode::CAST_FLOOR, srcType, dstType, srcBits, dstBits, satMode, mode); \
    REGISTER_CAST_##sizeMode(CastTrunc, RoundMode::CAST_TRUNC, srcType, dstType, srcBits, dstBits, satMode, mode); \
    REGISTER_CAST_##sizeMode(CastCeil, RoundMode::CAST_CEIL, srcType, dstType, srcBits, dstBits, satMode, mode);   \
    REGISTER_CAST_##sizeMode(CastNone, RoundMode::CAST_RINT, srcType, dstType, srcBits, dstBits, satMode, mode);   \
    REGISTER_CAST_NOT_SUPPORTED(CastOdd, srcType, dstType)

// ROUND GROUP 2
// support CAST_NONE
#define REGISTER_CAST_ROUND_GROUP2(sizeMode, srcType, dstType, srcBits, dstBits, satMode, mode)                  \
    REGISTER_CAST_NOT_SUPPORTED(CastRound, srcType, dstType);                                                    \
    REGISTER_CAST_NOT_SUPPORTED(CastRint, srcType, dstType);                                                     \
    REGISTER_CAST_NOT_SUPPORTED(CastFloor, srcType, dstType);                                                    \
    REGISTER_CAST_NOT_SUPPORTED(CastTrunc, srcType, dstType);                                                    \
    REGISTER_CAST_NOT_SUPPORTED(CastCeil, srcType, dstType);                                                     \
    REGISTER_CAST_##sizeMode(CastNone, RoundMode::CAST_NONE, srcType, dstType, srcBits, dstBits, satMode, mode); \
    REGISTER_CAST_NOT_SUPPORTED(CastOdd, srcType, dstType)

REGISTER_CAST_ROUND_GROUP1(UPPER_HALF, half, int32_t, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP0(LOWER_HALF, float, half, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(LOWER_HALF, float, int16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(LOWER_HALF, half, int8_t, 16, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(LOWER_HALF, half, uint8_t, 16, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(EQUAL, half, int16_t, 16, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(EQUAL, float, int32_t, 32, 32, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, half, float, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, uint8_t, half, 8, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int8_t, half, 8, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int16_t, float, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, uint8_t, uint16_t, 8, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int8_t, int16_t, 8, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, uint16_t, uint32_t, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int16_t, uint32_t, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_HALF, int16_t, int32_t, 16, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_QUATER, int8_t, int32_t, 8, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(UPPER_QUATER, uint8_t, uint32_t, 8, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, uint16_t, uint8_t, 16, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, int16_t, uint8_t, 16, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, uint32_t, uint16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, uint32_t, int16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, int32_t, uint16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_HALF, int32_t, int16_t, 32, 16, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_QUATER, int32_t, uint8_t, 32, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP2(LOWER_QUATER, uint32_t, uint8_t, 32, 8, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(EQUAL, int16_t, half, 16, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(EQUAL, int32_t, float, 32, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(TRUNCATE, half, half, 16, 16, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(TRUNCATE, float, float, 32, 32, SatMode::UNKNOWN, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(HALF2S4, half, int4b_t, 16, 4, SatMode::SAT, Mode::ZEROING);
REGISTER_CAST_ROUND_GROUP1(S322HALF, int32_t, half, 32, 16, SatMode::SAT, Mode::ZEROING);

#define REGISTER_DATA_TYPE_NOT_SUPPORT(rndStr)                         \
    template <typename U, typename T>               \
    __aicore__ inline void CastIntrinsicsImpl##rndStr(__ubuf__ U* dst, __ubuf__ T* src,     \
        const uint32_t calCount)                                                                        \
    {                                                                                                   \
        ASCENDC_ASSERT((false), "current convert is not supported");                                    \
    }

REGISTER_DATA_TYPE_NOT_SUPPORT(CastRint);
REGISTER_DATA_TYPE_NOT_SUPPORT(CastFloor);
REGISTER_DATA_TYPE_NOT_SUPPORT(CastTrunc);
REGISTER_DATA_TYPE_NOT_SUPPORT(CastCeil);
REGISTER_DATA_TYPE_NOT_SUPPORT(CastRound);
REGISTER_DATA_TYPE_NOT_SUPPORT(CastNone);
REGISTER_DATA_TYPE_NOT_SUPPORT(CastOdd);

// Cast::Level 2
template <typename U, typename T>
__aicore__ inline void CastImpl(__ubuf__ U *dst, __ubuf__ T *src, const RoundMode &roundMode, const uint32_t calCount)
{
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            CastIntrinsicsImplCastRint(dst, src, calCount);
            break;
        case RoundMode::CAST_FLOOR:
            CastIntrinsicsImplCastFloor(dst, src, calCount);
            break;
        case RoundMode::CAST_CEIL:
            CastIntrinsicsImplCastCeil(dst, src, calCount);
            break;
        case RoundMode::CAST_ROUND:
            CastIntrinsicsImplCastRound(dst, src, calCount);
            break;
        case RoundMode::CAST_TRUNC:
            CastIntrinsicsImplCastTrunc(dst, src, calCount);
            break;
        case RoundMode::CAST_ODD:
            CastIntrinsicsImplCastOdd(dst, src, calCount);
            break;
        case RoundMode::CAST_NONE:
            CastIntrinsicsImplCastNone(dst, src, calCount);
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

template <typename DST_TYPE, typename SRC_TYPE>
__simd_callee__ inline void GenLoadL0(MicroAPI::RegTensor<SRC_TYPE> &srcVreg, __ubuf__ SRC_TYPE *&srcAddr,
    MicroAPI::MaskReg &preg, const UnaryRepeatParams &repeatParams)
 
{
    MicroAPI::DataCopy<SRC_TYPE, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        srcVreg, srcAddr,
        static_cast<uint32_t>(repeatParams.srcBlkStride), static_cast<uint32_t>(repeatParams.srcRepStride), preg);
    if constexpr (SupportType<SRC_TYPE, int4b_t>() && sizeof(DST_TYPE) == 2) {
        MicroAPI::UnPack<uint16_t, uint8_t>(
            (MicroAPI::RegTensor<uint16_t> &)srcVreg, (MicroAPI::RegTensor<uint8_t> &)srcVreg);
        MicroAPI::UnPack<uint32_t, uint16_t>(
            (MicroAPI::RegTensor<uint32_t> &)srcVreg, (MicroAPI::RegTensor<uint16_t> &)srcVreg);
    } else if constexpr (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 2) {
        if constexpr (std::is_same_v<SRC_TYPE, int8_t>) {
            MicroAPI::UnPack<int16_t, int8_t>((MicroAPI::RegTensor<int16_t> &)srcVreg, srcVreg);
        } else {
            MicroAPI::UnPack<uint16_t, uint8_t>(
                (MicroAPI::RegTensor<uint16_t> &)srcVreg, (MicroAPI::RegTensor<uint8_t> &)srcVreg);
        }
    } else if constexpr (sizeof(SRC_TYPE) == 2 && sizeof(DST_TYPE) == 4) {
        if constexpr (std::is_same_v<SRC_TYPE, int16_t>) {
            MicroAPI::UnPack<int32_t, int16_t>((MicroAPI::RegTensor<int32_t> &)srcVreg, srcVreg);
        } else {
            MicroAPI::UnPack<uint32_t, uint16_t>(
                (MicroAPI::RegTensor<uint32_t> &)srcVreg, (MicroAPI::RegTensor<uint16_t> &)srcVreg);
        }
    } else if constexpr (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 4) {
        if constexpr (std::is_same_v<SRC_TYPE, int8_t>) {
            MicroAPI::UnPack<int16_t, int8_t>((MicroAPI::RegTensor<int16_t> &)srcVreg, srcVreg);
            MicroAPI::UnPack<int32_t, int16_t>(
                (MicroAPI::RegTensor<int32_t> &)srcVreg, (MicroAPI::RegTensor<int16_t> &)srcVreg);
        } else {
            MicroAPI::UnPack<uint16_t, uint8_t>(
                (MicroAPI::RegTensor<uint16_t> &)srcVreg, (MicroAPI::RegTensor<uint8_t> &)srcVreg);
            MicroAPI::UnPack<uint32_t, uint16_t>(
                (MicroAPI::RegTensor<uint32_t> &)srcVreg, (MicroAPI::RegTensor<uint16_t> &)srcVreg);
        }
    }
}
 
template <typename DST_TYPE, typename SRC_TYPE>
__simd_callee__ inline void GenStoreL0(__ubuf__ DST_TYPE *&dstAddr, MicroAPI::RegTensor<DST_TYPE> &dstVreg,
    MicroAPI::MaskReg &preg, const UnaryRepeatParams &repeatParams)
{
    if constexpr (SupportType<DST_TYPE, int4b_t>() && sizeof(SRC_TYPE) == 2) {
        MicroAPI::Pack<uint16_t, uint32_t>(
            (MicroAPI::RegTensor<uint16_t> &)dstVreg, (MicroAPI::RegTensor<uint32_t> &)dstVreg);
        MicroAPI::Pack<uint8_t, uint16_t>(
            (MicroAPI::RegTensor<uint8_t> &)dstVreg, (MicroAPI::RegTensor<uint16_t> &)dstVreg);
    } else if constexpr (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 2) {
        MicroAPI::Pack<uint8_t, uint16_t>(
            (MicroAPI::RegTensor<uint8_t> &)dstVreg, (MicroAPI::RegTensor<uint16_t> &)dstVreg);
    } else if constexpr (sizeof(DST_TYPE) == 2 && sizeof(SRC_TYPE) == 4) {
        MicroAPI::Pack<uint16_t, uint32_t>(
            (MicroAPI::RegTensor<uint16_t> &)dstVreg, (MicroAPI::RegTensor<uint32_t> &)dstVreg);
    } else if constexpr (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 4) {
        MicroAPI::Pack<uint16_t, uint32_t>(
            (MicroAPI::RegTensor<uint16_t> &)dstVreg, (MicroAPI::RegTensor<uint32_t> &)dstVreg);
        MicroAPI::Pack<uint8_t, uint16_t>(
            (MicroAPI::RegTensor<uint8_t> &)dstVreg, (MicroAPI::RegTensor<uint16_t> &)dstVreg);
    }
    MicroAPI::DataCopy<DST_TYPE, MicroAPI::DataCopyMode::DATA_BLOCK_COPY, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
        dstAddr, dstVreg,
        static_cast<uint32_t>(repeatParams.dstBlkStride), static_cast<uint32_t>(repeatParams.dstRepStride), preg);
}
 
template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode>
__simd_vf__ inline void CastIntrinsicsImplVF2(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const BasicAPIMaskStruct maskArrayStruct,
    uint8_t repeatTimes, const UnaryRepeatParams repeatParams)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    MicroAPI::MaskReg ldPreg;
    MicroAPI::MaskReg exPreg;
    MicroAPI::MaskReg stPreg;
    MicroAPI::MaskReg dumpPreg;
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    if constexpr (sizeof(DST_TYPE) == sizeof(SRC_TYPE)) {
        ldPreg = MicroAPI::MoveMask<SRC_TYPE>();
        exPreg = ldPreg;
        stPreg = ldPreg;
    } else if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
        ldPreg = MicroAPI::MoveMask<SRC_TYPE>();
        exPreg = ldPreg;
        MicroAPI::MaskPack(stPreg, ldPreg);
        if constexpr ((SupportType<DST_TYPE, int4b_t>() && sizeof(SRC_TYPE) == 2) ||
                      (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 4)) {
            MicroAPI::MaskPack(stPreg, stPreg);
        }
    } else if constexpr (sizeof(DST_TYPE) > sizeof(SRC_TYPE)) {
        stPreg = MicroAPI::MoveMask<DST_TYPE>();
        exPreg = stPreg;
        MicroAPI::MaskPack(ldPreg, stPreg);
        if constexpr ((SupportType<SRC_TYPE, int4b_t>() && sizeof(DST_TYPE) == 2) ||
                      (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 4)) {
            MicroAPI::MaskPack(ldPreg, ldPreg);
            if constexpr (SupportType<SRC_TYPE, int4b_t>() && sizeof(DST_TYPE) == 2) {
                MicroAPI::MaskUnPack(stPreg, ldPreg);
                MicroAPI::MaskUnPack(exPreg, stPreg);
                MicroAPI::MaskInterleave<uint16_t>(stPreg, dumpPreg, stPreg, stPreg);
            }
        }
    }
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        GenLoadL0<DST_TYPE, SRC_TYPE>(srcVreg, src, ldPreg, repeatParams);
        if constexpr (std::is_same_v<SRC_TYPE, int32_t> && std::is_same_v<DST_TYPE, half>) {
            MicroAPI::Cast<float, SRC_TYPE, castTrait>((MicroAPI::RegTensor<float> &)dstVreg, srcVreg, exPreg);
            float deqValueTmp = static_cast<float>(g_deqValue);
            MicroAPI::Muls((MicroAPI::RegTensor<float> &)dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, deqValueTmp, exPreg);
            MicroAPI::Cast<DST_TYPE, float, castTrait>(dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, exPreg);
        } else if constexpr (std::is_same_v<SRC_TYPE, float> && std::is_same_v<DST_TYPE, float>) {
            MicroAPI::Truncate<DST_TYPE, roundMode>(dstVreg, srcVreg, exPreg);
        } else {
            MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, exPreg);
        }
        GenStoreL0<DST_TYPE, SRC_TYPE>(dst, dstVreg, stPreg, repeatParams);
    }
}
 
template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__simd_vf__ inline void CastIntrinsicsImplCounterVF(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask,
    __ubuf__ uint64_t *maskBuf, uint8_t repeatTimes, const UnaryRepeatParams repeatParams)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    uint32_t sreg = static_cast<uint32_t>(mask);
    MicroAPI::MaskReg ldPreg;
    MicroAPI::MaskReg exPreg;
    MicroAPI::MaskReg stPreg;
    MicroAPI::MaskReg dumpPreg;
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    uint32_t countSreg = static_cast<uint32_t>(mask);
    if constexpr (!isSetMask) {
        // get SPR.MASK in VF
        MicroAPI::MaskReg sprLoadMaskReg = MicroAPI::MoveMask<uint16_t>();
        MicroAPI::DataCopy<uint64_t, MicroAPI::MaskDist::DIST_PACK>(maskBuf, sprLoadMaskReg);
        // insert membar(vec store operation) before load maskBuf[0](scalar load operation)
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::SCALAR_LOAD>();
        countSreg = static_cast<uint32_t>(maskBuf[0]);
    }
    uint16_t oneRepSize = GetVecLen() / sizeof(SRC_TYPE);
    if constexpr (sizeof(SRC_TYPE) < sizeof(DST_TYPE)) {
        oneRepSize = GetVecLen() / sizeof(DST_TYPE);
    }
    uint16_t newRepeatTimes = CeilDivision(countSreg, oneRepSize);
    for (uint16_t i = 0; i < newRepeatTimes; ++i) {
        if constexpr (sizeof(DST_TYPE) == sizeof(SRC_TYPE)) {
            ldPreg = MicroAPI::UpdateMask<SRC_TYPE>(countSreg);
            exPreg = ldPreg;
            stPreg = ldPreg;
        } else if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
            ldPreg = MicroAPI::UpdateMask<SRC_TYPE>(countSreg);
            exPreg = ldPreg;
            MicroAPI::MaskPack(stPreg, ldPreg);
            if constexpr ((SupportType<DST_TYPE, int4b_t>() && sizeof(SRC_TYPE) == 2) ||
                          (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 4)) {
                MicroAPI::MaskPack(stPreg, stPreg);
            }
        } else if constexpr (sizeof(DST_TYPE) > sizeof(SRC_TYPE)) {
            stPreg = MicroAPI::UpdateMask<DST_TYPE>(countSreg);
            exPreg = stPreg;
            MicroAPI::MaskPack(ldPreg, stPreg);
            if constexpr ((SupportType<SRC_TYPE, int4b_t>() && sizeof(DST_TYPE) == 2) ||
                          (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 4)) {
                MicroAPI::MaskPack(ldPreg, ldPreg);
                if constexpr (SupportType<SRC_TYPE, int4b_t>() && sizeof(DST_TYPE) == 2) {
                    MicroAPI::MaskUnPack(stPreg, ldPreg);
                    MicroAPI::MaskUnPack(exPreg, stPreg);
                    MicroAPI::MaskInterleave<uint16_t>(stPreg, dumpPreg, stPreg, stPreg);
                }
            }
        }
        GenLoadL0<DST_TYPE, SRC_TYPE>(srcVreg, src, ldPreg, repeatParams);
        if constexpr (std::is_same_v<SRC_TYPE, int32_t> && std::is_same_v<DST_TYPE, half>) {
            MicroAPI::Cast<float, SRC_TYPE, castTrait>((MicroAPI::RegTensor<float> &)dstVreg, srcVreg, exPreg);
            float deqValueTmp = static_cast<float>(g_deqValue);
            MicroAPI::Muls((MicroAPI::RegTensor<float> &)dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, deqValueTmp, exPreg);
            MicroAPI::Cast<DST_TYPE, float, castTrait>(dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, exPreg);
        } else if constexpr (std::is_same_v<SRC_TYPE, float> && std::is_same_v<DST_TYPE, float>) {
            MicroAPI::Truncate<DST_TYPE, roundMode>(dstVreg, srcVreg, exPreg);
        } else {
            MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, exPreg);
        }
        GenStoreL0<DST_TYPE, SRC_TYPE>(dst, dstVreg, stPreg, repeatParams);
    }
}
 
template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__aicore__ inline void CastIntrinsicsImpl(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask[],
    uint8_t repeatTimes, const UnaryRepeatParams &repeatParams)
{
    bool isCounterMode = Internal::IsCounterMode();
    uint16_t maskArraySize = (mask == nullptr) ? 0 : MASK_ARRAY_SIZE;
    BasicAPIMaskStruct maskArrayStruct;
    for (uint16_t i = 0; i < maskArraySize; i++) {
        maskArrayStruct.maskArray[i] = mask[i];
    }
    if (isCounterMode) {
        __ubuf__ uint64_t *maskBuf = nullptr;
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        }
        CastIntrinsicsImplCounterVF<DST_TYPE, SRC_TYPE, roundMode, isSetMask>(
                dst, src, mask[0], maskBuf, repeatTimes, repeatParams);
    } else {
            if constexpr (isSetMask) {
                if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
                    SetVectorMask<SRC_TYPE>(mask[1], mask[0]);
                } else {
                    SetVectorMask<DST_TYPE>(mask[1], mask[0]);
                }
            }
            CastIntrinsicsImplVF2<DST_TYPE, SRC_TYPE, roundMode>(
                dst, src, maskArrayStruct, repeatTimes, repeatParams);
    }
}
 
// Cast::Level 0 - mask bit mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const RoundMode &roundMode,
    const uint64_t mask[], uint8_t repeatTimes, const UnaryRepeatParams &repeatParams)
{
    constexpr bool cast_round_all = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>, Tuple<int32_t, float>,
                                          Tuple<int16_t, float>, Tuple<int32_t, half>, Tuple<int16_t, half>, Tuple<int8_t, half>,
                                          Tuple<uint8_t, half>, Tuple<int4b_t, half>, Tuple<half, int16_t>, Tuple<float, int32_t>, Tuple<half, int32_t>>();
 
    constexpr bool cast_none = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<float, half>,
                                        Tuple<half, int4b_t>, Tuple<half, uint8_t>, Tuple<uint16_t, uint8_t>, Tuple<uint32_t, uint8_t>,
                                        Tuple<half, int8_t>, Tuple<int16_t, int8_t>, Tuple<int32_t, int8_t>, Tuple<uint8_t, uint16_t>,
                                        Tuple<uint32_t, uint16_t>, Tuple<float, int16_t>, Tuple<uint8_t, int16_t>, Tuple<uint32_t, int16_t>,
                                        Tuple<int32_t, int16_t>, Tuple<uint8_t, uint32_t>, Tuple<uint16_t, uint32_t>, Tuple<int16_t, uint32_t>,
                                        Tuple<int16_t, int32_t>, Tuple<uint8_t, int32_t>, Tuple<uint16_t, int32_t>>();
 
    constexpr bool using_cast_rint = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<int8_t, half>, Tuple<uint8_t, half>, Tuple<int4b_t, half>,
                                            Tuple<half, float>, Tuple<half, int16_t>, Tuple<float, int32_t>>();
    constexpr bool cast_odd = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>>();
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast rint"); });
            }
            break;
        case RoundMode::CAST_FLOOR:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_FLOOR, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast floor"); });
            }
            break;
        case RoundMode::CAST_CEIL:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_CEIL, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast ceil"); });
            }
            break;
        case RoundMode::CAST_ROUND:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ROUND, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast round"); });
            }
            break;
        case RoundMode::CAST_TRUNC:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_TRUNC, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast trunc"); });
            }
            break;
        case RoundMode::CAST_ODD:
            if constexpr (cast_odd) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ODD, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast odd"); });
            }
            break;
        case RoundMode::CAST_NONE:
            if constexpr (cast_none) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_NONE, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else if constexpr (using_cast_rint) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast none"); });
            }
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__simd_vf__ inline void CastIntrinsicsImplVF1(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams repeatParams)
{
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, roundMode};
    uint32_t sreg = static_cast<uint32_t>(mask);
    MicroAPI::MaskReg ldPreg;
    MicroAPI::MaskReg exPreg;
    MicroAPI::MaskReg stPreg;
    MicroAPI::MaskReg dumpPreg;
    MicroAPI::RegTensor<SRC_TYPE> srcVreg;
    MicroAPI::RegTensor<DST_TYPE> dstVreg;
    if constexpr (sizeof(DST_TYPE) == sizeof(SRC_TYPE)) {
        if constexpr (isSetMask) {
            ldPreg = MicroAPI::UpdateMask<SRC_TYPE>(sreg);
        } else {
            ldPreg = MicroAPI::MoveMask<SRC_TYPE>();
        }
        exPreg = ldPreg;
        stPreg = ldPreg;
    } else if constexpr (sizeof(DST_TYPE) < sizeof(SRC_TYPE)) {
        if constexpr (isSetMask) {
            ldPreg = MicroAPI::UpdateMask<SRC_TYPE>(sreg);
        } else {
            ldPreg = MicroAPI::MoveMask<SRC_TYPE>();
        }
        exPreg = ldPreg;
        MicroAPI::MaskPack(stPreg, ldPreg);
        if constexpr ((SupportType<DST_TYPE, int4b_t>() && sizeof(SRC_TYPE) == 2) ||
                      (sizeof(DST_TYPE) == 1 && sizeof(SRC_TYPE) == 4)) {
            MicroAPI::MaskPack(stPreg, stPreg);
        }
    } else if constexpr (sizeof(DST_TYPE) > sizeof(SRC_TYPE)) {
        if constexpr (isSetMask) {
            stPreg = MicroAPI::UpdateMask<DST_TYPE>(sreg);
        } else {
            stPreg = MicroAPI::MoveMask<DST_TYPE>();
        }
        exPreg = stPreg;
        MicroAPI::MaskPack(ldPreg, stPreg);
        if constexpr ((SupportType<SRC_TYPE, int4b_t>() && sizeof(DST_TYPE) == 2) ||
                      (sizeof(SRC_TYPE) == 1 && sizeof(DST_TYPE) == 4)) {
            MicroAPI::MaskPack(ldPreg, ldPreg);
            if constexpr (SupportType<SRC_TYPE, int4b_t>() && sizeof(DST_TYPE) == 2) {
                MicroAPI::MaskUnPack(stPreg, ldPreg);
                MicroAPI::MaskUnPack(exPreg, stPreg);
                MicroAPI::MaskInterleave<uint16_t>(stPreg, dumpPreg, stPreg, stPreg);
            }
        }
    }
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        GenLoadL0<DST_TYPE, SRC_TYPE>(srcVreg, src, ldPreg, repeatParams);
        if constexpr (std::is_same_v<SRC_TYPE, int32_t> && std::is_same_v<DST_TYPE, half>) {
            MicroAPI::Cast<float, SRC_TYPE, castTrait>((MicroAPI::RegTensor<float> &)dstVreg, srcVreg, exPreg);
            float deqValueTmp = static_cast<float>(g_deqValue);
            MicroAPI::Muls((MicroAPI::RegTensor<float> &)dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, deqValueTmp, exPreg);
            MicroAPI::Cast<DST_TYPE, float, castTrait>(dstVreg, (MicroAPI::RegTensor<float> &)dstVreg, exPreg);
        } else if constexpr (std::is_same_v<SRC_TYPE, float> && std::is_same_v<DST_TYPE, float>) {
            MicroAPI::Truncate<DST_TYPE, roundMode>(dstVreg, srcVreg, exPreg);
        } else {
            MicroAPI::Cast<DST_TYPE, SRC_TYPE, castTrait>(dstVreg, srcVreg, exPreg);
        }
        GenStoreL0<DST_TYPE, SRC_TYPE>(dst, dstVreg, stPreg, repeatParams);
    }
}

template <typename DST_TYPE, typename SRC_TYPE, RoundMode roundMode, bool isSetMask>
__aicore__ inline void CastIntrinsicsImpl(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const uint64_t mask,
    uint8_t repeatTimes, const UnaryRepeatParams &repeatParams)
{
    bool isCounterMode = Internal::IsCounterMode();
    if (isCounterMode) {
        __ubuf__ uint64_t *maskBuf = nullptr;
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2);
        }
        CastIntrinsicsImplCounterVF<DST_TYPE, SRC_TYPE, roundMode, isSetMask>(
                dst, src, mask, maskBuf, repeatTimes, repeatParams);
    } else {
        CastIntrinsicsImplVF1<DST_TYPE, SRC_TYPE, roundMode, isSetMask>(
                dst, src, mask, repeatTimes, repeatParams);
    }
}
 
// Cast::Level 0 - mask count mode
template <typename DST_TYPE, typename SRC_TYPE, bool isSetMask = true>
__aicore__ inline void CastImpl(__ubuf__ DST_TYPE *dst, __ubuf__ SRC_TYPE *src, const RoundMode &roundMode,
    const uint64_t mask, uint8_t repeatTimes, const UnaryRepeatParams &repeatParams)
{
    constexpr bool cast_round_all = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>, Tuple<int32_t, float>,
                                          Tuple<int16_t, float>, Tuple<int32_t, half>, Tuple<int16_t, half>, Tuple<int8_t, half>,
                                          Tuple<uint8_t, half>, Tuple<int4b_t, half>, Tuple<half, int16_t>, Tuple<float, int32_t>, Tuple<half, int32_t>>();
 
    constexpr bool cast_none = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<float, half>,
                                        Tuple<half, int4b_t>, Tuple<half, uint8_t>, Tuple<uint16_t, uint8_t>, Tuple<uint32_t, uint8_t>,
                                        Tuple<half, int8_t>, Tuple<int16_t, int8_t>, Tuple<int32_t, int8_t>, Tuple<uint8_t, uint16_t>,
                                        Tuple<uint32_t, uint16_t>, Tuple<float, int16_t>, Tuple<uint8_t, int16_t>, Tuple<uint32_t, int16_t>,
                                        Tuple<int32_t, int16_t>, Tuple<uint8_t, uint32_t>, Tuple<uint16_t, uint32_t>, Tuple<int16_t, uint32_t>,
                                        Tuple<int16_t, int32_t>, Tuple<uint8_t, int32_t>, Tuple<uint16_t, int32_t>>();
 
    constexpr bool using_cast_rint = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<int8_t, half>, Tuple<uint8_t, half>, Tuple<int4b_t, half>,
                                            Tuple<half, float>, Tuple<half, int16_t>, Tuple<float, int32_t>>();
 
    constexpr bool cast_odd = SupportType<Tuple<DST_TYPE, SRC_TYPE>, Tuple<half, float>>();
    switch (roundMode) {
        case RoundMode::CAST_RINT:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast rint"); });
            }
            break;
        case RoundMode::CAST_FLOOR:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_FLOOR, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast floor"); });
            }
            break;
        case RoundMode::CAST_CEIL:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_CEIL, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast ceil"); });
            }
            break;
        case RoundMode::CAST_ROUND:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ROUND, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast round"); });
            }
            break;
        case RoundMode::CAST_TRUNC:
            if constexpr (cast_round_all) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_TRUNC, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast trunc"); });
            }
            break;
        case RoundMode::CAST_ODD:
            if constexpr (cast_odd) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_ODD, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast odd"); });
            }
            break;
        case RoundMode::CAST_NONE:
            if constexpr (cast_none) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_NONE, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else if constexpr (using_cast_rint) {
                CastIntrinsicsImpl<DST_TYPE, SRC_TYPE, RoundMode::CAST_RINT, isSetMask>(dst, src, mask, repeatTimes, repeatParams);
            } else {
                ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "illegal type for cast none"); });
            }
            break;
        default:
            ASCENDC_ASSERT(
                (false), { KERNEL_LOG(KERNEL_ERROR, "illegal input cast mode %d", static_cast<int32_t>(roundMode)); });
            break;
    }
}

template <typename U, typename T, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U *dst, __ubuf__ T *src, const uint32_t calCount)
{
    ASCENDC_ASSERT((false), "CastDeq is not supported");
}

template <typename U, typename T, bool isSetMask = true, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U *dst, __ubuf__ T *src, const uint64_t mask[2], uint8_t repeatTimes,
                                   const UnaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((false), "CastDeq is not supported");
}

template <typename U, typename T, bool isSetMask = true, bool isVecDeq, bool halfBlock>
__aicore__ inline void CastDeqImpl(__ubuf__ U *dst, __ubuf__ T *src, const int32_t mask, uint8_t repeatTimes,
                                   const UnaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((false), "CastDeq is not supported");
}

// AddReluCast::Level 0 - mask count mode
template <typename dstType, typename srcType, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ dstType *dst, __ubuf__ srcType *src0, __ubuf__ srcType *src1,
                                       const uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((false), "AddReluCast is not supported");
}

// AddReluCast::Level 0 - mask bit mode
template <typename dstType, typename srcType, bool isSetMask = true>
__aicore__ inline void AddReluCastImpl(__ubuf__ dstType *dst, __ubuf__ srcType *src0, __ubuf__ srcType *src1,
                                       const uint64_t mask[2], uint8_t repeatTimes,
                                       const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((false), "AddReluCast is not supported");
}

// AddReluCast::Level 2
template <typename dstType, typename srcType>
__aicore__ inline void AddReluCastImpl(__ubuf__ dstType *dst, __ubuf__ srcType *src0, __ubuf__ srcType *src1,
                                       const uint32_t calCount)
{
    ASCENDC_ASSERT((false), "AddReluCast is not supported");
}

// SubReluCast::Level 0 - mask count mode
template <typename dstType, typename srcType, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ dstType *dst, __ubuf__ srcType *src0, __ubuf__ srcType *src1,
                                       const uint64_t mask, uint8_t repeatTimes, const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((false), "SubReluCast is not supported");
}

// SubReluCast::Level 0 - mask bit mode
template <typename dstType, typename srcType, bool isSetMask = true>
__aicore__ inline void SubReluCastImpl(__ubuf__ dstType *dst, __ubuf__ srcType *src0, __ubuf__ srcType *src1,
                                       const uint64_t mask[2], uint8_t repeatTimes,
                                       const BinaryRepeatParams &repeatParams)
{
    ASCENDC_ASSERT((false), "SubReluCast is not supported");
}

// SubReluCast::Level 2
template <typename dstType, typename srcType>
__aicore__ inline void SubReluCastImpl(__ubuf__ dstType *dst, __ubuf__ srcType *src0, __ubuf__ srcType *src1,
                                       const uint32_t calCount)
{
    ASCENDC_ASSERT((false), "SubReluCast is not supported");
}

//  castDequanValue bit arrange
//  =========================================================================
//  | unused 17bit | 1bit signMode | 9bit offset | unused 5bit | 32bit scale|
//  =========================================================================
__aicore__ inline uint64_t MakeDeqScaleConfig(float scale, int16_t offset, bool signMode)
{
    constexpr uint64_t signModeBit = 46;
    constexpr uint64_t offsetMask = 0x1ff;
    constexpr uint64_t offsetBit = 37;
    uint64_t config = ((static_cast<uint64_t>(signMode) << signModeBit) | ((offset & offsetMask) << offsetBit) |
                       *(reinterpret_cast<uint32_t *>(&scale)));
    return config;
}
 
__aicore__ inline void SetDeqScaleImpl(float scale, int16_t offset, bool signMode)
{
    Internal::g_deqScale = MakeDeqScaleConfig(scale, offset, signMode);
}

template <typename T>
__aicore__ inline void SetDeqScaleImpl(const LocalTensor<T> &vdeqTensor, const VdeqInfo &vdeqInfo)
{
    for (uint8_t i = 0; i < VDEQ_TENSOR_SIZE; ++i) {
        float scale = vdeqInfo.vdeqScale[i];
        int16_t offset = vdeqInfo.vdeqOffset[i];
        bool signMode = vdeqInfo.vdeqSignMode[i];
        vdeqTensor.SetValue(i, static_cast<T>(MakeDeqScaleConfig(scale, offset, signMode)));
    }
    Internal::g_deqScale = reinterpret_cast<uint64_t>(vdeqTensor.GetPhyAddr());
}

template <typename T>
__aicore__ inline void SetDeqScaleImpl(T config)
{
    g_deqValue = config;
}

// Truncate::Level2
template <typename T, RoundMode roundMode>
__simd_vf__ inline void TruncateImpl(__ubuf__ T *dst, __ubuf__ T *src, const uint32_t calCount)
{
    static_assert(SupportType<T, half, float, bfloat16_t>(), "Failed to check dtype in Truncate, current api "
        "support dtype is src and dst both: half, float, bfloat16_t.");
    static_assert(SupportEnum<roundMode, RoundMode::CAST_RINT, RoundMode::CAST_FLOOR, RoundMode::CAST_CEIL,
        RoundMode::CAST_ROUND, RoundMode::CAST_TRUNC>(), "Failed to check dtype in Truncate, "
        "current api support roundMode is CAST_RINT, CAST_FLOOR, CAST_CEIL, CAST_ROUND, CAST_TRUNC.");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(T));
    const uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::RegTensor<T> vDstReg;
    MicroAPI::RegTensor<T> vSrcReg;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTimes; ++i) {
        mask = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(vSrcReg, src + i * sregLower);
        MicroAPI::Truncate<T, roundMode>(vDstReg, vSrcReg, mask);
        MicroAPI::DataCopy(dst + i * sregLower, vDstReg, mask);
    }
}

} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VCONV_IMPL_H
