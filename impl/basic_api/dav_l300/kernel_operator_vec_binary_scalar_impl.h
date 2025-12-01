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
 * \file kernel_operator_vec_binary_scalar_impl.h
 * \brief AscendC l300 support vector binary scalar api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"
#include "kernel_struct_unary.h"

namespace AscendC {
#define BIT_BY_BIT_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTimes, repeatParams)                 \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<DATA_TYPE> srcReg, dstReg;                                                                    \
            MaskReg preg = MovePredicate<DATA_TYPE>();                                                              \
            uint32_t srcBlkStride = static_cast<uint32_t>(repeatParams.srcBlkStride);                                          \
            uint32_t dstBlkStride = static_cast<uint32_t>(repeatParams.dstBlkStride);                                          \
            uint32_t srcRepStride = static_cast<uint32_t>(repeatParams.srcRepStride);                                          \
            uint32_t dstRepStride = static_cast<uint32_t>(repeatParams.dstRepStride);                                          \
            for (uint16_t i = 0; i < (uint16_t)(repeatTimes); ++i) {                                                \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(srcReg, src, srcBlkStride, srcRepStride, preg);  \
                OP_NAME(dstReg, srcReg, scalarValue, preg);                                                         \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, dstReg, dstBlkStride, dstRepStride, preg);  \
            }                                                                                                       \
        }                                                                                                           \

#define CONTINUOUS_MODE_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTimes, repeatParams)            \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<DATA_TYPE> srcReg, dstReg;                                                                    \
            uint32_t sreg = (uint32_t)(mask);                                                                       \
            MaskReg preg = CreatePredicate<DATA_TYPE>(sreg);                                                        \
            uint32_t srcBlkStride = static_cast<uint32_t>(repeatParams.srcBlkStride);                                          \
            uint32_t dstBlkStride = static_cast<uint32_t>(repeatParams.dstBlkStride);                                          \
            uint32_t srcRepStride = static_cast<uint32_t>(repeatParams.srcRepStride);                                          \
            uint32_t dstRepStride = static_cast<uint32_t>(repeatParams.dstRepStride);                                          \
            for (uint16_t i = 0; i < (uint16_t)(repeatTimes); ++i) {                                                \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(srcReg, src, srcBlkStride, srcRepStride, preg);  \
                OP_NAME(dstReg, srcReg, scalarValue, preg);                                                         \
                DataCopy<DATA_TYPE, PostLiteral::POST_MODE_UPDATE>(dst, dstReg, dstBlkStride, dstRepStride, preg);  \
            }                                                                                                       \
        }

// for Level 0 bit-by-bit mode binary scalar op
#define BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(FUNC_NAME)                                         \
    template <typename T, bool isSetMask = true>                                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],       \
        const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)                                           \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }

#define BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(FUNC_NAME)                              \
    template <typename T, bool isSetMask = true>                                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask[2],       \
        const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn = false)                     \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }


#define BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                 \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
        const uint64_t mask[2], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)                   \
    {                                                                                                               \
        if constexpr (isSetMask) {                                                                                  \
            SetVectorMask<DATA_TYPE>(mask[1], mask[0]);                                                             \
        }                                                                                                           \
        BIT_BY_BIT_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTimes, repeatParams);                \
    }

#define BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_BIT_BY_BIT_MODE_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                      \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
        const uint64_t mask[2], const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)     \
    {                                                                                                               \
        if constexpr (isSetMask) {                                                                                  \
            SetVectorMask<DATA_TYPE>(mask[1], mask[0]);                                                             \
        }                                                                                                           \
        BIT_BY_BIT_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTimes, repeatParams);                \
    }

// for Level 0 continuous mode binary scalar op
#define BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(FUNC_NAME)                                         \
    template <typename T, bool isSetMask = true>                                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,          \
        const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)                                           \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }

#define BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(FUNC_NAME)                              \
    template <typename T, bool isSetMask = true>                                                                    \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const uint64_t mask,          \
        const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn = false)                     \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }

#define BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                 \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
        const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams)                      \
    {                                                                                                               \
        CONTINUOUS_MODE_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTimes, repeatParams)            \
    }

#define BINARY_SCALAR_OP_SHIFTRIGHT_LEVEL0_CONTINUOUS_MODE_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                      \
    template <typename T = DATA_TYPE, bool isSetMask = true>                                                        \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
        const uint64_t mask, const uint8_t repeatTimes, const UnaryRepeatParams& repeatParams, bool roundEn)        \
    {                                                                                                               \
        CONTINUOUS_MODE_FUNC(OP_NAME, DATA_TYPE, dst, src, scalarValue, mask, repeatTimes, repeatParams)            \
    }

// for Level 2 binary scalar op
#define BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(FUNC_NAME)                                                         \
    template <typename T, bool isSetMask>                                                                           \
    __aicore__ inline void FUNC_NAME(__ubuf__ T* dst, __ubuf__ T* src, T scalarValue, const int32_t& calCount)      \
    {                                                                                                               \
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "current data type is not supported!"); });                \
    }


#define BINARY_SCALAR_OP_LEVEL2_IMPL(FUNC_NAME, OP_NAME, DATA_TYPE)                                                 \
    template <typename T = DATA_TYPE, bool isSetMask>                                                               \
    __aicore__ inline void FUNC_NAME(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, DATA_TYPE scalarValue,       \
                                     const int32_t& calCount)                                                       \
    {                                                                                                               \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<DATA_TYPE> srcReg, dstReg;                                                                    \
            MaskReg preg;                                                                                           \
            uint32_t sreg = (uint32_t)calCount;                                                                     \
            constexpr uint32_t sregLower = (uint32_t)(VECTOR_REG_WIDTH / sizeof(DATA_TYPE));                        \
            uint16_t repeatTimes = CeilDivision(calCount, sregLower);                                               \
            for (uint16_t i = 0; i < repeatTimes; ++i) {                                                            \
                preg = CreatePredicate<DATA_TYPE>(sreg);                                                            \
                DataCopy(srcReg, src, i * sregLower);                                                               \
                OP_NAME(dstReg, srcReg, scalarValue, preg);                                                         \
                DataCopy(dst, dstReg, i * sregLower, preg);                                                         \
            }                                                                                                       \
        }                                                                                                           \
    }
namespace Internal {
template <auto func, typename T, typename RegType>
__simd_vf__ inline void VecBinaryScalarLevel2VFImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue,
    const uint32_t calCount)
{
    RegType srcReg;
    RegType dstReg;
    uint32_t count = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg mask;
    constexpr uint32_t repeatStride = static_cast<uint32_t>(GetVecLen() / sizeof(T) * RegType::trait.REG_NUM);
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, repeatStride));
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<T, RegType::trait>(count);
        MicroAPI::DataCopy(srcReg, src + i * repeatStride);
        func(dstReg, srcReg, scalarValue, mask);
        MicroAPI::DataCopy(dst + i * repeatStride, dstReg, mask);
    }
}

template <auto func, typename T>
__aicore__ inline void VecBinaryScalarLevel2ImplTemplate(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue,
    const uint32_t calCount)
{
    if constexpr (SupportBytes<T, 8>()) {
        VecBinaryScalarLevel2VFImpl<func, T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>(dst, src,
            scalarValue, calCount);
    } else {
        VecBinaryScalarLevel2VFImpl<func, T, MicroAPI::RegTensor<T>>(dst, src, scalarValue, calCount);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, bool isNormalMode, typename T>
__simd_vf__ inline void VecBinaryScalarLevel0VFImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue,
    const BasicAPIMaskStruct maskArrayStruct, const uint64_t maskCount, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams, __ubuf__ uint64_t *maskBuf)
{
    uint32_t count = VecMicroGetCount<isSetMask, isNormalMode, isMaskBitMode>(maskArrayStruct.maskArray, maskCount, maskBuf);
    uint16_t newRepeatTimes = 0;
    newRepeatTimes = VecMicroGetRepeatTimes<T, isNormalMode>(count, repeatTime);
    MicroAPI::MaskReg maskReg;
    if constexpr (isNormalMode) {
        maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
    }
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);
    for (uint16_t index = 0; index < newRepeatTimes; ++index) {
        if constexpr (!isNormalMode) {
            maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
        }
        MicroAPI::RegTensor<T> dstVreg;
        MicroAPI::RegTensor<T> srcVreg;
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(srcVreg,
            src + index * repeatParams.srcRepStride * ElePerBlkT, repeatParams.srcBlkStride, maskReg);
        func(dstVreg, srcVreg, scalarValue, maskReg);
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + index * repeatParams.dstRepStride * ElePerBlkT, dstVreg, repeatParams.dstBlkStride, maskReg);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, typename T>
__aicore__ inline void VecBinaryScalarLevel0Template(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue,
    const uint64_t maskArray[], const uint64_t maskCount, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    if constexpr (isMaskBitMode) {
        ASCENDC_ASSERT(maskCount == 0, "maskCount must be 0 when isMaskBitMode is true.");
    } else {
        ASCENDC_ASSERT(maskArray == nullptr, "maskArray must be nullptr when isMaskBitMode is false.");
    }
    __ubuf__ uint64_t *maskBuf = nullptr;
    uint16_t maskArraySize = (maskArray == nullptr) ? 0 : MASK_ARRAY_SIZE;
    BasicAPIMaskStruct maskArrayStruct;
    for (uint16_t i = 0; i < maskArraySize; i++) {
        maskArrayStruct.maskArray[i] = maskArray[i];
    }
    if (Internal::IsCounterMode()) {
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2); // maskReg 256bit PK-> 128bit
        }
        VecBinaryScalarLevel0VFImpl<func, isSetMask, isMaskBitMode, false, T>(dst, src, scalarValue, maskArrayStruct,
            maskCount, repeatTime, repeatParams, maskBuf);
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        };
    } else {
        if constexpr (isMaskBitMode) {
            if constexpr (SupportBytes<T, 1>()) {
                ASCENDC_ASSERT(isSetMask, "mask must be set when sizeof(T) is 1.");
                auto eventIDV2S = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(eventIDV2S);
                WaitFlag<HardEvent::V_S>(eventIDV2S);
                maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 4);
                maskBuf[0] = maskArray[0];
                maskBuf[1] = maskArray[1];
                maskBuf[2] = maskArray[2];
                maskBuf[3] = maskArray[3];
                auto eventIDS2V = GetTPipePtr()->FetchEventID(HardEvent::S_V);
                SetFlag<HardEvent::S_V>(eventIDS2V);
                WaitFlag<HardEvent::S_V>(eventIDS2V);
            } else if constexpr (isSetMask) {
                SetVectorMask<T>(maskArray[1], maskArray[0]); // set mask to SPR.MASK, movp in VF
            }
        }
        // when isSetMask is false, normal mode, maskBuf = nullptr, not support B8
        VecBinaryScalarLevel0VFImpl<func, isSetMask, isMaskBitMode, true, T>(dst, src, scalarValue, maskArrayStruct,
            maskCount, repeatTime, repeatParams, maskBuf);
        if constexpr (isMaskBitMode && SupportBytes<T, 1>()) {
            AscendC::AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        }
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, bool isNormalMode, typename T, MicroAPI::LoadDist pattern, uint8_t scalarIdx>
__simd_vf__ inline void VecBinaryScalarLevel0VFImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const BasicAPIMaskStruct maskArrayStruct, const uint64_t maskCount, const uint8_t repeatTime,
    const UnaryRepeatParams repeatParams, __ubuf__ uint64_t *maskBuf)
{
    uint32_t count = VecMicroGetCount<isSetMask, isNormalMode, isMaskBitMode>(maskArrayStruct.maskArray, maskCount, maskBuf);
    uint16_t newRepeatTimes = 0;
    newRepeatTimes = VecMicroGetRepeatTimes<T, isNormalMode>(count, repeatTime);
    MicroAPI::MaskReg maskReg;
    MicroAPI::RegTensor<T> vSrcReg0;
    MicroAPI::RegTensor<T> vSrcReg1;
    MicroAPI::RegTensor<T> vDstReg0;
    if constexpr (isNormalMode) {
        maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
    }
    constexpr uint8_t ElePerBlkT = GetDataBlockSizeInBytes() / sizeof(T);
    for (uint16_t index = 0; index < newRepeatTimes; ++index) {
        if constexpr (!isNormalMode) {
            maskReg = VecMicroGetMaskReg<T, isSetMask, isNormalMode, isMaskBitMode>(maskBuf, count);
        }
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        if constexpr (scalarIdx == 0) {
            MicroAPI::DataCopy<T, pattern>(vSrcReg0, src0);
            MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vSrcReg1,
                src1 + index * repeatParams.srcRepStride * ElePerBlkT, repeatParams.srcBlkStride, maskReg);
        } else if constexpr (scalarIdx == 1) {
            MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(vSrcReg0,
                src0 + index * repeatParams.srcRepStride * ElePerBlkT, repeatParams.srcBlkStride, maskReg);
            MicroAPI::DataCopy<T, pattern>(vSrcReg1, src1);
        }
        func(vDstReg0, vSrcReg0, vSrcReg1, maskReg);
        MicroAPI::DataCopy<T, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            dst + index * repeatParams.dstRepStride * ElePerBlkT, vDstReg0, repeatParams.dstBlkStride, maskReg);
    }
}

template <auto func, bool isSetMask, bool isMaskBitMode, typename T, MicroAPI::LoadDist pattern, uint8_t scalarIdx>
__aicore__ inline void VecBinaryScalarLevel0Template(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const uint64_t maskArray[], const uint64_t maskCount, const uint8_t repeatTime,
    const UnaryRepeatParams &repeatParams)
{
    if constexpr (isMaskBitMode) {
        ASCENDC_ASSERT(maskCount == 0, "maskCount must be 0 when isMaskBitMode is true.");
    } else {
        ASCENDC_ASSERT(maskArray == nullptr, "maskArray must be nullptr when isMaskBitMode is false.");
    }
    __ubuf__ uint64_t *maskBuf = nullptr;
    uint16_t maskArraySize = (maskArray == nullptr) ? 0 : MASK_ARRAY_SIZE;
    BasicAPIMaskStruct maskArrayStruct;
    for (uint16_t i = 0; i < maskArraySize; i++) {
        maskArrayStruct.maskArray[i] = maskArray[i];
    }
    if (Internal::IsCounterMode()) {
        if constexpr (!isSetMask) {
            maskBuf = AscendCUtils::GetTemporaryBufferAddr<uint64_t>(TMP_UB_OFFSET, 2); // maskReg 256bit PK-> 128bit
        }
        VecBinaryScalarLevel0VFImpl<func, isSetMask, isMaskBitMode, false, T, pattern, scalarIdx>(dst, src0, src1, maskArrayStruct,
            maskCount, repeatTime, repeatParams, maskBuf);
        if constexpr (!isSetMask) {
            AscendCUtils::FreeTemporaryBuffer<uint64_t>(maskBuf);
        };
    } else {
        if constexpr (isMaskBitMode && isSetMask) {
            SetVectorMask<T>(maskArray[1], maskArray[0]); // set mask to SPR.MASK, movp in VF
        }
        // when isSetMask is false, normal mode, maskBuf = nullptr, not support B8
        VecBinaryScalarLevel0VFImpl<func, isSetMask, isMaskBitMode, true, T, pattern, scalarIdx>(dst, src0, src1, maskArrayStruct,
            maskCount, repeatTime, repeatParams, maskBuf);
    }
}
} // namespace Internal
/* **************************************************************************************************
 * Adds                                                                                             *
 * **************************************************************************************************/
// Adds::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int16_t, int32_t>()), "Adds not support current datatype!");
    constexpr auto func = MicroAPI::Adds<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int16_t, int32_t>()), "Adds not support current datatype!");
    constexpr auto func = MicroAPI::Adds<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Adds::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(AddsImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, int32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(AddsImpl, Adds, float)

/* **************************************************************************************************
 * Muls                                                                                             *
 * **************************************************************************************************/
// Muls::Level 0
namespace MicroAPIMuls {
template <typename T, typename RegT>
__simd_callee__ inline void Muls(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    if constexpr (SupportType<T, bfloat16_t>()) {
        MicroAPI::Duplicate(dstReg, scalarValue, mask);
        MicroAPI::Mul(dstReg, srcReg, dstReg, mask);
    } else {
        MicroAPI::Muls(dstReg, srcReg, scalarValue, mask);
    }
}
} // namespace MicroAPIMuls
template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int16_t, int32_t>()), "Muls not support current datatype!");
    constexpr auto func = MicroAPIMuls::Muls<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int16_t, int32_t>()), "Muls not support current datatype!");
    constexpr auto func = MicroAPIMuls::Muls<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Muls::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(MulsImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, int32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(MulsImpl, Muls, float)

/* **************************************************************************************************
 * Maxs                                                                                             *
 * **************************************************************************************************/
// Maxs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int16_t, int32_t>()), "Maxs not support current datatype!");
    constexpr auto func = MicroAPI::Maxs<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int16_t, int32_t>()), "Maxs not support current datatype!");
    constexpr auto func = MicroAPI::Maxs<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Maxs::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(MaxsImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, int32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(MaxsImpl, Maxs, float)

/* **************************************************************************************************
 * Mins                                                                                             *
 * **************************************************************************************************/
// Mins::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int16_t, int32_t>()), "Mins not support current datatype!");
    constexpr auto func = MicroAPI::Mins<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float, int16_t, int32_t>()), "Mins not support current datatype!");
    constexpr auto func = MicroAPI::Mins<T, T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Mins::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(MinsImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, int32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(MinsImpl, Mins, float)

/* **************************************************************************************************
 * LeakyRelu                                                                                        *
 * **************************************************************************************************/
// LeakyRelu::Level 0
// bit-by-bit mode
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL_NOT_SUPPORT(LeakyReluImpl)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(LeakyReluImpl, LeakyRelu, half)
BINARY_SCALAR_OP_LEVEL0_BIT_BY_BIT_MODE_IMPL(LeakyReluImpl, LeakyRelu, float)
// continuous mode
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL_NOT_SUPPORT(LeakyReluImpl)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(LeakyReluImpl, LeakyRelu, half)
BINARY_SCALAR_OP_LEVEL0_CONTINUOUS_MODE_IMPL(LeakyReluImpl, LeakyRelu, float)

// LeakyRelu::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(LeakyReluImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(LeakyReluImpl, LeakyRelu, half)
BINARY_SCALAR_OP_LEVEL2_IMPL(LeakyReluImpl, LeakyRelu, float)

/* **************************************************************************************************
 * ShiftLeft                                                                                        *
 * **************************************************************************************************/
// ShiftLeft::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t>()), "ShiftLeft not support current datatype!");
    constexpr auto func = MicroAPI::ShiftLefts<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftLeftImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t>()), "ShiftLeft not support current datatype!");
    constexpr auto func = MicroAPI::ShiftLefts<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// ShiftLeft::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(ShiftLeftImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftLeftImpl, ShiftLefts, int32_t)

/* **************************************************************************************************
 * ShiftRight                                                                                       *
 * **************************************************************************************************/
// ShiftRight::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams, bool roundEn = false)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t>()), "ShiftRight not support current datatype!");
    constexpr auto func = MicroAPI::ShiftRights<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void ShiftRightImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams, bool roundEn = false)
{
    static_assert((SupportType<T, int16_t, uint16_t, int32_t, uint32_t>()), "ShiftRight not support current datatype!");
    constexpr auto func = MicroAPI::ShiftRights<T, int16_t, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// ShiftRight::Level 2
BINARY_SCALAR_OP_LEVEL2_IMPL_NOT_SUPPORT(ShiftRightImpl)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, uint8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, int8_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, uint16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, int16_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, uint32_t)
BINARY_SCALAR_OP_LEVEL2_IMPL(ShiftRightImpl, ShiftRights, int32_t)

/* **************************************************************************************************
 * Subs: LocalTensor - Scalar                                             *
 * ************************************************************************************************* */
// Subs::Level 0
namespace MicroAPISubs {
template <typename T, typename RegT>
__simd_callee__ inline void Subs(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Sub(dstReg, srcReg, dstReg, mask);
}
} // namespace MicroAPISubs
template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    constexpr auto func = MicroAPISubs::Subs<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    constexpr auto func = MicroAPISubs::Subs<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Subs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert(
        (SupportType<T, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Subs not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPISubs::Subs<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPISubs::Subs<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Subs: Scalar - LocalTensor                                             *
 * ************************************************************************************************* */
// Subs::Level 0
namespace MicroAPISubs {
template <typename T, typename RegT>
__simd_callee__ inline void Subs2(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Sub(dstReg, dstReg, srcReg, mask);
}
} // namespace MicroAPISubs

template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    constexpr auto func = MicroAPISubs::Subs2<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    constexpr auto func = MicroAPISubs::Subs2<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Subs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const int32_t &calCount)
{
    static_assert(
        (SupportType<T, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Subs not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPISubs::Subs2<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPISubs::Subs2<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Divs                                             *
 * ************************************************************************************************* */
namespace MicroAPIDivs {
template <typename T, typename RegT>
__simd_callee__ inline void Divs(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Div(dstReg, srcReg, dstReg, mask);
}
} // namespace MicroAPIDivs
// Divs::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    constexpr auto func = MicroAPIDivs::Divs<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    constexpr auto func = MicroAPIDivs::Divs<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Divs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, half, float, int64_t, uint64_t>()),
        "Divs not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPIDivs::Divs<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIDivs::Divs<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Divs   Scalar / LocalTensor                                         *
 * ************************************************************************************************* */
namespace MicroAPIDivs {
template <typename T, typename RegT>
__simd_callee__ inline void Divs2(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Div(dstReg, dstReg, srcReg, mask);
}
} // namespace MicroAPIDivs
// Div::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    constexpr auto func = MicroAPIDivs::Divs2<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    constexpr auto func = MicroAPIDivs::Divs2<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Divs::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, T scalarValue, __ubuf__ T *src, const int32_t &calCount)
{
    static_assert((SupportType<T, half, float, int64_t, uint64_t>()),
        "Divs not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPIDivs::Divs2<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIDivs::Divs2<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Ands                                             *
 * ************************************************************************************************* */
namespace MicroAPIAnds {
template <typename T, typename RegT>
__simd_callee__ inline void Ands(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::And(dstReg, dstReg, srcReg, mask);
}
} // namespace MicroAPIAnds
// Ands::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ands not support current datatype!");
    constexpr auto func = MicroAPIAnds::Ands<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ands not support current datatype!");
    constexpr auto func = MicroAPIAnds::Ands<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Ands::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, int16_t, uint16_t, int64_t, uint64_t>()), "Ands not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPIAnds::Ands<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIAnds::Ands<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * Ors                                             *
 * ************************************************************************************************* */
namespace MicroAPIOrs {
template <typename T, typename RegT>
__simd_callee__ inline void Ors(RegT &dstReg, RegT &srcReg, T scalarValue, MicroAPI::MaskReg &mask)
{
    MicroAPI::Duplicate(dstReg, scalarValue, mask);
    MicroAPI::Or(dstReg, dstReg, srcReg, mask);
}
} // namespace MicroAPIOrs
// Ors::Level 0
template <typename T, bool isSetMask = true>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ors not support current datatype!");
    constexpr auto func = MicroAPIOrs::Ors<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, true>(dst, src, scalarValue, mask, 0, repeatTime,
        repeatParams);
}

template <typename T, bool isSetMask = true>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ors not support current datatype!");
    constexpr auto func = MicroAPIOrs::Ors<T, MicroAPI::RegTensor<T>>;
    Internal::VecBinaryScalarLevel0Template<func, isSetMask, false>(dst, src, scalarValue, nullptr, mask, repeatTime,
        repeatParams);
}

// Ors::Level 2
template <typename T, bool isSetMask = true>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src, T scalarValue, const int32_t &calCount)
{
    static_assert((SupportType<T, int16_t, uint16_t, int64_t, uint64_t>()), "Ors not support current datatype!");
    if constexpr (SupportBytes<T, 8>()) {
        constexpr auto func = MicroAPIOrs::Ors<T, MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    } else {
        constexpr auto func = MicroAPIOrs::Ors<T, MicroAPI::RegTensor<T>>;
        Internal::VecBinaryScalarLevel2ImplTemplate<func, T>(dst, src, scalarValue, calCount);
    }
}

/* **************************************************************************************************
 * FusedMulsCast                                            *
 * ************************************************************************************************* */
// FusedMulsCast::Level 2

template <typename T, typename U>
__simd_vf__ inline void FusedMulsCastImpl(__ubuf__ T *dst, __ubuf__ U *src0, __ubuf__ U *src1, const uint32_t calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, float>>(), "Failed to check dtype in "
        "FusedMulsCast, current api support dtype combination is src : float, dst: half, scalar: float.");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(U));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    U scalar = src1[0];
    MicroAPI::RegTensor<T> vDstReg;
    MicroAPI::RegTensor<U> vSrcReg;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<U>(sreg);
        MicroAPI::DataCopy(vSrcReg, src0 + i * sregLower);
        MicroAPI::FusedMulsCast(vDstReg, vSrcReg, scalar, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, vDstReg, mask);
    }
}

template <typename T, typename U>
__simd_vf__ inline void FusedMulsCastImpl(__ubuf__ T *dst, __ubuf__ U *src, U scalar, const uint32_t calCount)
{
    static_assert(SupportType<Tuple<T, U>, Tuple<half, float>>(), "Failed to check dtype in "
        "FusedMulsCast, current api support dtype combination is src : float, dst: half, scalar: float.");
    constexpr uint32_t sregLower = static_cast<uint32_t>(VECTOR_REG_WIDTH / sizeof(U));
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::RegTensor<T> vDstReg;
    MicroAPI::RegTensor<U> vSrcReg;
    MicroAPI::MaskReg mask;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        mask = MicroAPI::UpdateMask<U>(sreg);
        MicroAPI::DataCopy(vSrcReg, src + i * sregLower);
        MicroAPI::FusedMulsCast(vDstReg, vSrcReg, scalar, mask);
        MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(dst + i * sregLower, vDstReg, mask);
    }
}

template <typename T, BinaryScalarOp op, uint8_t scalarIdx>
__simd_vf__ inline void BinaryScalarOpTemplateCntB64(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t calCount)
{
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumTwo> dstReg, srcReg0, srcReg1, dupReg;
    MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> preReg;
    MicroAPI::MaskReg mask;
    MicroAPI::UnalignReg ureg;
    MicroAPI::RegTensor<uint32_t> zeroReg;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    uint32_t repeatStride = static_cast<uint32_t>(B64_DATA_NUM_PER_REPEAT) * 2;
    uint16_t repeatTime = CeilDivision(calCount, repeatStride);
    MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(zeroReg, 0, maskFull);

    if constexpr (scalarIdx == 0) {
        MicroAPI::DataCopyUnAlignPre(ureg, (__ubuf__ T *)src0);
        MicroAPI::DataCopyUnAlign(preReg, ureg, (__ubuf__ T *)src0);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1], (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(srcReg0, dupReg, maskFull);
    } else if (scalarIdx == 1) {
        MicroAPI::DataCopyUnAlignPre(ureg, (__ubuf__ T *)src1);
        MicroAPI::DataCopyUnAlign(preReg, ureg, (__ubuf__ T *)src1);
        MicroAPI::DeInterleave((MicroAPI::RegTensor<uint32_t> &)dupReg.reg[0],
            (MicroAPI::RegTensor<uint32_t> &)dupReg.reg[1], (MicroAPI::RegTensor<uint32_t> &)preReg, zeroReg);
        MicroAPI::Duplicate(srcReg1, dupReg, maskFull);
    }

    for (uint16_t i = 0; i < static_cast<uint16_t>(repeatTime); ++i) {
        mask = MicroAPI::UpdateMask<T, MicroAPI::RegTraitNumTwo>(sreg);
        if constexpr (scalarIdx == 0) {
            MicroAPI::DataCopy<T>(srcReg1, src1 + i * repeatStride);
        } else if (scalarIdx == 1) {
            MicroAPI::DataCopy<T>(srcReg0, src0 + i * repeatStride);
        }
        if constexpr (op == BinaryScalarOp::ADDS) {
            MicroAPI::Add(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::MULS) {
            MicroAPI::Mul(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::MAXS) {
            MicroAPI::Max(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::MINS) {
            MicroAPI::Min(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::SUBS) {
            MicroAPI::Sub(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::DIVS) {
            MicroAPI::Div(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::ANDS) {
            MicroAPI::And(dstReg, srcReg0, srcReg1, mask);
        } else if constexpr (op == BinaryScalarOp::ORS) {
            MicroAPI::Or(dstReg, srcReg0, srcReg1, mask);
        }
        MicroAPI::DataCopy<T>(dst + i * repeatStride, dstReg, mask);
    }
}

template <typename T, MicroAPI::LoadDist pattern, BinaryScalarOp op, uint8_t scalarIdx>
__simd_vf__ inline void BinaryScalarOpTemplateCnt(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1,
    const int32_t calCount)
{
    MicroAPI::RegTensor<T> vSrcReg0;
    MicroAPI::RegTensor<T> vSrcReg1;
    MicroAPI::RegTensor<T> vDstReg0;
    uint32_t sreg = static_cast<uint32_t>(calCount);
    MicroAPI::MaskReg preg;

    uint32_t sregLower = static_cast<uint32_t>(GetVecLen() / sizeof(T));
    uint16_t repeatTime = static_cast<uint16_t>(CeilDivision(calCount, sregLower));
    for (uint16_t i = 0; i < repeatTime; ++i) {
        preg = MicroAPI::UpdateMask<T>(sreg);

        if constexpr (scalarIdx == 0) {
            MicroAPI::DataCopy<T, pattern>(vSrcReg0, src0);
            MicroAPI::DataCopy(vSrcReg1, src1 + i * sregLower);
        } else if constexpr (scalarIdx == 1) {
            MicroAPI::DataCopy(vSrcReg0, src0 + i * sregLower);
            MicroAPI::DataCopy<T, pattern>(vSrcReg1, src1);
        }

        if constexpr (op == BinaryScalarOp::ADDS) {
            MicroAPI::Add(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::MULS) {
            MicroAPI::Mul(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::MAXS) {
            MicroAPI::Max(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::MINS) {
            MicroAPI::Min(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::SUBS) {
            MicroAPI::Sub(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::DIVS) {
            MicroAPI::Div(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::ANDS) {
            MicroAPI::And(vDstReg0, vSrcReg0, vSrcReg1, preg);
        } else if constexpr (op == BinaryScalarOp::ORS) {
            MicroAPI::Or(vDstReg0, vSrcReg0, vSrcReg1, preg);
        }

        MicroAPI::DataCopy(dst + i * sregLower, vDstReg0, preg);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Adds not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Adds not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Add<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AddsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Adds not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 1) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B8, BinaryScalarOp::ADDS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 2) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::ADDS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::ADDS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::ADDS, scalarIdx>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Maxs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Maxs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Max<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MaxsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Maxs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 1) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B8, BinaryScalarOp::MAXS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 2) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::MAXS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::MAXS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::MAXS, scalarIdx>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Mins not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Mins not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Min<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MinsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, uint8_t, int8_t, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Mins not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 1) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B8, BinaryScalarOp::MINS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 2) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::MINS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::MINS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::MINS, scalarIdx>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Muls not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Muls not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Mul<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void MulsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert(
        (SupportType<T, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Muls not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::MULS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::MULS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::MULS, scalarIdx>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, bfloat16_t, float, int16_t, int32_t>()), "Subs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Sub<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void SubsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert(
        (SupportType<T, half, bfloat16_t, float, int16_t, int32_t, int64_t, uint64_t>()),
        "Subs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::SUBS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::SUBS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::SUBS, scalarIdx>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, half, float>()), "Divs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Div<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void DivsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, half, float, int64_t, uint64_t>()),
        "Divs not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::DIVS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::DIVS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::DIVS, scalarIdx>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ands not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ands not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::And<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void AndsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, int16_t, uint16_t, int64_t, uint64_t>()), "Ands not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::ANDS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::ANDS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::ANDS, scalarIdx>(dst, src0, src1, calCount);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask[],
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ors not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, true, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, mask, 0, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const uint64_t mask,
    const uint8_t repeatTime, const UnaryRepeatParams &repeatParams)
{
    static_assert((SupportType<T, int16_t, uint16_t>()), "Ors not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    constexpr auto func = MicroAPI::Or<T, MicroAPI::MaskMergeMode::ZEROING, MicroAPI::RegTensor<T>>;
    if constexpr (sizeof(T) == 2) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B16, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    } else if constexpr (sizeof(T) == 4) {
        Internal::VecBinaryScalarLevel0Template<func, isSetMask, false, T, MicroAPI::LoadDist::DIST_BRC_B32, scalarIdx>
                                        (dst, src0, src1, nullptr, mask, repeatTime, repeatParams);
    }
}

template <typename T, bool isSetMask = true, uint8_t scalarIdx = 1>
__aicore__ inline void OrsImpl(__ubuf__ T *dst, __ubuf__ T *src0, __ubuf__ T *src1, const int32_t &calCount)
{
    static_assert((SupportType<T, int16_t, uint16_t, int64_t, uint64_t>()), "Ors not support current datatype!");
    static_assert(scalarIdx == 0 || scalarIdx == 1);
    if constexpr (sizeof(T) == 2) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B16, BinaryScalarOp::ORS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 4) {
        BinaryScalarOpTemplateCnt<T, MicroAPI::LoadDist::DIST_BRC_B32, BinaryScalarOp::ORS, scalarIdx>(dst,
            src0, src1, calCount);
    } else if constexpr (sizeof(T) == 8) {
        BinaryScalarOpTemplateCntB64<T, BinaryScalarOp::ORS, scalarIdx>(dst, src0, src1, calCount);
    }
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_SCALAR_IMPL_H
