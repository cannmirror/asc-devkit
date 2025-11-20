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
 * \file kernel_operator_vec_reduce_impl.h
 * \brief AscendC l311 support reduce api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H

#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {

#define FLOAT_MAX (__FLT_MAX__)
#define FLOAT_MIN (-__FLT_MAX__)
#define HALF_MAX (65504.0)
#define HALF_MIN (-65504.0)
#define REDUCE_ADJUST_REPEAT_PARAM(src, repeatTime, dstRepStride, srcRepStride)                                     \
    uint16_t newRepeat = static_cast<uint16_t>(repeatTime);                                                         \
    uint32_t newDstRepStride = dstRepStride;                                                                    \
    __ubuf__ T* newSrc = src;                                                                                 \
    if (dstRepStride == 0 && repeatTime > 0) {                                                                      \
        newRepeat = 1;                                                                                          \
        newDstRepStride = 1;                                                                                    \
        uint32_t srcStrideOffset = srcRepStride * ONE_BLK_SIZE / sizeof(T) * (repeatTime - 1);                    \
        newSrc += srcStrideOffset;                                                                              \
    }

#define CONTINUOUS_MODE_REDUCE_VF(REDUCE_FUNC, dstStrideOffset)                                                 \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        RegTensor<T> vreg0;                                                                                   \
        RegTensor<T> vreg1;                                                                                   \
        UnalignReg ureg;                                                                                        \
        uint32_t sreg = mask;                                                                                   \
        MaskReg preg = CreatePredicate<T>(sreg);                                                              \
        for (uint16_t idx = 0; idx < newRepeat; ++idx) {                                                        \
            DataCopy<T, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
            REDUCE_FUNC<T>(vreg1, vreg0, preg);                                                               \
            DataCopyUnAlign<T>(dst, vreg1, ureg, dstStrideOffset);                                            \
            DataCopyUnAlignPost<T>(dst, ureg, dstStrideOffset * (newDstRepStride - 1));                       \
        }                                                                                                       \
    }

#define REDUCE_CONTINUOUS_MODE_IMPL(REDUCE_FUNC, DATA_TYPE, dstStrideOffset)                                            \
    template <class T, bool isSetMask>                                                                                \
    __aicore__ inline void REDUCE_FUNC##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t repeatTime,    \
        const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)         \
    {                                                                                                                   \
        REDUCE_ADJUST_REPEAT_PARAM(src, repeatTime, dstRepStride, srcRepStride)                                             \
        CONTINUOUS_MODE_REDUCE_VF(REDUCE_FUNC, dstStrideOffset)                                                         \
    }


#define BLOCK_REDUCE_IMPL_NOT_SUPPORT(REDUCE_FUNC, DATA_TYPE)                                                           \
    template <typename T, bool isSetMask>                                                                             \
    __aicore__ inline void REDUCE_FUNC##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t repeatTime,    \
        const int32_t mask, const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)         \
    {                                                                                                                   \
        static_assert(!std::is_same_v<uint32_t, T>, "current data type is not supported!");                           \
    }

#define BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(REDUCE_TYPE, DATA_TYPE)   \
    REDUCE_CONTINUOUS_MODE_IMPL(BlockReduce##REDUCE_TYPE, DATA_TYPE, DEFAULT_BLK_NUM)


BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(Max, half)
BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(Max, float)
BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(Min, half)
BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(Min, float)
BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(Sum, half)
BLOCK_REDUCE_CONTINUOUS_MODE_IMPL(Sum, float)

#define BITBYBIT_MODE_REDUCE_VF(REDUCE_FUNC, dstStrideOffset)                                                   \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        RegTensor<T> vreg0;                                                                                   \
        RegTensor<T> vreg1;                                                                                   \
        UnalignReg ureg;                                                                                        \
        MaskReg preg = MovePredicate<T>();                                                                    \
        for (uint16_t idx = 0; idx < newRepeat; ++idx) {                                                        \
            DataCopy<T, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
            REDUCE_FUNC<T>(vreg1, vreg0, preg);                                                               \
            DataCopyUnAlign<T>(dst, vreg1, ureg, dstStrideOffset);                                            \
            DataCopyUnAlignPost<T>(dst, ureg, dstStrideOffset * (newDstRepStride - 1));                       \
        }                                                                                                       \
    }

#define REDUCE_BITBYBIT_MODE_IMPL(REDUCE_FUNC, DATA_TYPE, dstStrideOffset)                                              \
    template <typename T, bool isSetMask>                                                                             \
    __aicore__ inline void REDUCE_FUNC##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src, const int32_t repeatTime,    \
        const uint64_t mask[2], const int32_t dstRepStride, const int32_t srcBlkStride, const int32_t srcRepStride)     \
    {                                                                                                                   \
        if constexpr (isSetMask) {                                                                                      \
            SetVectorMask<T>(mask[1], mask[0]);                                                                       \
        }                                                                                                               \
                                                                                                                        \
        REDUCE_ADJUST_REPEAT_PARAM(src, repeatTime, dstRepStride, srcRepStride)                                             \
        BITBYBIT_MODE_REDUCE_VF(REDUCE_FUNC, dstStrideOffset)                                                           \
    }

#define BLOCK_REDUCE_BITBYBIT_MODE_IMPL(REDUCE_TYPE, DATA_TYPE)   \
    REDUCE_BITBYBIT_MODE_IMPL(BlockReduce##REDUCE_TYPE, DATA_TYPE, DEFAULT_BLK_NUM)

BLOCK_REDUCE_BITBYBIT_MODE_IMPL(Max, half)
BLOCK_REDUCE_BITBYBIT_MODE_IMPL(Max, float)
BLOCK_REDUCE_BITBYBIT_MODE_IMPL(Min, half)
BLOCK_REDUCE_BITBYBIT_MODE_IMPL(Min, float)
BLOCK_REDUCE_BITBYBIT_MODE_IMPL(Sum, half)
BLOCK_REDUCE_BITBYBIT_MODE_IMPL(Sum, float)

/* **************************************** Whole Reduce Interface ****************************************** */
#define WHOLE_REDUCE_ADJUST_REPEAT_PARAM(src, repeatTime, dstRepStride, srcRepStride)                               \
    uint16_t newRepeat = static_cast<uint16_t>(repeatTime);                                                         \
    uint32_t newDstRepStride = dstRepStride;                                                                    \
    uint32_t dstStrideOffset = (dstRepStride > 1) ? 2 : 1;                                                      \
    __ubuf__ T* newSrc = src;                                                                                 \
    if (dstRepStride == 0 && repeatTime > 0) {                                                                      \
        newRepeat = 1;                                                                                          \
        newDstRepStride = 1;                                                                                    \
        uint32_t srcStrideOffset = srcRepStride * ONE_BLK_SIZE / sizeof(T) * (repeatTime - 1);                    \
        newSrc += srcStrideOffset;                                                                              \
        dstStrideOffset = 1;                                                                                    \
    }

#define WHOLE_REDUCE_CONTINUOUS_MODE_REDUCE_VF(REDUCE_TYPE)                                                     \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        RegTensor<T> vreg0;                                                                                   \
        RegTensor<T> vreg1;                                                                                   \
        UnalignReg ureg;                                                                                        \
        uint32_t sreg = mask;                                                                                   \
        MaskReg preg = CreatePredicate<T>(sreg);                                                              \
        for (uint16_t idx = 0; idx < newRepeat; ++idx) {                                                        \
            DataCopy<T, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
            Reduce##REDUCE_TYPE<T>(vreg1, vreg0, preg);                                                       \
            DataCopyUnAlign<T>(dst, vreg1, ureg, dstStrideOffset);                                            \
            DataCopyUnAlignPost<T>(dst, ureg, newDstRepStride - dstStrideOffset);                             \
        }                                                                                                       \
    }

#define WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(REDUCE_TYPE, DATA_TYPE)                                               \
    template <class T, bool isSetMask>                                                                        \
    __aicore__ inline void WholeReduce##REDUCE_TYPE##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src,     \
        const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,  \
        const int32_t srcRepStride, ReduceOrder order)                                                          \
    {                                                                                                           \
        WHOLE_REDUCE_ADJUST_REPEAT_PARAM(src, repeatTime, dstRepStride, srcRepStride)                          \
        WHOLE_REDUCE_CONTINUOUS_MODE_REDUCE_VF(REDUCE_TYPE)                                                     \
    }

WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Max, int8_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Max, int16_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Max, int32_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Max, uint8_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Max, uint16_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Max, uint32_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Max, half)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Max, float)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, int8_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, int16_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, int32_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, uint8_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, uint16_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, uint32_t)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, half)
WHOLE_REDUCE_CONTINUOUS_MODE_IMPL(Min, float)

#define WHOLE_REDUCE_SUM_CONTINUOUS_MODE_IMPL(DATA_TYPE)                                                            \
    template <class T, bool isSetMask>                                                                            \
    __aicore__ inline void WholeReduceSumImpl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src,                     \
        const int32_t mask, const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,      \
        const int32_t srcRepStride)                                                                                 \
    {                                                                                                               \
        REDUCE_ADJUST_REPEAT_PARAM(src, repeatTime, dstRepStride, srcRepStride)                                    \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<T> vreg0;                                                                                   \
            RegTensor<T> vreg1;                                                                                   \
            UnalignReg ureg;                                                                                        \
            uint32_t sreg = mask;                                                                                   \
            MaskReg preg = CreatePredicate<T>(sreg);                                                              \
            for (uint16_t idx = 0; idx < newRepeat; ++idx) {                                                        \
                DataCopy<T, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
                ReduceSum<T, T>(vreg1, vreg0, preg);                                                            \
                DataCopyUnAlign<T>(dst, vreg1, ureg, 1);                                                          \
                DataCopyUnAlignPost<T>(dst, ureg, newDstRepStride - 1);                                           \
            }                                                                                                       \
        }                                                                                                           \
    }

WHOLE_REDUCE_SUM_CONTINUOUS_MODE_IMPL(int32_t)
WHOLE_REDUCE_SUM_CONTINUOUS_MODE_IMPL(uint32_t)
WHOLE_REDUCE_SUM_CONTINUOUS_MODE_IMPL(half)
WHOLE_REDUCE_SUM_CONTINUOUS_MODE_IMPL(float)


#define WHOLE_REDUCE_BITBYBIT_MODE_IMPL(REDUCE_TYPE, DATA_TYPE)                                                     \
    template <typename T, bool isSetMask>                                                                         \
    __aicore__ inline void WholeReduce##REDUCE_TYPE##Impl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src,         \
        const uint64_t mask[2], const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,  \
        const int32_t srcRepStride, ReduceOrder order)                                                              \
    {                                                                                                               \
        if constexpr (isSetMask) {                                                                                  \
            SetVectorMask<T>(mask[1], mask[0]);                                                                   \
        }                                                                                                           \
                                                                                                                    \
        WHOLE_REDUCE_ADJUST_REPEAT_PARAM(src, repeatTime, dstRepStride, srcRepStride)                              \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<T> vreg0;                                                                                   \
            RegTensor<T> vreg1;                                                                                   \
            UnalignReg ureg;                                                                                        \
            MaskReg preg = MovePredicate<T>();                                                                    \
            for (uint16_t i = 0; i < newRepeat; ++i) {                                                              \
                DataCopy<T, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
                Reduce##REDUCE_TYPE<T>(vreg1, vreg0, preg);                                                       \
                DataCopyUnAlign<T>(dst, vreg1, ureg, dstStrideOffset);                                            \
                DataCopyUnAlignPost<T>(dst, ureg, newDstRepStride - dstStrideOffset);                             \
            }                                                                                                       \
        }                                                                                                           \
    }

WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Max, int16_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Max, int32_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Max, uint16_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Max, uint32_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Max, half)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Max, float)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, int16_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, int32_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, uint16_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, uint32_t)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, half)
WHOLE_REDUCE_BITBYBIT_MODE_IMPL(Min, float)

#define WHOLE_REDUCE_SUM_BITBYBIT_MODE_IMPL(DATA_TYPE)                                                              \
    template <class T, bool isSetMask>                                                                            \
    __aicore__ inline void WholeReduceSumImpl(__ubuf__ DATA_TYPE* dst, __ubuf__ DATA_TYPE* src,                     \
        const uint64_t mask[2], const int32_t repeatTime, const int32_t dstRepStride, const int32_t srcBlkStride,  \
        const int32_t srcRepStride)                                                                                 \
    {                                                                                                               \
        REDUCE_ADJUST_REPEAT_PARAM(src, repeatTime, dstRepStride, srcRepStride)                                    \
        __VEC_SCOPE__                                                                                               \
        {                                                                                                           \
            RegTensor<T> vreg0;                                                                                   \
            RegTensor<T> vreg1;                                                                                   \
            UnalignReg ureg;                                                                                        \
            MaskReg preg = MovePredicate<T>();                                                                    \
            for (uint16_t i = 0; i < newRepeat; ++i) {                                                              \
                DataCopy<T, PostLiteral::POST_MODE_UPDATE>(vreg0, newSrc, srcBlkStride, srcRepStride, preg);      \
                ReduceSum<T, T>(vreg1, vreg0, preg);                                                            \
                DataCopyUnAlign<T>(dst, vreg1, ureg, 1);                                                          \
                DataCopyUnAlignPost<T>(dst, ureg, newDstRepStride - 1);                                           \
            }                                                                                                       \
        }                                                                                                           \
    }

WHOLE_REDUCE_SUM_BITBYBIT_MODE_IMPL(int32_t)
WHOLE_REDUCE_SUM_BITBYBIT_MODE_IMPL(uint32_t)
WHOLE_REDUCE_SUM_BITBYBIT_MODE_IMPL(half)
WHOLE_REDUCE_SUM_BITBYBIT_MODE_IMPL(float)


/* **************************************** Pair Reduce Interface ****************************************** */

REDUCE_CONTINUOUS_MODE_IMPL(PairReduceSum, half, FULL_MASK_LEN / HALF_FACTOR)
REDUCE_CONTINUOUS_MODE_IMPL(PairReduceSum, float, HLAF_MASK_LEN / HALF_FACTOR)
REDUCE_BITBYBIT_MODE_IMPL(PairReduceSum, half, FULL_MASK_LEN / HALF_FACTOR)
REDUCE_BITBYBIT_MODE_IMPL(PairReduceSum, float, HLAF_MASK_LEN / HALF_FACTOR)


/* ****************************************** Reduce Interface ******************************************** */

// Level 2
// 将所有输入数据用vadd/vmax/vmin指令压缩成一条指令可以处理的数据量
#define REDUCE_TO_VEC(opName, vregType, type, dataCount, offsetNum, initVal, typeName)       \
    __VEC_SCOPE__                                                                            \
    {                                                                                        \
        uint16_t loop_num = (dataCount + ELE_CNT_B##type - 1) / ELE_CNT_B##type;             \
        for (uint16_t i = 0; i < 1; i++) {                                                   \
            MaskReg pgAll = CreatePredicate<typeName>();                                     \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                    \
            RegTensor<typeName> vreg1;                                                       \
            Duplicate<typeName, typeName>(vreg1, initVal);                                   \
            uint32_t sreg = (uint32_t)dataCount;                                             \
            for (uint16_t j = 0; j < loop_num; ++j) {                                        \
                MaskReg preg =  CreatePredicate<typeName>(sreg);                              \
                AddrReg offset1 = CreateAddrReg<typeName>(0, static_cast<uint16_t>(offsetNum)); \
                RegTensor<typeName> vreg0;                                                    \
                DataCopy<typeName>(vreg0, src, offset1);                                     \
                opName<typeName, Mode::MERGING>(vreg1, vreg0, vreg1, preg);               \
            }                                                                                \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, vreg1, offset0, pgAll);     \
        }                                                                                    \
    }

#define VEC_REDUCE_SUM_IMPL(opName, dst, src, type, dataCount, typeName)           \
    __VEC_SCOPE__                                                                            \
    {                                                                                        \
        uint16_t loop_num = (uint16_t)(dataCount + ELE_CNT_B##type - 1) / ELE_CNT_B##type;   \
        uint32_t sreg = (uint32_t)dataCount;                                                 \
        for (uint16_t i = 0; i < loop_num; i++) {                                            \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                    \
            MaskReg preg =  CreatePredicate<typeName>(sreg);                                 \
            MaskReg preg1 = CreatePredicate<typeName, Pat::VL1>();                          \
            RegTensor<typeName> dstVreg;                                                     \
            RegTensor<typeName> srcVreg;                                                     \
            DataCopy<typeName>(srcVreg, src, offset0);                                       \
            opName<typeName, typeName, Mode::MERGING>(dstVreg, srcVreg, preg);             \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(dst, dstVreg, offset0, preg1);    \
        }                                                                                    \
    }

#define VEC_REDUCE_IMPL(opName, dst, src, type, dataCount, typeName)                        \
    __VEC_SCOPE__                                                                            \
    {                                                                                        \
        uint16_t loop_num = (uint16_t)(dataCount + ELE_CNT_B##type - 1) / ELE_CNT_B##type;   \
        uint32_t sreg = (uint32_t)dataCount;                                                 \
        for (uint16_t i = 0; i < loop_num; i++) {                                            \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                    \
            MaskReg preg =  CreatePredicate<typeName>(sreg);                                 \
            MaskReg preg1 = CreatePredicate<typeName,  Pat::VL1>();                          \
            RegTensor<typeName> dstVreg;                                                     \
            RegTensor<typeName> srcVreg;                                                     \
            DataCopy<typeName>(srcVreg, src, offset0);                                       \
            opName<typeName, Mode::MERGING>(dstVreg, srcVreg, preg);                         \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(dst, dstVreg, offset0, preg1);    \
        }                                                                                    \
    }

#define REDUCE_SUM_NO_INDEX_IMPL(reduceFuncName, instrName, typeName, vregType, type, initVal)           \
    template <>                                                                                          \
    __aicore__ inline void reduceFuncName##Impl<typeName>(                                               \
        __ubuf__ typeName * dst, __ubuf__ typeName * src, __ubuf__ typeName * work, const int32_t count) \
    {                                                                                                    \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                            \
        REDUCE_TO_VEC(instrName, vregType, type, count, elementNumPerInstr, initVal, typeName);          \
        VEC_REDUCE_SUM_IMPL(reduceFuncName, dst, work, type, elementNumPerInstr, typeName);              \
    }

#define REDUCE_NO_INDEX_IMPL(reduceFuncName, instrName, typeName, vregType, type, initVal)               \
    template <>                                                                                          \
    __aicore__ inline void reduceFuncName##Impl<typeName>(                                               \
        __ubuf__ typeName * dst, __ubuf__ typeName * src, __ubuf__ typeName * work, const int32_t count) \
    {                                                                                                    \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                            \
        REDUCE_TO_VEC(instrName, vregType, type, count, elementNumPerInstr, initVal, typeName);          \
        VEC_REDUCE_IMPL(Reduce##instrName, dst, work, type, elementNumPerInstr, typeName);               \
    }

template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceSumImpl is not supported!"); });
}

// ReduceSumImpl::Level 2
REDUCE_SUM_NO_INDEX_IMPL(ReduceSum, Add, float, f32, 32, 0.0f)
REDUCE_SUM_NO_INDEX_IMPL(ReduceSum, Add, uint32_t, u32, 32, 0)
REDUCE_SUM_NO_INDEX_IMPL(ReduceSum, Add, int32_t, s32, 32, 0)
REDUCE_SUM_NO_INDEX_IMPL(ReduceSum, Add, half, f16, 16, 0.0f)

template <typename T>
__aicore__ inline void ReduceMaxNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, half, f16, 16, HALF_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, float, f32, 32, FLOAT_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, uint32_t, u32, 32, 0)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, int32_t, s32, 32, INT32_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, uint16_t, u16, 16, 0)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, int16_t, s16, 16, INT16_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, uint8_t, u8, 8, 0)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, Max, int8_t, s8, 8, INT8_MIN)

template <typename T>
__aicore__ inline void ReduceMinNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, half, f16, 16, HALF_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, float, f32, 32, FLOAT_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, uint32_t, u32, 32, UINT32_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, int32_t, s32, 32, INT32_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, uint16_t, u16, 16, UINT16_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, int16_t, s16, 16, INT16_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, uint8_t, u8, 8, UINT8_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, Min, int8_t, s8, 8, INT8_MAX)

// vcmp_lt/vcmp_gt指令用于寻找在当前加载的数据中找到比之前数据更小/更大的数的下标
// vsel指令用于更新已加载的最大/最小值数据下标
#define REDUCE_TO_VEC_WITH_INDEX(opName, cmpOp, type, count, offsetNum, initVal, typeName, indexType, \
    initIndexType)                                                                                              \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        uint16_t loop_num = (uint16_t)(count + ELE_CNT_B##type - 1) / ELE_CNT_B##type;                          \
        for (uint16_t i = 0; i < 1; i++) {                                                                      \
            MaskReg pgAll = CreatePredicate<typeName>();                                                        \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                                       \
            RegTensor<indexType> dstIndexVreg;                                                                  \
            RegTensor<indexType> indexVreg;                                                                     \
            RegTensor<initIndexType> initIndexVreg;                                                             \
            RegTensor<typeName> dstVreg;                                                                        \
            Duplicate<indexType, indexType>(dstIndexVreg, 0);                                                   \
            CreateVecIndex<initIndexType, Order::INC_ORDER_VALUE, indexType>(initIndexVreg, 0);                \
            Duplicate<typeName, typeName>(dstVreg, initVal);                                                   \
            indexVreg = (RegTensor<indexType> &)initIndexVreg;                                                  \
            uint32_t sreg = (uint32_t)count;                                                                    \
            for (uint16_t j = 0; j < loop_num; j++) {                                                           \
                AddrReg offset1 = CreateAddrReg<typeName>(0, static_cast<uint16_t>(offsetNum));                 \
                MaskReg preg = CreatePredicate<typeName>(sreg);                                                 \
                MaskReg pd;                                                                                     \
                MaskReg indexPg;                                                                                \
                RegTensor<typeName> srcVreg;                                                                    \
                DataCopy<typeName>(srcVreg, src, offset1);                                                      \
                Compare<typeName, cmpOp>(indexPg, srcVreg, dstVreg, preg);                                      \
                Select<indexType>(dstIndexVreg, indexVreg, dstIndexVreg, indexPg);                              \
                opName<typeName, Mode::MERGING>(dstVreg, srcVreg, dstVreg, preg);                               \
                Adds<indexType, uint32_t, Mode::MERGING>(indexVreg, indexVreg, offsetNum, preg);                \
            }                                                                                                   \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, dstVreg, offset0, pgAll);                      \
            DataCopy<indexType, DistVST::DIST_NORM_B##type>(tempBuf, dstIndexVreg, offset0, pgAll);             \
        }                                                                                                       \
    }

#define REDUCE_MAX_MIN_WITH_INDEX_IMPL(                                                   \
    opName, type, count, initVal, slideNum, typeName, indexType, initIndexType)           \
    __VEC_SCOPE__                                                                         \
    {                                                                                     \
        uint16_t loop_num = (uint16_t)(count + ELE_CNT_B##type - 1) / ELE_CNT_B##type;    \
        uint32_t sreg = (uint32_t)count;                                                  \
        for (uint16_t i = 0; i < loop_num; i++) {                                         \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                 \
            MaskReg preg = CreatePredicate<typeName>(sreg);                               \
            MaskReg preg1 = CreatePredicate<typeName, Pat::VL2>();                        \
            MaskReg pd;                                                                   \
            RegTensor<indexType> indexVreg, zeroVreg;                                     \
            RegTensor<indexType> dstIndexVreg;                                            \
            RegTensor<typeName> srcVreg;                                                  \
            RegTensor<typeName> dstVreg;                                                  \
            Duplicate<indexType, initIndexType>(dstIndexVreg, initVal);                   \
            Duplicate<indexType, indexType>(zeroVreg, 0);                                 \
            DataCopy<typeName>(srcVreg, work, offset0);                                   \
            DataCopy<indexType>(indexVreg, tempBuf, offset0);                             \
            opName<typeName, Mode::MERGING>(dstVreg, pd, srcVreg, preg);                  \
            Select<indexType>(dstIndexVreg, indexVreg, dstIndexVreg, pd);                 \
            ReduceMin<indexType>(dstIndexVreg, dstIndexVreg, preg);                       \
            RegTensor<typeName> temZeroVreg = (RegTensor<typeName> &)zeroVreg;            \
            Slide<typeName>(dstVreg, temZeroVreg, dstVreg, (uint16_t)1);                  \
            RegTensor<typeName> temDestIndexVreg = (RegTensor<typeName> &)dstIndexVreg;   \
            Slide<typeName>(dstVreg, dstVreg, temDestIndexVreg, (uint16_t)slideNum);      \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(dst, dstVreg, offset0, preg1); \
        }                                                                                 \
    }

#define REDUCE_MAX_WITH_INDEX_IMPL(typeName, indexType, initIndexType, type, initVal, indexInitVal)           \
    template <>                                                                                          \
    __aicore__ inline void ReduceMaxWithIndexImpl<typeName>(                                             \
        __ubuf__ typeName * dst, __ubuf__ typeName * src, __ubuf__ typeName * work, const int32_t count) \
    {                                                                                                    \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                            \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                            \
        __ubuf__ indexType *tempBuf =                                                                    \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);          \
        REDUCE_TO_VEC_WITH_INDEX(Max, CMPMODE::GT, type, count, elementNumPerInstr, initVal, typeName, indexType, initIndexType); \
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMax, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType)  \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                           \
    }

#define REDUCE_MIN_WITH_INDEX_IMPL(typeName, indexType, initIndexType, type, initVal, indexInitVal)           \
    template <>                                                                                          \
    __aicore__ inline void ReduceMinWithIndexImpl<typeName>(                                             \
        __ubuf__ typeName * dst, __ubuf__ typeName * src, __ubuf__ typeName * work, const int32_t count) \
    {                                                                                                    \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                            \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                            \
        __ubuf__ indexType *tempBuf =                                                                    \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);          \
        REDUCE_TO_VEC_WITH_INDEX(Min, CMPMODE::LT, type, count, elementNumPerInstr, initVal, typeName, indexType, initIndexType);\
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMin, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType)  \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                           \
    }

template <typename T>
__aicore__ inline void ReduceMaxWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxWithIndexImpl is not supported!"); });
}

REDUCE_MAX_WITH_INDEX_IMPL(half, uint16_t, int16_t, 16, HALF_MIN, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(float, uint32_t, int32_t, 32, FLOAT_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(uint32_t, uint32_t,int32_t,   32, 0, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(int32_t, uint32_t, int32_t, 32, INT32_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(uint16_t, uint16_t, int16_t, 16, 0, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(int16_t, uint16_t, int16_t,  16, INT16_MIN, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(uint8_t, uint8_t, int8_t,  8, 0, UINT8_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(int8_t, uint8_t, int8_t, 8, INT8_MIN, UINT8_MAX)

template <typename T>
__aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinWithIndexImpl is not supported!"); });
}

REDUCE_MIN_WITH_INDEX_IMPL(half, uint16_t, int16_t, 16, HALF_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(float, uint32_t, int32_t, 32, FLOAT_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(uint32_t, uint32_t, int32_t, 32, UINT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(int32_t, uint32_t, int32_t, 32, INT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(uint16_t, uint16_t, int16_t, 16, UINT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(int16_t, uint16_t, int16_t, 16, INT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(uint8_t, uint8_t, int8_t, 8, UINT8_MAX, UINT8_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(int8_t, uint8_t, int8_t, 8, INT8_MAX, UINT8_MAX)

// level 0 mask 连续模式
// mask 连续模式将所有输入数据用vadd/vmax/vmin指令压缩成一条指令可以处理的数据量后累加
#define REDUCE_TO_VEC_CONTINUOUS_MODE(opName, typeName, type, initVal)                                         \
    __VEC_SCOPE__                                                                                               \
    {                                                                                                           \
        int32_t newRepeat = repeatTime;                                                                        \
        uint32_t oneBlockNum = ONE_BLK_SIZE / B##type##_BYTE_SIZE;                                              \
        uint32_t srcStrideOffset = srcRepStride * oneBlockNum;                                                  \
        __ubuf__ typeName *newSrc = src;                                                                        \
        for (uint16_t i = 0; i < 1; i++) {                                                                      \
            MaskReg pgAll = CreatePredicate<typeName>();                                                        \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                                       \
            RegTensor<typeName> vreg1;                                                                          \
            Duplicate<typeName, typeName>(vreg1, initVal);                                                      \
            uint32_t sreg = (uint32_t)mask;                                                                     \
            MaskReg preg = CreatePredicate<typeName>(sreg);                                                     \
            for (uint16_t j = 0; j < (uint16_t)newRepeat; ++j) {                                                \
                RegTensor<typeName> vreg0;                                                                      \
                DataCopy<typeName, PostLiteral ::POST_MODE_UPDATE>(vreg0, newSrc, (uint32_t)DEFAULT_BLK_STRIDE, \
                    srcRepStride, preg);                                                                        \
                opName<typeName, Mode::MERGING>(vreg1, vreg0, vreg1, preg);                                     \
            }                                                                                                   \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, vreg1, offset0, pgAll);                        \
        }                                                                                                       \
    }

// L0 连续模式 no index
#define REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(reduceFuncName, instrName, typeName, vregType, type, initVal) \
    template <>                                                                                             \
    __aicore__ inline void reduceFuncName##Impl(__ubuf__ typeName *dst,                                    \
        __ubuf__ typeName *src,                                                                            \
        __ubuf__ typeName *work,                                                                           \
        const int32_t mask,                                                                                \
        const int32_t repeatTime,                                                                         \
        const int32_t srcRepStride)                                                                        \
    {                                                                                                      \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                              \
        REDUCE_TO_VEC_CONTINUOUS_MODE(instrName, typeName, type, initVal)                                \
        VEC_REDUCE_IMPL(Reduce##instrName, dst, work, type, elementNumPerInstr, typeName)            \
    }

#define REDUCE_SUM_NO_INDEX_CONTINUOUS_MODE_IMPL(reduceFuncName, instrName, typeName, type, initVal)       \
    template <>                                                                                            \
    __aicore__ inline void reduceFuncName##Impl(__ubuf__ typeName *dst,                                    \
        __ubuf__ typeName *src,                                                                            \
        __ubuf__ typeName *work,                                                                           \
        const int32_t mask,                                                                                \
        const int32_t repeatTime,                                                                         \
        const int32_t srcRepStride)                                                                        \
    {                                                                                                      \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                              \
        REDUCE_TO_VEC_CONTINUOUS_MODE(instrName, typeName, type, initVal);                                 \
        VEC_REDUCE_SUM_IMPL(reduceFuncName, dst, work, type, elementNumPerInstr, typeName);                \
    }

//level 0 mask 连续
template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceSumImpl is not supported!"); });
}

REDUCE_SUM_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceSum, Add, float, 32, 0.0f)
REDUCE_SUM_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceSum, Add, uint32_t, 32, 0)
REDUCE_SUM_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceSum, Add, int32_t, 32, 0)
REDUCE_SUM_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceSum, Add, half, 16, 0.0f)

template <typename T>
__aicore__ inline void ReduceMaxNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMaxNoIndex, Max, half, f16, 16, HALF_MIN)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMaxNoIndex, Max, float, f32, 32, FLOAT_MIN)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMaxNoIndex, Max, uint32_t, u32, 32, 0)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMaxNoIndex, Max, int32_t, s32, 32, INT32_MIN)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMaxNoIndex, Max, uint16_t, u16, 16, 0)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMaxNoIndex, Max, int16_t, s16, 16, INT16_MIN)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMaxNoIndex, Max, uint8_t, u8, 8, 0)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMaxNoIndex, Max, int8_t, s8, 8, INT8_MIN)

template <typename T>
__aicore__ inline void ReduceMinNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, half, f16, 16, HALF_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, float, f32, 32, FLOAT_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, uint32_t, u32, 32, UINT32_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, int32_t, s32, 32, INT32_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, uint16_t, u16, 16, UINT16_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, int16_t, s16, 16, INT16_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, uint8_t, u8, 8, UINT8_MAX)
REDUCE_NO_INDEX_CONTINUOUS_MODE_IMPL(ReduceMinNoIndex, Min, int8_t, s8, 8, INT8_MAX)

// level 0 mark 连续 with index
#define REDUCE_TO_VEC_WITH_INDEX_CONTINUOUS_MODE(                                                    \
    opName, cmpOp, type, offsetNum, initVal, typeName, indexType, initIndexType)                     \
    __VEC_SCOPE__                                                                                    \
    {                                                                                                \
        int32_t newRepeat = repeatTime;                                                             \
        uint32_t oneBlockNum = ONE_BLK_SIZE / B##type##_BYTE_SIZE;                                   \
        uint32_t srcStrideOffset = srcRepStride * oneBlockNum;                                       \
        __ubuf__ typeName *newSrc = src;                                                             \
        for (uint16_t i = 0; i < 1; i++) {                                                           \
            MaskReg pgAll = CreatePredicate<typeName>();                                             \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                            \
            RegTensor<indexType> dstIndexVreg;                                                       \
            RegTensor<indexType> indexVreg;                                                          \
            RegTensor<initIndexType> initIndexVreg;                                                  \
            RegTensor<typeName> dstVreg;                                                             \
            Duplicate<indexType, indexType>(dstIndexVreg, 0);                                    \
            CreateVecIndex<initIndexType, Order::INC_ORDER_VALUE, indexType>(initIndexVreg, 0);   \
            Duplicate<typeName, typeName>(dstVreg, initVal);                                     \
            indexVreg = (RegTensor<indexType> &)initIndexVreg;                                       \
            uint32_t sreg = (uint32_t)mask;                                                          \
            MaskReg preg = CreatePredicate<typeName>(sreg);                                          \
            uint32_t strideConfig = ((uint32_t)DEFAULT_BLK_STRIDE) << 16 | (srcRepStride & 0xFFFFU); \
            for (uint16_t j = 0; j < (uint16_t)newRepeat; j++) {                                     \
                MaskReg pd;                                                                          \
                MaskReg indexPg;                                                                     \
                RegTensor<typeName> srcVreg;                                                         \
                DataCopy<typeName, PostLiteral ::POST_MODE_UPDATE>(                                  \
                    srcVreg, newSrc, (uint32_t)DEFAULT_BLK_STRIDE, srcRepStride, preg);              \
                Compare<typeName, cmpOp>(indexPg, srcVreg, dstVreg, preg);                \
                Select<indexType>(dstIndexVreg, indexVreg, dstIndexVreg, indexPg);                   \
                opName<typeName, Mode::MERGING>(dstVreg, srcVreg, dstVreg, preg);                  \
                Adds<indexType, uint32_t, Mode::MERGING>(indexVreg, indexVreg, offsetNum, preg);     \
            }                                                                                        \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, dstVreg, offset0, pgAll);           \
            DataCopy<indexType, DistVST::DIST_NORM_B##type>(tempBuf, dstIndexVreg, offset0, pgAll);  \
        }                                                                                            \
    }


#define REDUCE_MAX_WITH_INDEX_IMPL_CONTINUOUS_MODE(typeName, indexType, initIndexType, type, initVal, indexInitVal)   \
    template <typename T = typeName>                                                                             \
    __aicore__ inline void ReduceMaxWithIndexImpl(__ubuf__ typeName *dst,                                        \
        __ubuf__ typeName *src,                                                                                  \
        __ubuf__ typeName *work,                                                                                 \
        const int32_t mask,                                                                                      \
        const int32_t repeatTime,                                                                               \
        const int32_t srcRepStride)                                                                              \
    {                                                                                                            \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                                    \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                                    \
        __ubuf__ indexType *tempBuf =                                                                            \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);                  \
        REDUCE_TO_VEC_WITH_INDEX_CONTINUOUS_MODE(Max, CMPMODE::GT, type, elementNumPerInstr, initVal, typeName, indexType, initIndexType) \
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMax, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType) \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                                   \
    }  // namespace AscendC

#define REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(typeName, indexType, initIndexType, type, initVal, indexInitVal)   \
    template <typename T = typeName>                                                                             \
    __aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ typeName *dst,                                        \
        __ubuf__ typeName *src,                                                                                  \
        __ubuf__ typeName *work,                                                                                 \
        const int32_t mask,                                                                                      \
        const int32_t repeatTime,                                                                               \
        const int32_t srcRepStride)                                                                              \
    {                                                                                                            \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                                    \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                                    \
        __ubuf__ indexType *tempBuf =                                                                            \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);                  \
        REDUCE_TO_VEC_WITH_INDEX_CONTINUOUS_MODE(Min, CMPMODE::LT, type, elementNumPerInstr, initVal, typeName, indexType, initIndexType) \
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMin, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType) \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                                   \
    }

template <typename T>
__aicore__ inline void ReduceMaxWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxWithIndexImpl is not supported!"); });
}

REDUCE_MAX_WITH_INDEX_IMPL_CONTINUOUS_MODE(half, uint16_t, int16_t, 16, HALF_MIN, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_CONTINUOUS_MODE(float, uint32_t, int32_t, 32, FLOAT_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_CONTINUOUS_MODE(uint32_t, uint32_t, int32_t, 32, 0, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_CONTINUOUS_MODE(int32_t, uint32_t, int32_t, 32, INT32_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_CONTINUOUS_MODE(uint16_t, uint16_t, int16_t, 16, 0, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_CONTINUOUS_MODE(int16_t, uint16_t, int16_t, 16, INT16_MIN, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_CONTINUOUS_MODE(uint8_t, uint8_t, int8_t, 8, 0, UINT8_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_CONTINUOUS_MODE(int8_t, uint8_t, int8_t, 8, INT8_MIN, UINT8_MAX)

template <typename T>
__aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const int32_t mask,
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxWithIndexImpl is not supported!"); });
}

REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(half, uint16_t, int16_t, 16, HALF_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(float, uint32_t, int32_t, 32, FLOAT_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(uint32_t, uint32_t, int32_t, 32, UINT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(int32_t, uint32_t, int32_t, 32, INT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(uint16_t, uint16_t, int16_t, 16, UINT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(int16_t, uint16_t, int16_t, 16, INT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(uint8_t, uint8_t, int8_t, 8, UINT8_MAX, UINT8_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_CONTINUOUS_MODE(int8_t, uint8_t, int8_t, 8, INT8_MAX, UINT8_MAX)

// level 0  bit by bit
// mask bit by bit模式将所有输入数据用vadd/vmax/vmin指令压缩成一条指令可以处理的数据量后累加
#define REDUCE_TO_VEC_BIT_BY_BIT_MODE(opName, type, initVal, typeName)                   \
    SetVectorMask<typeName>(mask[1], mask[0]);                                           \
    __VEC_SCOPE__                                                                        \
    {                                                                                    \
        int32_t newRepeat = repeatTime;                                                 \
        uint32_t oneBlockNum = ONE_BLK_SIZE / B##type##_BYTE_SIZE;                       \
        uint32_t srcStrideOffset = srcRepStride * oneBlockNum;                           \
        __ubuf__ typeName *newSrc = src;                                                 \
        for (uint16_t i = 0; i < 1; i++) {                                               \
            MaskReg pgAll = CreatePredicate<typeName>();                                 \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                \
            RegTensor<typeName> vreg1;                                                   \
            Duplicate<typeName, typeName>(vreg1, initVal);                               \
            MaskReg preg = MovePredicate<typeName>();                                    \
            for (uint16_t i = 0; i < (uint16_t)newRepeat; ++i) {                         \
                RegTensor<typeName> vreg0;                                               \
                DataCopy<typeName, PostLiteral ::POST_MODE_UPDATE>(                      \
                    vreg0, newSrc, (uint32_t)DEFAULT_BLK_STRIDE, srcRepStride, preg);    \
                opName<typeName, Mode::MERGING>(vreg1, vreg0, vreg1, preg);              \
            }                                                                            \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, vreg1, offset0, pgAll); \
        }                                                                                \
    }

// L0 REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL
#define REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(reduceFuncName, instrName, typeName, vregType, type, initVal) \
    template <typename T = typeName>                                                                       \
    __aicore__ inline void reduceFuncName##Impl(__ubuf__ typeName *dst,                                    \
        __ubuf__ typeName *src,                                                                            \
        __ubuf__ typeName *work,                                                                           \
        const uint64_t mask[2],                                                                            \
        const int32_t repeatTime,                                                                         \
        const int32_t srcRepStride)                                                                        \
    {                                                                                                      \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                              \
        REDUCE_TO_VEC_BIT_BY_BIT_MODE(instrName, type, initVal, typeName);                                 \
        VEC_REDUCE_IMPL(Reduce##instrName, dst, work, type, elementNumPerInstr, typeName);                 \
    }

#define REDUCE_SUM_NO_INDEX_BIT_BY_BIT_MODE_IMPL(reduceFuncName, instrName, typeName, vregType, type, initVal) \
    template <typename T = typeName>                                                                       \
    __aicore__ inline void reduceFuncName##Impl(__ubuf__ typeName *dst,                                    \
        __ubuf__ typeName *src,                                                                            \
        __ubuf__ typeName *work,                                                                           \
        const uint64_t mask[2],                                                                            \
        const int32_t repeatTime,                                                                         \
        const int32_t srcRepStride)                                                                        \
    {                                                                                                      \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                              \
        REDUCE_TO_VEC_BIT_BY_BIT_MODE(instrName, type, initVal, typeName);                                 \
        VEC_REDUCE_SUM_IMPL(reduceFuncName, dst, work, type, elementNumPerInstr, typeName);                \
    }

template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceSumImpl is not supported!"); });
}

REDUCE_SUM_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceSum, Add, float, f32, 32, 0.0f);
REDUCE_SUM_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceSum, Add, uint32_t, u32, 32, 0);
REDUCE_SUM_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceSum, Add, int32_t, s32, 32, 0);
REDUCE_SUM_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceSum, Add, half, f16, 16, 0.0f);

template <typename T>
__aicore__ inline void ReduceMaxNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMaxNoIndex, Max, half, f16, 16, HALF_MIN)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMaxNoIndex, Max, float, f32, 32, FLOAT_MIN)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMaxNoIndex, Max, uint32_t, u32, 32, 0)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMaxNoIndex, Max, int32_t, s32, 32, INT32_MIN)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMaxNoIndex, Max, uint16_t, u16, 16, 0)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMaxNoIndex, Max, int16_t, s16, 16, INT16_MIN)

template <typename T>
__aicore__ inline void ReduceMinNoIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work, const uint64_t mask[2],
    const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, half, f16, 16, HALF_MIN)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, float, f32, 32, FLOAT_MIN)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, uint32_t, u32, 32, 0)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, int32_t, s32, 32, INT32_MIN)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, uint16_t, u16, 16, 0)
REDUCE_NO_INDEX_BIT_BY_BIT_MODE_IMPL(ReduceMinNoIndex, Min, int16_t, s16, 16, INT16_MIN)

// level 0 bit by bit with index
#define REDUCE_TO_VEC_WITH_INDEX_BIT_BY_BITS_MODE(                                                   \
    opName, cmpOp, type, offsetNum, initVal, typeName, indexType, initIndexType)                     \
    SetVectorMask<typeName>(mask[1], mask[0]);                                                       \
    __VEC_SCOPE__                                                                                    \
    {                                                                                                \
        int32_t newRepeat = repeatTime;                                                             \
        uint32_t oneBlockNum = ONE_BLK_SIZE / B##type##_BYTE_SIZE;                                   \
        uint32_t srcStrideOffset = srcRepStride * oneBlockNum;                                       \
        __ubuf__ typeName *newSrc = src;                                                             \
        for (uint16_t i = 0; i < 1; i++) {                                                           \
            MaskReg pgAll = CreatePredicate<typeName>();                                             \
            AddrReg offset0 = CreateAddrReg<typeName>(0);                                            \
            RegTensor<indexType> dstIndexVreg;                                                       \
            RegTensor<indexType> indexVreg;                                                          \
            RegTensor<initIndexType> initIndexVreg;                                                  \
            RegTensor<typeName> dstVreg;                                                             \
            Duplicate<indexType, indexType>(dstIndexVreg, 0);                                        \
            CreateVecIndex<initIndexType, Order::INC_ORDER_VALUE, indexType>(initIndexVreg, 0);      \
            Duplicate<typeName, typeName>(dstVreg, initVal);                                         \
            indexVreg = (RegTensor<indexType> &)initIndexVreg;                                       \
            MaskReg preg = MovePredicate<typeName>();                                                \
            for (uint16_t j = 0; j < (uint16_t)newRepeat; j++) {                                     \
                MaskReg pd;                                                                          \
                MaskReg indexPg;                                                                     \
                RegTensor<typeName> srcVreg;                                                         \
                DataCopy<typeName, PostLiteral ::POST_MODE_UPDATE>(                                  \
                    srcVreg, newSrc, (uint32_t)DEFAULT_BLK_STRIDE, srcRepStride, preg);              \
                Compare<typeName, cmpOp>(indexPg, srcVreg, dstVreg, preg);                           \
                Select<indexType>(dstIndexVreg, indexVreg, dstIndexVreg, indexPg);                   \
                opName<typeName, Mode::MERGING>(dstVreg, srcVreg, dstVreg, preg);                    \
                Adds<indexType, uint32_t, Mode::MERGING>(indexVreg, indexVreg, offsetNum, preg);     \
            }                                                                                        \
            DataCopy<typeName, DistVST::DIST_NORM_B##type>(work, dstVreg, offset0, pgAll);           \
            DataCopy<indexType, DistVST::DIST_NORM_B##type>(tempBuf, dstIndexVreg, offset0, pgAll);  \
        }                                                                                            \
    }

#define REDUCE_MAX_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(typeName, indexType, initIndexType, type, initVal, indexInitVal)   \
    template <typename T = typeName>                                                                              \
    __aicore__ inline void ReduceMaxWithIndexImpl(__ubuf__ typeName *dst,                                         \
        __ubuf__ typeName *src,                                                                                   \
        __ubuf__ typeName *work,                                                                                  \
        const uint64_t mask[2],                                                                                   \
        const int32_t repeatTime,                                                                                \
        const int32_t srcRepStride)                                                                               \
    {                                                                                                             \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                                     \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                                     \
        __ubuf__ indexType *tempBuf =                                                                             \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);                   \
        REDUCE_TO_VEC_WITH_INDEX_BIT_BY_BITS_MODE(Max, CMPMODE::GT, type, elementNumPerInstr, initVal, typeName, indexType, initIndexType)\
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMax, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType) \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                                    \
    }

#define REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(typeName, indexType, initIndexType, type, initVal, indexInitVal)   \
    template <typename T = typeName>                                                                              \
    __aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ typeName *dst,                                         \
        __ubuf__ typeName *src,                                                                                   \
        __ubuf__ typeName *work,                                                                                  \
        const uint64_t mask[2],                                                                                   \
        const int32_t repeatTime,                                                                                \
        const int32_t srcRepStride)                                                                               \
    {                                                                                                             \
        uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                                     \
        int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                                     \
        __ubuf__ indexType *tempBuf =                                                                             \
            AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);                   \
        REDUCE_TO_VEC_WITH_INDEX_BIT_BY_BITS_MODE(Min, CMPMODE::LT, type, elementNumPerInstr, initVal, typeName, indexType, initIndexType) \
        REDUCE_MAX_MIN_WITH_INDEX_IMPL(ReduceMin, type, elementNumPerInstr, indexInitVal, slideNum, typeName, indexType, initIndexType)           \
        AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                                    \
    }

template <typename T>
__aicore__ inline void ReduceMaxWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work,
    const uint64_t mask[2], const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxWithIndexImpl is not supported!"); });
}

REDUCE_MAX_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(half, uint16_t, int16_t, 16, HALF_MIN, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(float, uint32_t, int32_t, 32, FLOAT_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(uint32_t, uint32_t, int32_t, 32, 0, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(int32_t, uint32_t, int32_t, 32, INT32_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(uint16_t, uint16_t, int16_t, 16, 0, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(int16_t, uint16_t, int16_t, 16, INT16_MIN, UINT16_MAX)

template <typename T>
__aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ T *dst, __ubuf__ T *src, __ubuf__ T *work,
    const uint64_t mask[2], const int32_t repeatTime, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxWithIndexImpl is not supported!"); });
}

REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(half, uint16_t, int16_t, 16, HALF_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(float, uint32_t, int32_t, 32, FLOAT_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(uint32_t, uint32_t, int32_t, 32, UINT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(int32_t, uint32_t, int32_t, 32, INT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(uint16_t, uint16_t, int16_t, 16, UINT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL_BIT_BY_BITS_MODE(int16_t, uint16_t, int16_t, 16, INT16_MAX, UINT16_MAX)

template <typename T, bool isSetMask>
__aicore__ inline void RepeatReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, 
    const int32_t repeatTime, const int32_t elemsInOneRepeate, const int32_t dstBlkStride, const int32_t srcBlkStride,
    const int32_t dstRepStride, const int32_t srcRepStride)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "RepeatReduceSum is not supported!"); });
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue, uint32_t &maxMinIndex)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetReduceMaxMinCount is not supported!"); });
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(uint32_t &maxMinValue)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetReduceMaxMinCount is not supported!"); });
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T &maxMinValue, T &maxMinIndex)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetReduceMaxMinCount is not supported!"); });
}

template <typename T>
__aicore__ inline void GetReduceMaxMinCountImpl(T &maxMinValue)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetReduceMaxMinCount is not supported!"); });
}

template <typename T>
__aicore__ inline T GetAccValImpl()
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "GetAccVal is not supported!"); });
}

} // end namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H