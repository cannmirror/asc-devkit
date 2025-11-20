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
 * \brief AscendC l210 support reduce memory base api.
 */
#ifndef ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_REDUCE_IMPL_H
#include "kernel_utils.h"
#include "kernel_operator_common_impl.h"

namespace AscendC {
#define FLOAT_MAX  (__FLT_MAX__)
#define FLOAT_MIN  (-__FLT_MAX__)
#define HALF_MAX   (65504.0)
#define HALF_MIN   (-65504.0)

// 将所有输入数据用vadd/vmax/vmin指令压缩成一条指令可以处理的数据量
#define REDUCE_TO_VEC(opName, vregType, type, dataCount, offsetNum, initVal)       \
    __VEC_SCOPE__                                                                  \
    {                                                                              \
        for (uint16_t i = 0; i < 1; i++) {                                         \
            vector_bool pgAll = pge_b##type(PAT_ALL);                              \
            vector_address offset0 = vag_b##type(0);                               \
            vector_##vregType vreg1;                                               \
            vbr(vreg1, initVal);                                                   \
            for (uint16_t i = 0; i <= get_vloopn_bound_b##type(dataCount); ++i) {  \
                vector_bool preg = vpd_b##type();                                  \
                vector_address offset1 = vag_b##type(offsetNum);                   \
                vector_##vregType vreg0;                                           \
                vld(vreg0, src, offset1, NORM);                                    \
                v##opName(vreg1, vreg0, vreg1, preg);                              \
            }                                                                      \
            vst(vreg1, work, offset0, NORM_B##type, pgAll);                        \
        }                                                                          \
    }

#define VEC_REDUCE_IMPL(opName, dst, src, vregType, type, dataCount)               \
    __VEC_SCOPE__                                                                  \
    {                                                                              \
        for (uint16_t i = 0; i <= get_vloopn_bound_b##type(dataCount); i++) {      \
            vector_address offset0 = vag_b##type(0);                               \
            vector_bool preg = vpd_b##type();                                      \
            vector_bool preg1 = pge_b##type(PAT_VL1);                              \
            vector_##vregType dstVreg;                                             \
            vector_##vregType srcVreg;                                             \
            vld(srcVreg, src, offset0, NORM);                                      \
            v##opName(dstVreg, srcVreg, preg);                                     \
            vst(dstVreg, dst, offset0, NORM_B##type, preg1);                       \
        }                                                                          \
    }

#define REDUCE_NO_INDEX_IMPL(reduceFuncName, instrName, typeName, vregType, type, initVal)      \
template <>                                                                                     \
__aicore__ inline void reduceFuncName##Impl<typeName>(__ubuf__ typeName* dst,                   \
    __ubuf__ typeName* src, __ubuf__ typeName* work, const int32_t count)                       \
{                                                                                               \
    uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                       \
    REDUCE_TO_VEC(instrName, vregType, type, count, elementNumPerInstr, initVal);               \
    VEC_REDUCE_IMPL(c##instrName, dst, work, vregType, type, elementNumPerInstr);               \
}

template <typename T>
__aicore__ inline void ReduceSumImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* work, const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceSumImpl is not supported!"); });
}

// ReduceSumImpl::Level 2
REDUCE_NO_INDEX_IMPL(ReduceSum, add, float, f32, 32, 0.0f)
REDUCE_NO_INDEX_IMPL(ReduceSum, add, uint32_t, u32, 32, 0)
REDUCE_NO_INDEX_IMPL(ReduceSum, add, int32_t, s32, 32, 0)
REDUCE_NO_INDEX_IMPL(ReduceSum, add, half, f16, 16, 0.0f)

template <typename T>
__aicore__ inline void ReduceMaxNoIndexImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* work,
    const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, max, half, f16, 16, HALF_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, max, float, f32, 32, FLOAT_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, max, uint32_t, u32, 32, 0)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, max, int32_t, s32, 32, INT32_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, max, uint16_t, u16, 16, 0)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, max, int16_t, s16, 16, INT16_MIN)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, max, uint8_t, u8, 8, 0)
REDUCE_NO_INDEX_IMPL(ReduceMaxNoIndex, max, int8_t, s8, 8, INT8_MIN)

template <typename T>
__aicore__ inline void ReduceMinNoIndexImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* work,
    const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinNoIndexImpl is not supported!"); });
}

REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, min, half, f16, 16, HALF_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, min, float, f32, 32, FLOAT_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, min, uint32_t, u32, 32, UINT32_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, min, int32_t, s32, 32, INT32_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, min, uint16_t, u16, 16, UINT16_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, min, int16_t, s16, 16, INT16_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, min, uint8_t, u8, 8, UINT8_MAX)
REDUCE_NO_INDEX_IMPL(ReduceMinNoIndex, min, int8_t, s8, 8, INT8_MAX)

// vcmp_lt/vcmp_gt指令用于寻找在当前加载的数据中找到比之前数据更小/更大的数的下标
// vsel指令用于更新已加载的最大/最小值数据下标
#define REDUCE_TO_VEC_WITH_INDEX(opName, cmpOp, vregType, type, count, offsetNum, initVal)    \
    __VEC_SCOPE__ {                                                                           \
        for (uint16_t i = 0; i < 1; i++) {                                                    \
            vector_bool pgAll = pge_b##type(PAT_ALL);                                         \
            vector_address offset0 = vag_b##type(0);                                          \
            vector_u##type dstIndexVreg;                                                      \
            vector_u##type indexVreg;                                                         \
            vector_s##type initIndexVreg;                                                     \
            vector_##vregType dstVreg;                                                        \
            vbr(dstIndexVreg, 0);                                                             \
            vci(initIndexVreg, 0);                                                            \
            vbr(dstVreg, initVal);                                                            \
            indexVreg = vector_u##type(initIndexVreg);                                        \
            for (uint16_t j = 0; j <= get_vloopn_bound_b##type(count); j++) {                 \
                vector_address offset1 = vag_b##type(offsetNum);                              \
                vector_bool preg = vpd_b##type();                                             \
                vector_bool pd;                                                               \
                vector_bool indexPg;                                                          \
                vector_bool preg1 = pge_b##type(PAT_VL1);                                     \
                vector_##vregType srcVreg;                                                    \
                vld(srcVreg, src, offset1, NORM);                                             \
                vcmp_##cmpOp(indexPg, srcVreg, dstVreg, preg);                                \
                vsel(dstIndexVreg, indexVreg, dstIndexVreg, indexPg);                         \
                v##opName(dstVreg, srcVreg, dstVreg, preg);                                   \
                vadds(indexVreg, indexVreg, offsetNum, preg);                                 \
            }                                                                                 \
            vst(dstVreg, work, offset0, NORM_B##type, pgAll);                                 \
            vst(dstIndexVreg, tempBuf, offset0, NORM_B##type, pgAll);                         \
        }                                                                                     \
    }

#define REDUCE_MAX_MIN_WITH_INDEX_IMPL(opName, vregType, type, count, initVal, slideNum)         \
    __VEC_SCOPE__ {                                                                              \
        for (uint16_t i = 0; i <= get_vloopn_bound_b##type(count); i++) {                        \
            vector_address offset0 = vag_b##type(0);                                             \
            vector_bool preg = vpd_b##type();                                                    \
            vector_bool preg1 = pge_b##type(PAT_VL2);                                            \
            vector_bool pd;                                                                      \
            vector_u##type indexVreg, zeroVreg;                                                  \
            vector_u##type dstIndexVreg;                                                         \
            vector_##vregType srcVreg;                                                           \
            vector_##vregType dstVreg;                                                           \
            vbr(dstIndexVreg, initVal);                                                          \
            vbr(zeroVreg, 0);                                                                    \
            vld(srcVreg, work, offset0, NORM);                                                   \
            vld(indexVreg, tempBuf, offset0, NORM);                                              \
            vcb##opName(dstVreg, pd, srcVreg, preg);                                             \
            vsel(dstIndexVreg, indexVreg, dstIndexVreg, pd);                                     \
            vcmin(dstIndexVreg, dstIndexVreg, preg);                                             \
            vslide(dstVreg, (vector_##vregType)zeroVreg, dstVreg, 1);                            \
            vslide(dstVreg, dstVreg, (vector_##vregType)dstIndexVreg, slideNum);                 \
            vst(dstVreg, dst, offset0, NORM_B##type, preg1);                                     \
        }                                                                                        \
    }

#define REDUCE_MAX_WITH_INDEX_IMPL(typeName, indexType, vregType, type, initVal, indexInitVal)                         \
template <>                                                                                                            \
__aicore__ inline void ReduceMaxWithIndexImpl<typeName>(__ubuf__ typeName* dst, __ubuf__ typeName* src,                \
    __ubuf__ typeName* work, const int32_t count)                                                                      \
{                                                                                                                      \
    uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                                              \
    int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                                              \
    __ubuf__ indexType* tempBuf = AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);  \
    REDUCE_TO_VEC_WITH_INDEX(max, gt, vregType, type, count, elementNumPerInstr, initVal)                              \
    REDUCE_MAX_MIN_WITH_INDEX_IMPL(max, vregType, type, elementNumPerInstr, indexInitVal, slideNum)                    \
    AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                                             \
}

#define REDUCE_MIN_WITH_INDEX_IMPL(typeName, indexType, vregType, type, initVal, indexInitVal)                         \
template <>                                                                                                            \
__aicore__ inline void ReduceMinWithIndexImpl<typeName>(__ubuf__ typeName* dst, __ubuf__ typeName* src,                \
    __ubuf__ typeName* work, const int32_t count)                                                                      \
{                                                                                                                      \
    uint32_t elementNumPerInstr = VECTOR_REG_WIDTH / B##type##_BYTE_SIZE;                                              \
    int16_t slideNum = (int16_t)(elementNumPerInstr - 1);                                                              \
    __ubuf__ indexType* tempBuf = AscendCUtils::GetTemporaryBufferAddr<indexType>(TMP_UB_OFFSET, elementNumPerInstr);  \
    REDUCE_TO_VEC_WITH_INDEX(min, lt, vregType, type, count, elementNumPerInstr, initVal)                              \
    REDUCE_MAX_MIN_WITH_INDEX_IMPL(min, vregType, type, elementNumPerInstr, indexInitVal, slideNum)                    \
    AscendCUtils::FreeTemporaryBuffer<indexType>(tempBuf);                                                             \
}

template <typename T>
__aicore__ inline void ReduceMaxWithIndexImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* work,
    const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMaxWithIndexImpl is not supported!"); });
}

REDUCE_MAX_WITH_INDEX_IMPL(half, uint16_t, f16, 16, HALF_MIN, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(float, uint32_t, f32, 32, FLOAT_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(uint32_t, uint32_t, u32, 32, 0, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(int32_t, uint32_t, s32, 32, INT32_MIN, UINT32_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(uint16_t, uint16_t, u16, 16, 0, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(int16_t, uint16_t, s16, 16, INT16_MIN, UINT16_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(uint8_t, uint8_t, u8, 8, 0, UINT8_MAX)
REDUCE_MAX_WITH_INDEX_IMPL(int8_t, uint8_t, s8, 8, INT8_MIN, UINT8_MAX)

template <typename T>
__aicore__ inline void ReduceMinWithIndexImpl(__ubuf__ T* dst, __ubuf__ T* src, __ubuf__ T* work,
    const int32_t count)
{
    ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "ReduceMinWithIndexImpl is not supported!"); });
}

REDUCE_MIN_WITH_INDEX_IMPL(half, uint16_t, f16, 16, HALF_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(float, uint32_t, f32, 32, FLOAT_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(uint32_t, uint32_t, u32, 32, UINT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(int32_t, uint32_t, s32, 32, INT32_MAX, UINT32_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(uint16_t, uint16_t, u16, 16, UINT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(int16_t, uint16_t, s16, 16, INT16_MAX, UINT16_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(uint8_t, uint8_t, u8, 8, UINT8_MAX, UINT8_MAX)
REDUCE_MIN_WITH_INDEX_IMPL(int8_t, uint8_t, s8, 8, INT8_MAX, UINT8_MAX)
} // end namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_BINARY_IMPL_H
