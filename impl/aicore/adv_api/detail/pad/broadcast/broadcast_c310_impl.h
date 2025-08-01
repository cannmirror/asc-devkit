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
 * \file broadcast_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_PAD_BROADCAST_BROADCAST_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_PAD_BROADCAST_BROADCAST_C310_IMPL_H

#include "kernel_tensor.h"
#include "kernel_operator_intf.h"
#include "broadcast_gather_c310_impl.h"

namespace AscendC {
namespace BroadcastInternal {
template <typename T>
__aicore__ inline void E2bLoad(MicroAPI::RegTensor<T>& dstReg, __local_mem__ T* srcUb)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B16>(dstReg, srcUb);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_E2B_B32>(dstReg, srcUb);
    }
}

template <typename T>
__aicore__ inline void BrcLoad(MicroAPI::RegTensor<T>& dstReg, __local_mem__ T* srcUb)
{
    if constexpr (sizeof(T) == 2) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B16>(dstReg, srcUb);
    } else if constexpr (sizeof(T) == 4) {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B32>(dstReg, srcUb);
    } else {
        MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_BRC_B8>(dstReg, srcUb);
    }
}

template <typename T>
__aicore__ inline void BrcDuplicate(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint32_t dstSize)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t repeatTimes = CeilDivision(dstSize, VF_LEN);
    uint32_t sreg = dstSize;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        BrcLoad<T>(srcReg, srcUb);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pregCnt = MicroAPI::UpdateMask<T>(sreg);
            MicroAPI::DataCopy(dstUb + i * VF_LEN, srcReg, pregCnt);
        }
    }
}

template <typename T>
__aicore__ inline void GenLastGatherIndex(__local_mem__ T* indexUb, uint32_t size1, uint32_t offset)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::RegTensor<T> indexReg;
        MicroAPI::RegTensor<T> tmpReg;

        MicroAPI::Duplicate(indexReg, (T)size1, pregFull);
        MicroAPI::Arange(tmpReg, (T)offset);
        MicroAPI::Div(indexReg, tmpReg, indexReg, pregFull);

        MicroAPI::DataCopy(indexUb, indexReg, pregFull);
    }
}

template <typename T>
__aicore__ inline void GenNlastGatherIndex(__local_mem__ T* indexUb, uint32_t size1, uint32_t offset)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::RegTensor<T> indexReg;
        MicroAPI::RegTensor<T> tmpReg;
        MicroAPI::RegTensor<T> dstReg;

        MicroAPI::Duplicate(indexReg, (T)size1, pregFull);
        MicroAPI::Arange(tmpReg, (T)offset);
        MicroAPI::Div(dstReg, tmpReg, indexReg, pregFull);
        MicroAPI::Mul(dstReg, indexReg, dstReg, pregFull);
        MicroAPI::Sub(indexReg, tmpReg, dstReg, pregFull);

        MicroAPI::DataCopy(indexUb, indexReg, pregFull);
    }
}

template <typename T, typename IndexT>
__aicore__ inline void BrcLastGatherOne(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, __local_mem__ IndexT* indexUb, uint16_t size0, uint16_t size1)
{
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    uint32_t main = size0 * size1;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<IndexT>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::RegTensor<T> dummyReg;
        MicroAPI::RegTensor<IndexT> srcReg1;
        MicroAPI::RegTensor<IndexT> srcReg2;
        MicroAPI::RegTensor<IndexT> indexReg1;
        MicroAPI::RegTensor<IndexT> indexReg2;

        MicroAPI::DataCopy(indexReg1, indexUb);
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            MicroAPI::DataCopy(indexReg2, indexUb + VF_LEN_HALF);
        }
        uint32_t sreg = main;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            MicroAPI::DataCopyGather(srcReg1, srcUb, indexReg1, pregFull);
            MicroAPI::DataCopyGather(srcReg2, srcUb, indexReg2, pregFull);
            MicroAPI::DeInterleave(
                srcReg, dummyReg, (MicroAPI::RegTensor<T>&)srcReg1, (MicroAPI::RegTensor<T>&)srcReg2);
        } else {
            MicroAPI::DataCopyGather(srcReg, srcUb, indexReg1, pregCnt);
        }
        MicroAPI::DataCopy(dstUb, srcReg, pregCnt);
    }
}

template <typename T, typename IndexT>
__aicore__ inline void BrcLastGatherTwo(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, __local_mem__ IndexT* indexUb, uint16_t size0, uint16_t size1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    uint16_t factor = VF_LEN / size1;
    uint16_t repeatTimes = CeilDivision(size0, factor) - 1;
    uint32_t main = factor * size1;
    uint32_t mainBlock = main * repeatTimes;
    uint32_t offset = factor * repeatTimes;
    uint32_t tail = size0 * size1 - mainBlock;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<IndexT>();
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::RegTensor<T> dummyReg;
        MicroAPI::RegTensor<IndexT> indexReg1;
        MicroAPI::RegTensor<IndexT> indexReg2;
        MicroAPI::RegTensor<IndexT> factorReg;
        MicroAPI::RegTensor<IndexT> srcReg1;
        MicroAPI::RegTensor<IndexT> srcReg2;
        MicroAPI::RegTensor<IndexT> dstReg;
        MicroAPI::RegTensor<IndexT> tmpReg;
        MicroAPI::UnalignReg ureg0;

        MicroAPI::Duplicate(factorReg, (IndexT)factor, pregFull);
        MicroAPI::DataCopy(indexReg1, indexUb);
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            MicroAPI::DataCopy(indexReg2, indexUb + VF_LEN_HALF);
        }
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            MicroAPI::Muls(tmpReg, factorReg, (IndexT)i, pregFull);
            MicroAPI::Add(dstReg, tmpReg, indexReg1, pregFull);
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                MicroAPI::DataCopyGather(srcReg1, srcUb, dstReg, pregFull);
                MicroAPI::Add(dstReg, tmpReg, indexReg2, pregFull);
                MicroAPI::DataCopyGather(srcReg2, srcUb, dstReg, pregFull);
                MicroAPI::DeInterleave(
                    srcReg, dummyReg, (MicroAPI::RegTensor<T>&)srcReg1, (MicroAPI::RegTensor<T>&)srcReg2);
            } else {
                MicroAPI::DataCopyGather(srcReg, srcUb, dstReg, pregFull);
            }
            MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, main);
        }
        MicroAPI::Adds(dstReg, indexReg1, (IndexT)offset, pregFull);
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            MicroAPI::DataCopyGather(srcReg1, srcUb, dstReg, pregFull);
            MicroAPI::Adds(dstReg, indexReg2, (IndexT)offset, pregFull);
            MicroAPI::DataCopyGather(srcReg2, srcUb, dstReg, pregFull);
            MicroAPI::DeInterleave(
                srcReg, dummyReg, (MicroAPI::RegTensor<T>&)srcReg1, (MicroAPI::RegTensor<T>&)srcReg2);
        } else {
            MicroAPI::DataCopyGather(srcReg, srcUb, dstReg, pregFull);
        }
        MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, tail);
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}

template <typename T, typename IndexT>
__aicore__ inline void BrcNlastGatherOne(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, __local_mem__ IndexT* indexUb, uint16_t size0, uint16_t size1)
{
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    uint32_t main = size0 * size1;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<IndexT>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<IndexT> indexReg1;
        MicroAPI::RegTensor<IndexT> indexReg2;
        MicroAPI::RegTensor<IndexT> srcReg1;
        MicroAPI::RegTensor<IndexT> srcReg2;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::RegTensor<T> dummyReg;
        MicroAPI::UnalignReg ureg0;

        uint32_t sreg = main;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(indexReg1, indexUb);
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            MicroAPI::DataCopy(indexReg2, indexUb + VF_LEN_HALF);
            MicroAPI::DataCopyGather(srcReg1, srcUb, indexReg1, pregFull);
            MicroAPI::DataCopyGather(srcReg2, srcUb, indexReg2, pregFull);
            MicroAPI::DeInterleave(
                srcReg, dummyReg, (MicroAPI::RegTensor<T>&)srcReg1, (MicroAPI::RegTensor<T>&)srcReg2);
        } else {
            MicroAPI::DataCopyGather(srcReg, srcUb, indexReg1, pregCnt);
        }
        MicroAPI::DataCopy(dstUb, srcReg, pregCnt);
    }
}

template <typename T, typename IndexT>
__aicore__ inline void BrcNlastGatherTwo(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, __local_mem__ IndexT* indexUb, uint16_t size0, uint16_t size1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    uint16_t factor = VF_LEN / size1;
    uint16_t repeatTimes = CeilDivision(size0, factor) - 1;
    uint32_t main = factor * size1;
    uint32_t mainBlock = main * repeatTimes;
    uint32_t tail = size0 * size1 - mainBlock;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<IndexT>();
        MicroAPI::RegTensor<IndexT> indexReg1;
        MicroAPI::RegTensor<IndexT> indexReg2;
        MicroAPI::RegTensor<IndexT> srcReg1;
        MicroAPI::RegTensor<IndexT> srcReg2;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::RegTensor<T> dummyReg;
        MicroAPI::UnalignReg ureg0;

        MicroAPI::DataCopy(indexReg1, indexUb);
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            MicroAPI::DataCopy(indexReg2, indexUb + VF_LEN_HALF);
            MicroAPI::DataCopyGather(srcReg1, srcUb, indexReg1, pregFull);
            MicroAPI::DataCopyGather(srcReg2, srcUb, indexReg2, pregFull);
            MicroAPI::DeInterleave(
                srcReg, dummyReg, (MicroAPI::RegTensor<T>&)srcReg1, (MicroAPI::RegTensor<T>&)srcReg2);
        } else {
            MicroAPI::DataCopyGather(srcReg, srcUb, indexReg1, pregFull);
        }
        for (uint16_t i = 0; i < repeatTimes; ++i) { MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, main); }
        MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, tail);
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}

template <typename T>
__aicore__ inline void BrcLastE2B(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = VF_LEN / size1;
    uint16_t repeatTimes = CeilDivision(size0, factor);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;

        uint32_t sreg = size0 * size1;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            pregCnt = MicroAPI::UpdateMask<T>(sreg);
            E2bLoad<T>(srcReg, srcUb + i * DEFAULT_BLK_NUM);
            MicroAPI::DataCopy(dstUb + i * VF_LEN, srcReg, pregCnt);
        }
    }
}

template <typename T>
__aicore__ inline void BrcLastE2BLargerThanVL(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1, uint16_t size2, uint16_t srcStride0)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = VF_LEN / size2;
    uint16_t repeatTimes = CeilDivision(size1, factor);
    uint32_t preg = size1 * size2;
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;

        for (uint16_t i = 0; i < size0; ++i) {
            sreg = preg;
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                pregCnt = MicroAPI::UpdateMask<T>(sreg);
                E2bLoad<T>(srcReg, srcUb + j * DEFAULT_BLK_NUM + i * srcStride0);
                MicroAPI::DataCopy(dstUb + i * size1 * size2 + j * VF_LEN, srcReg, pregCnt);
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcLastE2BLessThanVL(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1, uint16_t size2, uint16_t srcStride0)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint32_t preg = size1 * size2;
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        sreg = preg;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            E2bLoad<T>(srcReg, srcUb + i * srcStride0);
            MicroAPI::DataCopy(dstUb + i * size1 * size2, srcReg, pregCnt);
        }
    }
}

template <typename T>
__aicore__ inline void BrcLastE2B(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1,
    uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = VF_LEN / size3;
    uint16_t repeatTimes = CeilDivision(size2, factor);
    uint32_t preg = size2 * size3;
    uint32_t sreg;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size1; ++j) {
                sreg = preg;
                for (uint16_t k = 0; k < repeatTimes; ++k) {
                    pregCnt = MicroAPI::UpdateMask<T>(sreg);
                    E2bLoad<T>(srcReg, srcUb + i * srcStride0 + j * srcStride1 + k * DEFAULT_BLK_NUM);
                    MicroAPI::DataCopy(
                        dstUb + i * size1 * size2 * size3 + j * size2 * size3 + k * VF_LEN, srcReg, pregCnt);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcNlastGatherBOne(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, __local_mem__ uint32_t* indexUb, uint16_t size0, uint16_t size1)
{
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    uint32_t main = size0 * size1;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::RegTensor<uint32_t> indexReg;

        MicroAPI::DataCopy(indexReg, indexUb);
        MicroAPI::DataCopyGatherB(srcReg, srcUb, indexReg, pregFull);
        uint32_t sreg = main;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(dstUb, srcReg, pregCnt);
    }
}

template <typename T>
__aicore__ inline void BrcNlastGatherBTwo(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, __local_mem__ uint32_t* indexUb, uint16_t size0, uint16_t size1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    uint16_t factor = VF_LEN / size1;
    uint16_t repeatTimes = CeilDivision(size0, factor) - 1;
    uint32_t main = factor * size1;
    uint32_t mainBlock = main * repeatTimes;
    uint32_t tail = size0 * size1 - mainBlock;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::RegTensor<uint32_t> indexReg;

        MicroAPI::DataCopy(indexReg, indexUb);
        MicroAPI::DataCopyGatherB(srcReg, srcUb, indexReg, pregFull);
        uint32_t sreg = main;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < repeatTimes; ++i) { MicroAPI::DataCopy(dstUb + i * main, srcReg, pregCnt); }
        sreg = tail;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(dstUb + mainBlock, srcReg, pregCnt);
    }
}

template <typename T>
__aicore__ inline void BrcLastLessThanVLAligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t srcStride0, uint16_t srcStride1)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;

        uint32_t sreg = size2;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size1; ++i) {
            for (uint16_t j = 0; j < size0; ++j) {
                BrcLoad<T>(srcReg, srcUb + j * srcStride0 + i * srcStride1);
                MicroAPI::DataCopy(dstUb + j * size1 * size2 + i * size2, srcReg, pregCnt);
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcLastLessThanVLAligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1, uint16_t srcStride2)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;

        uint32_t sreg = size3;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size2; ++j) {
                for (uint16_t k = 0; k < size1; ++k) {
                    BrcLoad<T>(srcReg, srcUb + i * srcStride0 + j * srcStride2 + k * srcStride1);
                    MicroAPI::DataCopy(
                        dstUb + i * size1 * size2 * size3 + k * size2 * size3 + j * size3, srcReg, pregCnt);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcNlastLessThanVLAligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1, uint16_t srcStride2)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;

        uint32_t sreg = size3;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size1; ++i) {
            for (uint16_t j = 0; j < size0; ++j) {
                for (uint16_t k = 0; k < size2; ++k) {
                    MicroAPI::DataCopy(srcReg, srcUb + i * srcStride1 + j * srcStride0 + k * srcStride2);
                    MicroAPI::DataCopy(
                        dstUb + j * size1 * size2 * size3 + i * size2 * size3 + k * size3, srcReg, pregCnt);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcLastLessThanVLUnaligned(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0;

        uint32_t sreg = size1;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            BrcLoad<T>(srcReg, srcUb + i);
            MicroAPI::Duplicate(srcReg, srcReg, pregCnt);
            MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, size1);
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}

template <typename T>
__aicore__ inline void BrcLastLessThanVLUnaligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t srcStride0, uint16_t srcStride1)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0;

        uint32_t sreg = size2;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size1; ++j) {
                BrcLoad<T>(srcReg, srcUb + j * srcStride1 + i * srcStride0);
                MicroAPI::Duplicate(srcReg, srcReg, pregCnt);
                MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, size2);
            }
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}

template <typename T>
__aicore__ inline void BrcLastLessThanVLUnaligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1, uint16_t srcStride2)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0;

        uint32_t sreg = size3;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size1; ++j) {
                for (uint16_t k = 0; k < size2; ++k) {
                    BrcLoad<T>(srcReg, srcUb + i * srcStride0 + j * srcStride1 + k * srcStride2);
                    MicroAPI::Duplicate(srcReg, srcReg, pregCnt);
                    MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, size3);
                }
            }
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}
template <typename T>
__aicore__ inline void BrcNlastLessThanVLUnaligned(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0;

        uint32_t sreg = size1;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        MicroAPI::DataCopy(srcReg, srcUb);
        for (uint16_t i = 0; i < size0; ++i) { MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, size1); }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}
template <typename T>
__aicore__ inline void BrcNlastLessThanVLUnaligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t srcStride0, uint16_t srcStride1)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0, ureg1;

        uint32_t sreg = size2;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size1; ++j) {
                auto srcUbT = srcUb + i * srcStride0 + j * srcStride1;
                MicroAPI::DataCopyUnAlignPre(ureg0, srcUbT);
                MicroAPI::DataCopyUnAlign(srcReg, ureg0, srcUbT, size2);
                MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg1, size2);
            }
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg1, 0);
    }
}

template <typename T>
__aicore__ inline void BrcNlastLessThanVLUnaligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1, uint16_t srcStride2)
{
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0, ureg1;

        uint32_t sreg = size3;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size1; ++j) {
                for (uint16_t k = 0; k < size2; ++k) {
                    __local_mem__ T* srcUbTmp = srcUb + i * srcStride0 + j * srcStride1 + k * srcStride2;
                    MicroAPI::DataCopyUnAlignPre(ureg0, srcUbTmp);
                    MicroAPI::DataCopyUnAlign(srcReg, ureg0, srcUbTmp, size3);
                    MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg1, size3);
                }
            }
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg1, 0);
    }
}

template <typename T>
__aicore__ inline void BrcLastLargerThanVLAligned(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = CeilDivision(size1, VF_LEN);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;

        for (uint16_t i = 0; i < size0; ++i) {
            BrcLoad<T>(srcReg, srcUb + i);
            uint32_t sreg = size1;
            for (uint16_t j = 0; j < factor; ++j) {
                pregCnt = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::DataCopy(dstUb + i * size1 + j * VF_LEN, srcReg, pregCnt);
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcLastLargerThanVLAligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t srcStride0, uint16_t srcStride1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = CeilDivision(size2, VF_LEN);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;

        for (uint16_t i = 0; i < size1; ++i) {
            for (uint16_t j = 0; j < size0; ++j) {
                BrcLoad<T>(srcReg, srcUb + i * srcStride1 + j * srcStride0);
                uint32_t sreg = size2;
                for (uint16_t k = 0; k < factor; ++k) {
                    pregCnt = MicroAPI::UpdateMask<T>(sreg);
                    MicroAPI::DataCopy(dstUb + j * size1 * size2 + i * size2 + k * VF_LEN, srcReg, pregCnt);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcLastLargerThanVLAligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1, uint16_t srcStride2)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = CeilDivision(size3, VF_LEN);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size2; ++j) {
                uint32_t sreg = size3;
                for (uint16_t k = 0; k < factor; ++k) {
                    pregCnt = MicroAPI::UpdateMask<T>(sreg);
                    for (uint16_t t = 0; t < size1; ++t) {
                        BrcLoad<T>(srcReg, srcUb + i * srcStride0 + j * srcStride2 + t * srcStride1);
                        MicroAPI::DataCopy(
                            dstUb + i * size1 * size2 * size3 + t * size2 * size3 + j * size3 + k * VF_LEN, srcReg,
                            pregCnt);
                    }
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcNlastLargerThanVLAlignedWithBlock(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = CeilDivision(size1, VF_LEN);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;

        for (uint16_t i = 0; i < size0; ++i) {
            uint32_t sreg = size1;
            for (uint16_t j = 0; j < factor; ++j) {
                pregCnt = MicroAPI::UpdateMask<T>(sreg);
                MicroAPI::DataCopy(srcReg, srcUb + j * VF_LEN);
                MicroAPI::DataCopy(dstUb + i * size1 + j * VF_LEN, srcReg, pregCnt);
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcNlastLargerThanVLAlignedWithBlock(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    uint16_t size0, uint16_t size1, uint16_t size2, uint16_t srcStride0, uint16_t srcStride1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = CeilDivision(size2, VF_LEN);
    uint16_t jStride = srcStride1 == 0 ? 0 : VF_LEN;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;

        for (uint16_t i = 0; i < size0; ++i) {
            uint32_t sreg = size2;
            for (uint16_t j = 0; j < factor; ++j) {
                pregCnt = MicroAPI::UpdateMask<T>(sreg);
                for (uint16_t k = 0; k < size1; ++k) {
                    MicroAPI::DataCopy(srcReg, srcUb + k * srcStride1 + i * srcStride0 + j * VF_LEN);
                    MicroAPI::DataCopy(dstUb + i * size1 * size2 + k * size2 + j * VF_LEN, srcReg, pregCnt);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcNlastLargerThanVLAlignedWithBlock(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    uint16_t size0, uint16_t size1, uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1,
    uint16_t srcStride2)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = CeilDivision(size3, VF_LEN);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        for (uint16_t i = 0; i < size1; ++i) {
            uint32_t sreg = size3;
            for (uint16_t j = 0; j < factor; ++j) {
                pregCnt = MicroAPI::UpdateMask<T>(sreg);
                for (uint16_t k = 0; k < size0; ++k) {
                    for (uint16_t t = 0; t < size2; ++t) {
                        MicroAPI::DataCopy(
                            srcReg, srcUb + j * VF_LEN + i * srcStride1 + k * srcStride0 + t * srcStride2);
                        MicroAPI::DataCopy(
                            dstUb + k * size1 * size2 * size3 + i * size2 * size3 + t * size3 + j * VF_LEN, srcReg,
                            pregCnt);
                    }
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcNlastLargerThanVLAlignedWithVL(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = CeilDivision(size1, VF_LEN);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::RegTensor<T> srcReg;

        for (uint16_t i = 0; i < factor; ++i) {
            MicroAPI::DataCopy(srcReg, srcUb + i * VF_LEN);
            for (uint16_t j = 0; j < size0; ++j) {
                MicroAPI::DataCopy(dstUb + i * VF_LEN + j * size1, srcReg, pregFull);
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcNlastLargerThanVLAlignedWithVL(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t srcStride0, uint16_t srcStride1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = CeilDivision(size2, VF_LEN);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::RegTensor<T> srcReg;

        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < factor; ++j) {
                for (uint16_t k = 0; k < size1; ++k) {
                    MicroAPI::DataCopy(srcReg, srcUb + i * srcStride0 + j * VF_LEN + k * srcStride1);
                    MicroAPI::DataCopy(dstUb + j * VF_LEN + k * size2 + i * size1 * size2, srcReg, pregFull);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcNlastLargerThanVLAlignedWithVL(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1, uint16_t srcStride2)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = CeilDivision(size3, VF_LEN);

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T>();
        MicroAPI::RegTensor<T> srcReg;

        for (uint16_t i = 0; i < size1; ++i) {
            for (uint16_t j = 0; j < factor; ++j) {
                for (uint16_t k = 0; k < size0; ++k) {
                    for (uint16_t t = 0; t < size2; ++t) {
                        MicroAPI::DataCopy(
                            srcReg, srcUb + i * srcStride1 + j * VF_LEN + k * srcStride0 + t * srcStride2);
                        MicroAPI::DataCopy(
                            dstUb + j * VF_LEN + t * size3 + i * size2 * size3 + k * size1 * size2 * size3, srcReg,
                            pregFull);
                    }
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcLastLargerThanVLUnaligned(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = size1 / VF_LEN;
    uint32_t size1tail = size1 - factor * VF_LEN;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0;

        uint32_t sreg = size1tail;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            BrcLoad<T>(srcReg, srcUb + i);
            for (uint16_t j = 0; j < factor; ++j) { MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, VF_LEN); }
            MicroAPI::Duplicate(srcReg, srcReg, pregCnt);
            MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, size1tail);
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}

template <typename T>
__aicore__ inline void BrcLastLargerThanVLUnaligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t srcStride0, uint16_t srcStride1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = size2 / VF_LEN;
    uint32_t size2tail = size2 - factor * VF_LEN;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0;

        uint32_t sreg = size2tail;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size1; ++j) {
                BrcLoad<T>(srcReg, srcUb + j * srcStride1 + i * srcStride0);
                for (uint16_t k = 0; k < factor; ++k) { MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, VF_LEN); }
                MicroAPI::Duplicate(srcReg, srcReg, pregCnt);
                MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, size2tail);
            }
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}

template <typename T>
__aicore__ inline void BrcLastLargerThanVLUnaligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1, uint16_t srcStride2)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = size3 / VF_LEN;
    uint32_t size3tail = size3 - factor * VF_LEN;

    __VEC_SCOPE__
    {
        MicroAPI::MaskReg pregCnt;
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0;

        uint32_t sreg = size3tail;
        pregCnt = MicroAPI::UpdateMask<T>(sreg);
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size1; ++j) {
                for (uint16_t k = 0; k < size2; ++k) {
                    BrcLoad<T>(srcReg, srcUb + i * srcStride0 + j * srcStride1 + k * srcStride2);
                    for (uint16_t t = 0; t < factor; ++t) { MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, VF_LEN); }
                    MicroAPI::Duplicate(srcReg, srcReg, pregCnt);
                    MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, size3tail);
                }
            }
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}

template <typename T>
__aicore__ inline void BrcNlastLargerThanVLUnaligned(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0, uint16_t size1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = size1 / VF_LEN;
    uint32_t size1tail = size1 - factor * VF_LEN;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::RegTensor<T> tmpReg;
        MicroAPI::UnalignReg ureg0;

        uint32_t sreg = size1tail;
        MicroAPI::DataCopy(tmpReg, srcUb + factor * VF_LEN);
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < factor; ++j) {
                MicroAPI::DataCopy(srcReg, srcUb + j * VF_LEN);
                MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg0, VF_LEN);
            }
            MicroAPI::DataCopyUnAlign(dstUb, tmpReg, ureg0, size1tail);
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg0, 0);
    }
}

template <typename T>
__aicore__ inline void BrcNlastLargerThanVLUnaligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t srcStride0, uint16_t srcStride1)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = size2 / VF_LEN;
    uint32_t size2tail = size2 - factor * VF_LEN;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0, ureg1;

        uint32_t sreg = size2tail;
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size1; ++j) {
                __local_mem__ T* tmpSrcUb = srcUb + i * srcStride0 + j * srcStride1;
                MicroAPI::DataCopyUnAlignPre(ureg0, tmpSrcUb);
                for (uint16_t k = 0; k < factor; ++k) {
                    MicroAPI::DataCopyUnAlign(srcReg, ureg0, tmpSrcUb, VF_LEN);
                    MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg1, VF_LEN);
                }
                MicroAPI::DataCopyUnAlign(srcReg, ureg0, tmpSrcUb, sreg);
                MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg1, sreg);
            }
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg1, 0);
    }
}

template <typename T>
__aicore__ inline void BrcNlastLargerThanVLUnaligned(__local_mem__ T* dstUb, __local_mem__ T* srcUb, uint16_t size0,
    uint16_t size1, uint16_t size2, uint16_t size3, uint16_t srcStride0, uint16_t srcStride1, uint16_t srcStride2)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    uint16_t factor = size3 / VF_LEN;
    uint32_t size3tail = size3 - factor * VF_LEN;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> srcReg;
        MicroAPI::UnalignReg ureg0, ureg1;

        uint32_t sreg = size3tail;
        for (uint16_t i = 0; i < size0; ++i) {
            for (uint16_t j = 0; j < size1; ++j) {
                for (uint16_t k = 0; k < size2; ++k) {
                    __local_mem__ T* tmpSrcUb = srcUb + i * srcStride0 + j * srcStride1 + k * srcStride2;
                    MicroAPI::DataCopyUnAlignPre(ureg0, tmpSrcUb);
                    for (uint16_t t = 0; t < factor; ++t) {
                        MicroAPI::DataCopyUnAlign(srcReg, ureg0, tmpSrcUb, VF_LEN);
                        MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg1, VF_LEN);
                    }
                    MicroAPI::DataCopyUnAlign(srcReg, ureg0, tmpSrcUb, sreg);
                    MicroAPI::DataCopyUnAlign(dstUb, srcReg, ureg1, sreg);
                }
            }
        }
        MicroAPI::DataCopyUnAlignPost(dstUb, ureg1, 0);
    }
}

template <typename T, int32_t constRank = -1>
__aicore__ inline bool BrcLastWrapperForTwoDim(__local_mem__ T* dstUb, __local_mem__ T* srcUb, const uint32_t* dstShape)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    using GatherIndexType = typename ExtractSignedTypeBySize<sizeof(T)>::T;
    using BrcIndexType = typename ExtractIndexTypeBySize<sizeof(T)>::T;

    uint16_t sizeI[2];
    sizeI[0] = static_cast<uint16_t>(dstShape[0]);
    sizeI[1] = static_cast<uint16_t>(dstShape[1]);

    if (sizeI[1] == oneBlockElementNum && sizeof(T) != sizeof(uint8_t)) {
        BrcLastE2B(dstUb, srcUb, sizeI[0], sizeI[1]);
    } else if (sizeI[1] < VF_LEN_HALF) {
        LocalTensor<T> indexLocal;
        PopStackBuffer<T, TPosition::LCM>(indexLocal);
        __local_mem__ GatherIndexType* indexUb1 = (__local_mem__ GatherIndexType*)indexLocal.GetPhyAddr();
        __local_mem__ GatherIndexType* indexUb2 = (__local_mem__ GatherIndexType*)indexLocal.GetPhyAddr(VF_LEN);
        GenLastGatherIndex<GatherIndexType>(indexUb1, sizeI[1], 0);
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            GenLastGatherIndex<GatherIndexType>(indexUb2, sizeI[1], VF_LEN_HALF);
        }
        __local_mem__ BrcIndexType* indexUb = (__local_mem__ BrcIndexType*)indexLocal.GetPhyAddr();
        if (sizeI[0] * sizeI[1] < VF_LEN) {
            BrcLastGatherOne<T, BrcIndexType>(dstUb, srcUb, indexUb, sizeI[0], sizeI[1]);
        } else if (sizeI[1] < VF_LEN) {
            BrcLastGatherTwo<T, BrcIndexType>(dstUb, srcUb, indexUb, sizeI[0], sizeI[1]);
        }
    } else if (sizeI[1] <= VF_LEN) {
        BrcLastLessThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1]);
    } else {
        if (sizeI[1] % oneBlockElementNum == 0) {
            BrcLastLargerThanVLAligned<T>(dstUb, srcUb, sizeI[0], sizeI[1]);
        } else {
            if constexpr (constRank == -1) {
                return true;
            } else {
                BrcLastLargerThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1]);
            }
        }
    }
    return false;
}

template <typename T, int32_t constRank = -1>
__aicore__ inline bool BrcLastWrapperForThreeDim(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, const uint32_t* dstShape, const uint32_t* srcStride)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    uint16_t sizeI[3];
    uint16_t stride[3];
    sizeI[0] = static_cast<uint16_t>(dstShape[0]);
    sizeI[1] = static_cast<uint16_t>(dstShape[1]);
    sizeI[2] = static_cast<uint16_t>(dstShape[2]);
    stride[0] = static_cast<uint16_t>(srcStride[0]);
    stride[1] = static_cast<uint16_t>(srcStride[1]);
    stride[2] = static_cast<uint16_t>(srcStride[2]);

    if (sizeI[2] == oneBlockElementNum && sizeof(T) != sizeof(uint8_t) && sizeI[1] * sizeI[2] > VF_LEN_HALF
        && sizeI[1] % DEFAULT_BLK_NUM == 0 && stride[1] != 0) {
        if (sizeI[1] * sizeI[2] > VF_LEN) {
            BrcLastE2BLargerThanVL(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0]);
        } else {
            BrcLastE2BLessThanVL(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0]);
        }
    } else if (sizeI[2] < VF_LEN_HALF && sizeof(T) != sizeof(uint8_t)) {
        uint32_t newDstShape[3] = {dstShape[0], dstShape[1], dstShape[2]};
        uint32_t newSrcStride[3] = {srcStride[0], srcStride[1], srcStride[2]};
        GatherWrapper(dstUb, srcUb, newDstShape, newSrcStride);
    } else if (sizeI[2] <= VF_LEN) {
        if (sizeI[2] % oneBlockElementNum == 0) {
            BrcLastLessThanVLAligned<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0], stride[1]);
        } else {
            if constexpr (constRank == -1) {
                return true;
            } else {
                BrcLastLessThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0], stride[1]);
            }
        }
    } else {
        if (sizeI[2] % oneBlockElementNum == 0) {
            BrcLastLargerThanVLAligned<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0], stride[1]);
        } else {
            if constexpr (constRank == -1) {
                return true;
            } else {
                BrcLastLargerThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0], stride[1]);
            }
        }
    }
    return false;
}

template <typename T, int32_t constRank = -1>
__aicore__ inline bool BrcLastWrapperForFourDim(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, const uint32_t* dstShape, const uint32_t* srcStride)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    uint16_t sizeI[4];
    uint16_t stride[4];
    sizeI[0] = static_cast<uint16_t>(dstShape[0]);
    sizeI[1] = static_cast<uint16_t>(dstShape[1]);
    sizeI[2] = static_cast<uint16_t>(dstShape[2]);
    sizeI[3] = static_cast<uint16_t>(dstShape[3]);
    stride[0] = static_cast<uint16_t>(srcStride[0]);
    stride[1] = static_cast<uint16_t>(srcStride[1]);
    stride[2] = static_cast<uint16_t>(srcStride[2]);
    stride[3] = static_cast<uint16_t>(srcStride[3]);

    if (sizeI[3] == oneBlockElementNum && sizeof(T) != sizeof(uint8_t) && stride[2] != 0
        && sizeI[2] % DEFAULT_BLK_NUM == 0) {
        BrcLastE2B(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1]);
    } else if (sizeI[3] < VF_LEN_HALF && sizeof(T) != sizeof(uint8_t)) {
        uint32_t newDstShape[4] = {dstShape[0], dstShape[1], dstShape[2], dstShape[3]};
        uint32_t newSrcStride[4] = {srcStride[0], srcStride[1], srcStride[2], srcStride[3]};
        GatherWrapperForFourDim(dstUb, srcUb, newDstShape, newSrcStride);
    } else if (sizeI[3] <= VF_LEN) {
        if (sizeI[3] % oneBlockElementNum == 0) {
            BrcLastLessThanVLAligned<T>(
                dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
        } else {
            if constexpr (constRank == -1) {
                return true;
            } else {
                BrcLastLessThanVLUnaligned<T>(
                    dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
            }
        }
    } else {
        if (sizeI[3] % oneBlockElementNum == 0) {
            BrcLastLargerThanVLAligned<T>(
                dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
        } else {
            if constexpr (constRank == -1) {
                return true;
            } else {
                BrcLastLargerThanVLUnaligned<T>(
                    dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
            }
        }
    }
    return false;
}

template <typename T, int32_t constRank = -1>
__aicore__ inline bool BrcNlastWrapperForTwoDim(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, const uint32_t* dstShape)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    using GatherIndexType = typename ExtractSignedTypeBySize<sizeof(T)>::T;
    using BrcIndexType = typename ExtractIndexTypeBySize<sizeof(T)>::T;
    uint16_t sizeI[2];
    sizeI[0] = static_cast<uint16_t>(dstShape[0]);
    sizeI[1] = static_cast<uint16_t>(dstShape[1]);

    if (sizeI[1] < VF_LEN_HALF) {
        LocalTensor<T> indexLocal;
        PopStackBuffer<T, TPosition::LCM>(indexLocal);
        if (sizeI[1] % oneBlockElementNum == 0) {
            __local_mem__ uint32_t* indexUb = (__local_mem__ uint32_t*)indexLocal.GetPhyAddr();
            if (sizeI[1] / oneBlockElementNum == 1) {
                indexUb[0] = 0;
                indexUb[1] = 0;
                indexUb[2] = 0;
                indexUb[3] = 0;
                indexUb[4] = 0;
                indexUb[5] = 0;
                indexUb[6] = 0;
                indexUb[7] = 0;
            } else if (sizeI[1] / oneBlockElementNum == 2) {
                indexUb[0] = 0;
                indexUb[1] = 32;
                indexUb[2] = 0;
                indexUb[3] = 32;
                indexUb[4] = 0;
                indexUb[5] = 32;
                indexUb[6] = 0;
                indexUb[7] = 32;
            } else {
                indexUb[0] = 0;
                indexUb[1] = 32;
                indexUb[2] = 64;
                indexUb[3] = 0;
                indexUb[4] = 32;
                indexUb[5] = 64;
                indexUb[6] = 0;
                indexUb[7] = 0;
            }
            event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIdSToV);
            WaitFlag<HardEvent::S_V>(eventIdSToV);
            if (sizeI[0] * sizeI[1] < VF_LEN) {
                BrcNlastGatherBOne<T>(dstUb, srcUb, (__local_mem__ uint32_t*)indexUb, sizeI[0], sizeI[1]);
            } else if (sizeI[1] < VF_LEN) {
                BrcNlastGatherBTwo<T>(dstUb, srcUb, (__local_mem__ uint32_t*)indexUb, sizeI[0], sizeI[1]);
            }
        } else {
            __local_mem__ GatherIndexType* indexUb1 = (__local_mem__ GatherIndexType*)indexLocal.GetPhyAddr();
            __local_mem__ GatherIndexType* indexUb2 = (__local_mem__ GatherIndexType*)indexLocal.GetPhyAddr(VF_LEN);
            GenNlastGatherIndex<GatherIndexType>(indexUb1, sizeI[1], 0);
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                GenNlastGatherIndex<GatherIndexType>(indexUb2, sizeI[1], VF_LEN_HALF);
            }
            __local_mem__ BrcIndexType* indexUb = (__local_mem__ BrcIndexType*)indexLocal.GetPhyAddr();
            if (sizeI[0] * sizeI[1] < VF_LEN) {
                BrcNlastGatherOne<T, BrcIndexType>(dstUb, srcUb, indexUb, sizeI[0], sizeI[1]);
            } else if (sizeI[1] < VF_LEN) {
                BrcNlastGatherTwo<T, BrcIndexType>(dstUb, srcUb, indexUb, sizeI[0], sizeI[1]);
            }
        }
    } else if (sizeI[1] <= VF_LEN) {
        BrcNlastLessThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1]);
    } else {
        if (sizeI[1] % oneBlockElementNum == 0) {
            if (sizeI[1] % VF_LEN == 0 && sizeI[0] > DEFAULT_BLK_NUM) {
                BrcNlastLargerThanVLAlignedWithVL<T>(dstUb, srcUb, sizeI[0], sizeI[1]);
            } else {
                BrcNlastLargerThanVLAlignedWithBlock<T>(dstUb, srcUb, sizeI[0], sizeI[1]);
            }
        } else {
            if constexpr (constRank == -1) {
                return true;
            } else {
                BrcNlastLargerThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1]);
            }
        }
    }
    return false;
}

template <typename T, int32_t constRank = -1>
__aicore__ inline bool BrcNlastWrapperForThreeDim(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, const uint32_t* dstShape, const uint32_t* srcStride)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    uint16_t sizeI[3];
    uint16_t stride[3];
    sizeI[0] = static_cast<uint16_t>(dstShape[0]);
    sizeI[1] = static_cast<uint16_t>(dstShape[1]);
    sizeI[2] = static_cast<uint16_t>(dstShape[2]);
    stride[0] = static_cast<uint16_t>(srcStride[0]);
    stride[1] = static_cast<uint16_t>(srcStride[1]);
    stride[2] = static_cast<uint16_t>(srcStride[2]);

    if (sizeI[2] < VF_LEN_HALF && sizeof(T) != sizeof(uint8_t)) {
        uint32_t newDstShape[3] = {dstShape[0], dstShape[1], dstShape[2]};
        uint32_t newSrcStride[3] = {srcStride[0], srcStride[1], srcStride[2]};
        GatherWrapper(dstUb, srcUb, newDstShape, newSrcStride);
    } else if (sizeI[2] <= VF_LEN) {
        BrcNlastLessThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0], stride[1]);
    } else {
        if (sizeI[2] % oneBlockElementNum == 0) {
            if (sizeI[2] % VF_LEN == 0) {
                BrcNlastLargerThanVLAlignedWithVL<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0], stride[1]);
            } else {
                BrcNlastLargerThanVLAlignedWithBlock<T>(
                    dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0], stride[1]);
            }
        } else {
            if constexpr (constRank == -1) {
                return true;
            } else {
                BrcNlastLargerThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], stride[0], stride[1]);
            }
        }
    }
    return false;
}

template <typename T, int32_t constRank = -1>
__aicore__ inline bool BrcNlastWrapperForFourDim(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, const uint32_t* dstShape, const uint32_t* srcStride)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    uint16_t sizeI[4];
    uint16_t stride[4];
    sizeI[0] = static_cast<uint16_t>(dstShape[0]);
    sizeI[1] = static_cast<uint16_t>(dstShape[1]);
    sizeI[2] = static_cast<uint16_t>(dstShape[2]);
    sizeI[3] = static_cast<uint16_t>(dstShape[3]);
    stride[0] = static_cast<uint16_t>(srcStride[0]);
    stride[1] = static_cast<uint16_t>(srcStride[1]);
    stride[2] = static_cast<uint16_t>(srcStride[2]);
    stride[3] = static_cast<uint16_t>(srcStride[3]);

    if (sizeI[3] < VF_LEN_HALF && sizeof(T) != sizeof(uint8_t)) {
        uint32_t newDstShape[4] = {dstShape[0], dstShape[1], dstShape[2], dstShape[3]};
        uint32_t newSrcStride[4] = {srcStride[0], srcStride[1], srcStride[2], srcStride[3]};
        GatherWrapperForFourDim(dstUb, srcUb, newDstShape, newSrcStride);
    } else if (sizeI[3] <= VF_LEN) {
        if (sizeI[3] % oneBlockElementNum == 0) {
            BrcNlastLessThanVLAligned<T>(
                dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
        } else {
            if constexpr (constRank == -1) {
                return true;
            } else {
                BrcNlastLessThanVLUnaligned<T>(
                    dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
            }
        }
    } else {
        if (sizeI[3] % oneBlockElementNum == 0) {
            if (sizeI[3] % VF_LEN == 0) {
                BrcNlastLargerThanVLAlignedWithVL<T>(
                    dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
            } else {
                BrcNlastLargerThanVLAlignedWithBlock<T>(
                    dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
            }
        } else {
            if constexpr (constRank == -1) {
                return true;
            } else {
                BrcNlastLargerThanVLUnaligned<T>(
                    dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
            }
        }
    }
    return false;
}

template <typename T>
__aicore__ inline void BrcNlastWrapperForMoreDim(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    const uint32_t* dstShape, const uint32_t* dstStride, const uint32_t* srcStride)
{
    uint16_t sizeI[4];
    uint16_t stride[4];
    sizeI[0] = static_cast<uint16_t>(dstShape[1]);
    sizeI[1] = static_cast<uint16_t>(dstShape[2]);
    sizeI[2] = static_cast<uint16_t>(dstShape[3]);
    sizeI[3] = static_cast<uint16_t>(dstShape[4]);
    stride[0] = static_cast<uint16_t>(srcStride[1]);
    stride[1] = static_cast<uint16_t>(srcStride[2]);
    stride[2] = static_cast<uint16_t>(srcStride[3]);
    stride[3] = static_cast<uint16_t>(srcStride[4]);
    uint32_t totalDim = 9;

    __local_mem__ T* srcUbTmp = srcUb;
    __local_mem__ T* dstUbTmp = dstUb;
    for (uint16_t p = 0; p < static_cast<uint16_t>(dstShape[0]); ++p) {
        dstUb = dstUbTmp + p * dstStride[0];
        srcUb = srcUbTmp + p * srcStride[0];
        uint32_t newDstShape[4] = {dstShape[1], dstShape[2], dstShape[3], dstShape[4]};
        uint32_t newSrcStride[4] = {srcStride[1], srcStride[2], srcStride[3], srcStride[4]};
        GatherWrapperForFourDim(dstUb, srcUb, newDstShape, newSrcStride);
    }
}

template <typename T>
__aicore__ inline void BrcNlastWrapperForMoreDimDynamicShape(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    const uint32_t dim, const uint32_t* dstShape, const uint32_t* dstStride, const uint32_t* srcStride)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint16_t VF_LEN_HALF = GetVecLen() / 2 / sizeof(T);
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    uint16_t sizeI[4] = {1, 1, 1, 1};
    if (dim > 4) {
        sizeI[0] = dstShape[dim - 4];
        sizeI[1] = dstShape[dim - 3];
        sizeI[2] = dstShape[dim - 2];
        sizeI[3] = dstShape[dim - 1];
    } else {
        for (uint16_t i = 0; i < dim; ++i) { sizeI[4 - dim + i] = dstShape[i]; }
    }
    uint32_t totalDim = 9;
    uint16_t loops[5] = {1, 1, 1, 1, 1};
    for (int16_t i = dim - 5, j = 4; i >= 0; --i, --j) { loops[j] = static_cast<uint16_t>(dstShape[i]); }
    uint16_t stride[4] = {0, 0, 0, 0};
    if (dim > 4) {
        stride[0] = srcStride[dim - 4];
        stride[1] = srcStride[dim - 3];
        stride[2] = srcStride[dim - 2];
        stride[3] = srcStride[dim - 1];
    } else {
        for (uint16_t i = 0; i < dim; ++i) { stride[4 - dim + i] = srcStride[i]; }
    }
    __local_mem__ T* srcUbTmp = srcUb;
    __local_mem__ T* dstUbTmp = dstUb;
    for (uint16_t i = 0; i < loops[0]; ++i) {
        for (uint16_t j = 0; j < loops[1]; ++j) {
            for (uint16_t k = 0; k < loops[2]; ++k) {
                for (uint16_t t = 0; t < loops[3]; ++t) {
                    for (uint16_t p = 0; p < loops[4]; ++p) {
                        dstUb = dstUbTmp + p * dstStride[(dim - 5 + totalDim) % totalDim]
                                + t * dstStride[(dim - 6 + totalDim) % totalDim]
                                + k * dstStride[(dim - 7 + totalDim) % totalDim]
                                + j * dstStride[(dim - 8 + totalDim) % totalDim]
                                + i * dstStride[(dim - 9 + totalDim) % totalDim];
                        srcUb = srcUbTmp + p * srcStride[(dim - 5 + totalDim) % totalDim]
                                + t * srcStride[(dim - 6 + totalDim) % totalDim]
                                + k * srcStride[(dim - 7 + totalDim) % totalDim]
                                + j * srcStride[(dim - 8 + totalDim) % totalDim]
                                + i * srcStride[(dim - 9 + totalDim) % totalDim];
                        if (sizeI[3] < VF_LEN_HALF && sizeof(T) != sizeof(uint8_t)) {
                            uint32_t newDstShape[4] = {
                                dstShape[dim - 4], dstShape[dim - 3], dstShape[dim - 2], dstShape[dim - 1]};
                            uint32_t newSrcStride[4] = {
                                srcStride[dim - 4], srcStride[dim - 3], srcStride[dim - 2], srcStride[dim - 1]};
                            GatherWrapperForFourDim(dstUb, srcUb, newDstShape, newSrcStride);
                        } else if (sizeI[3] <= VF_LEN) {
                            if (sizeI[3] % oneBlockElementNum == 0) {
                                BrcNlastLessThanVLAligned<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3],
                                    stride[0], stride[1], stride[2]);
                            } else {
                                BrcNlastLessThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3],
                                    stride[0], stride[1], stride[2]);
                            }
                        } else {
                            if (sizeI[3] % oneBlockElementNum == 0) {
                                BrcNlastLargerThanVLAlignedWithBlock<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2],
                                    sizeI[3], stride[0], stride[1], stride[2]);
                            } else {
                                BrcNlastLargerThanVLUnaligned<T>(dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3],
                                    stride[0], stride[1], stride[2]);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void BrcLastWrapperForMoreDimDynamicShape(__local_mem__ T* dstUb, __local_mem__ T* srcUb,
    const uint32_t dim, const uint32_t* dstShape, const uint32_t* dstStride, const uint32_t* srcStride)
{
    constexpr uint16_t VF_LEN = GetVecLen() / sizeof(T);
    constexpr uint32_t oneBlockElementNum = GetDataBlockSizeInBytes() / sizeof(T);
    uint16_t sizeI[4] = {1, 1, 1, 1};
    if (dim > 4) {
        sizeI[0] = dstShape[dim - 4];
        sizeI[1] = dstShape[dim - 3];
        sizeI[2] = dstShape[dim - 2];
        sizeI[3] = dstShape[dim - 1];
    } else {
        for (uint16_t i = 0; i < dim; ++i) { sizeI[4 - dim + i] = dstShape[i]; }
    }
    uint32_t totalDim = 9;
    uint16_t loops[5] = {1, 1, 1, 1, 1};
    for (int16_t i = dim - 5, j = 4; i >= 0; --i, --j) { loops[j] = static_cast<uint16_t>(dstShape[i]); }
    uint16_t stride[4] = {0, 0, 0, 0};
    if (dim > 4) {
        stride[0] = srcStride[dim - 4];
        stride[1] = srcStride[dim - 3];
        stride[2] = srcStride[dim - 2];
        stride[3] = srcStride[dim - 1];
    } else {
        for (uint16_t i = 0; i < dim; ++i) { stride[4 - dim + i] = srcStride[i]; }
    }
    __local_mem__ T* srcUbTmp = srcUb;
    __local_mem__ T* dstUbTmp = dstUb;
    for (uint16_t i = 0; i < loops[0]; ++i) {
        for (uint16_t j = 0; j < loops[1]; ++j) {
            for (uint16_t k = 0; k < loops[2]; ++k) {
                for (uint16_t t = 0; t < loops[3]; ++t) {
                    for (uint16_t p = 0; p < loops[4]; ++p) {
                        dstUb = dstUbTmp + p * dstStride[(dim - 5 + totalDim) % totalDim]
                                + t * dstStride[(dim - 6 + totalDim) % totalDim]
                                + k * dstStride[(dim - 7 + totalDim) % totalDim]
                                + j * dstStride[(dim - 8 + totalDim) % totalDim]
                                + i * dstStride[(dim - 9 + totalDim) % totalDim];
                        srcUb = srcUbTmp + p * srcStride[(dim - 5 + totalDim) % totalDim]
                                + t * srcStride[(dim - 6 + totalDim) % totalDim]
                                + k * srcStride[(dim - 7 + totalDim) % totalDim]
                                + j * srcStride[(dim - 8 + totalDim) % totalDim]
                                + i * srcStride[(dim - 9 + totalDim) % totalDim];
                        if (sizeI[3] <= VF_LEN) {
                            BrcLastLessThanVLUnaligned<T>(
                                dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
                        } else {
                            BrcLastLargerThanVLUnaligned<T>(
                                dstUb, srcUb, sizeI[0], sizeI[1], sizeI[2], sizeI[3], stride[0], stride[1], stride[2]);
                        }
                    }
                }
            }
        }
    }
}
} // namespace BroadcastInternal
} // namespace AscendC
#endif