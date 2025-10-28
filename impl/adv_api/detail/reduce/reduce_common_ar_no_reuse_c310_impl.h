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
#ifndef DETAIL_REDUCE_REDUCE_COMMON_AR_NO_REUSE_C310_IMPL_H
#define DETAIL_REDUCE_REDUCE_COMMON_AR_NO_REUSE_C310_IMPL_H

#include "kernel_operator_intf.h"
#include "kernel_tensor.h"
#include "reduce_common_util_impl.h"
#include "reduce_common_util_c310_impl.h"

namespace AscendC {
template <class T>
__aicore__ inline void ReduceARAligned(__ubuf__ T *dst, __ubuf__ T *src, uint32_t aLength, uint32_t rLength)
{
    constexpr uint16_t sregLower = static_cast<uint16_t>(GetVecLen() / sizeof(T));
    if (rLength <= sregLower) {
        __VEC_SCOPE__
        {
            uint32_t count = rLength;
            MicroAPI::RegTensor<T> src0Reg;
            MicroAPI::RegTensor<T> dst0Reg;
            MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<T>(count);
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                DataCopy(src0Reg, src + j * rLength);
                ReduceSum(dst0Reg, src0Reg, preg0);
                DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((dst + j), dst0Reg, pregOne);
            }
        }
    } else if (rLength <= 2 * sregLower) {
        uint32_t count = rLength - sregLower;
        uint16_t aOffset = 0;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> src0Reg;
            MicroAPI::RegTensor<T> src1Reg;
            MicroAPI::RegTensor<T> dst0Reg;
            MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<T>(count);
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
            MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                aOffset = j * rLength;
                DataCopy(src0Reg, src + aOffset);
                DataCopy(src1Reg, src + sregLower + aOffset);
                Add(dst0Reg, src0Reg, src1Reg, preg0);
                Select(dst0Reg, dst0Reg, src0Reg, preg0);
                ReduceSum(dst0Reg, dst0Reg, pregFull);
                DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((dst + j), dst0Reg, pregOne);
            }
        }
    } else {
        uint32_t rHeadLength = ReduceOpInternal::CalculateRMainBlock(rLength); // calculate 2^k
        uint16_t tailLength = static_cast<uint16_t>(rLength - rHeadLength);
        uint16_t repeatTimes1 = static_cast<uint16_t>(CeilDivision(tailLength, sregLower) / 2);
        uint16_t repeatTimes2 = static_cast<uint16_t>(CeilDivision(tailLength, sregLower) % 2);
        uint16_t remainingoffset = (repeatTimes1 + repeatTimes2) * sregLower * 2;
        uint16_t repeatTimes3 = (rHeadLength - remainingoffset) / sregLower / 2;
        uint16_t halfAddCount = CeilDivision(CeilDivision(rHeadLength, sregLower), 2);
        uint16_t halfAddRepeatTimes = ReduceOpInternal::CalculateFolds(CeilDivision(halfAddCount, sregLower));
        uint16_t halfAddBlock = static_cast<uint16_t>(CeilDivision(halfAddCount, sregLower));
        uint32_t count = 0;
        uint16_t currentHalfAddTimes;
        uint16_t aOffset = 0;
        __ubuf__ T *tmpBuf = AscendCUtils::GetTemporaryBufferAddr<T>(TMP_UB_OFFSET, rHeadLength / sregLower / 2);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> src0Reg;
            MicroAPI::RegTensor<T> src1Reg;
            MicroAPI::RegTensor<T> src2Reg;
            MicroAPI::RegTensor<T> src3Reg;
            MicroAPI::RegTensor<T> dst0Reg;
            MicroAPI::RegTensor<T> dst1Reg;
            MicroAPI::RegTensor<T> meanReg;

            MicroAPI::MaskReg preg0;
            MicroAPI::MaskReg preg1;
            MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();

            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                count = tailLength;
                aOffset = j * rLength;
                // main tail block add to main block
                for (uint16_t i = 0; i < repeatTimes1; i++) {
                    DataCopy(src0Reg, src + aOffset + 2 * i * sregLower);
                    DataCopy(src1Reg, src + aOffset + (2 * i + 1) * sregLower);
                    DataCopy(src2Reg, src + aOffset + rHeadLength + 2 * i * sregLower);
                    DataCopy(src3Reg, src + aOffset + rHeadLength + (2 * i + 1) * sregLower);
                    preg0 = MicroAPI::UpdateMask<T>(count);
                    MicroAPI::Add(dst0Reg, src0Reg, src2Reg, preg0); // first main block adds first tail block
                    preg1 = MicroAPI::UpdateMask<T>(count);
                    MicroAPI::Add(dst1Reg, src1Reg, src3Reg, preg1); // second main block adds second tail block
                    Select(dst1Reg, dst1Reg, src1Reg, preg1);
                    Add(dst0Reg, dst0Reg, dst1Reg, pregFull);
                    ReduceSum(dst0Reg, dst0Reg, pregFull);
                    DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((tmpBuf + i), dst0Reg, pregOne);
                }

                // remaining tail block add to main block
                for (uint16_t i = 0; i < repeatTimes2; i++) {
                    DataCopy(src0Reg, src + aOffset + 2 * repeatTimes1 * sregLower);
                    DataCopy(src1Reg, src + aOffset + (2 * repeatTimes1 + 1) * sregLower);
                    DataCopy(src2Reg, src + aOffset + rHeadLength + 2 * repeatTimes1 * sregLower);
                    preg0 = MicroAPI::UpdateMask<T>(count);
                    Add(dst0Reg, src0Reg, src2Reg, preg0); // first main block adds first tail block
                    Select(dst0Reg, dst0Reg, src0Reg, preg0);
                    MicroAPI::Add(dst0Reg, dst0Reg, src1Reg, pregFull); // above add result adds second main block
                    ReduceSum(dst0Reg, dst0Reg, pregFull);
                    DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((tmpBuf + repeatTimes1 + i), dst0Reg,
                        pregOne);
                }

                // remaining main block add to another main block
                for (uint16_t i = 0; i < repeatTimes3; i++) {
                    DataCopy(src0Reg, src + aOffset + remainingoffset + 2 * i * sregLower);
                    DataCopy(src1Reg, src + aOffset + remainingoffset + (2 * i + 1) * sregLower);
                    Add(dst0Reg, src0Reg, src1Reg, pregFull); // first main block adds second main block
                    ReduceSum(dst0Reg, dst0Reg, pregFull);
                    DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((tmpBuf + repeatTimes1 + repeatTimes2 + i),
                        dst0Reg, pregOne);
                }

                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

                // Processes the 2^(k-1) data in tmp buffer.
                currentHalfAddTimes = halfAddBlock;
                for (uint16_t k = 0; k < static_cast<uint16_t>(halfAddRepeatTimes); k++) {
                    currentHalfAddTimes = currentHalfAddTimes / ReduceOpInternal::NO_REUSE_FOLD_NUM;
                    for (uint16_t i = 0; i < currentHalfAddTimes; i++) {
                        DataCopy(src0Reg, tmpBuf + i * sregLower);
                        DataCopy(src1Reg, tmpBuf + (currentHalfAddTimes + i) * sregLower);
                        Add(dst0Reg, src0Reg, src1Reg, pregFull);
                        DataCopy(tmpBuf + i * sregLower, dst0Reg, pregFull);
                    }
                    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                }

                count = halfAddCount;
                preg0 = MicroAPI::UpdateMask<T>(count);
                DataCopy(src0Reg, tmpBuf);
                ReduceSum(dst0Reg, src0Reg, preg0);
                DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((dst + j), dst0Reg, pregOne);
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
            }
        }
        AscendCUtils::FreeTemporaryBuffer<T>(tmpBuf);
    }
}

template <class T>
__aicore__ inline void ReduceARUnAligned(__ubuf__ T *dst, __ubuf__ T *src, uint32_t aLength, uint32_t rLength)
{
    constexpr uint16_t sregLower = static_cast<uint16_t>(GetVecLen() / sizeof(T));
    if (rLength <= sregLower) {
        __VEC_SCOPE__
        {
            uint32_t count = rLength;
            MicroAPI::RegTensor<T> src0Reg;
            MicroAPI::RegTensor<T> dst0Reg;
            MicroAPI::UnalignReg src0Ureg;
            MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<T>(count);
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
            uint64_t hoistSrcAddr = (uint64_t)src;
            MicroAPI::DataCopyUnAlignPre(src0Ureg, ((__ubuf__ T *&)hoistSrcAddr));
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                MicroAPI::DataCopyUnAlign(src0Reg, src0Ureg, ((__ubuf__ T *&)hoistSrcAddr), rLength);
                ReduceSum(dst0Reg, src0Reg, preg0);
                DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((dst + j), dst0Reg, pregOne);
            }
        }
    } else if (rLength <= 2 * sregLower) {
        uint32_t count = rLength - sregLower;
        uint16_t aOffset = 0;
        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> src0Reg;
            MicroAPI::RegTensor<T> src1Reg;
            MicroAPI::RegTensor<T> dst0Reg;
            MicroAPI::UnalignReg src0Ureg;
            MicroAPI::UnalignReg src1Ureg;
            MicroAPI::MaskReg preg0 = MicroAPI::UpdateMask<T>(count);
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();
            MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
            uint64_t hoistSrc0Addr = (uint64_t)src;
            uint64_t hoistSrc1Addr = (uint64_t)src + static_cast<uint64_t>(sregLower * sizeof(T));
            MicroAPI::DataCopyUnAlignPre(src0Ureg, ((__ubuf__ T *&)hoistSrc0Addr));
            MicroAPI::DataCopyUnAlignPre(src1Ureg, ((__ubuf__ T *&)hoistSrc1Addr));
            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                MicroAPI::DataCopyUnAlignPre(src0Ureg, ((__ubuf__ T *&)hoistSrc0Addr));
                MicroAPI::DataCopyUnAlignPre(src1Ureg, ((__ubuf__ T *&)hoistSrc1Addr));
                MicroAPI::DataCopyUnAlign(src0Reg, src0Ureg, ((__ubuf__ T *&)hoistSrc0Addr), rLength);
                MicroAPI::DataCopyUnAlign(src1Reg, src1Ureg, ((__ubuf__ T *&)hoistSrc1Addr), rLength);
                Add(dst0Reg, src0Reg, src1Reg, preg0);
                Select(dst0Reg, dst0Reg, src0Reg, preg0);
                ReduceSum(dst0Reg, dst0Reg, pregFull);
                DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((dst + j), dst0Reg, pregOne);
            }
        }
    } else {
        uint32_t rHeadLength = ReduceOpInternal::CalculateRMainBlock(rLength); // calculate 2^k
        uint16_t tailLength = static_cast<uint16_t>(rLength - rHeadLength);
        uint16_t repeatTimes1 = static_cast<uint16_t>(CeilDivision(tailLength, sregLower) / 2);
        uint16_t repeatTimes2 = static_cast<uint16_t>(CeilDivision(tailLength, sregLower) % 2);
        uint16_t remainingoffset = (repeatTimes1 + repeatTimes2) * sregLower * 2;
        uint16_t repeatTimes3 = (rHeadLength - remainingoffset) / sregLower / 2;
        uint16_t halfAddCount = CeilDivision(CeilDivision(rHeadLength, sregLower), 2);
        uint16_t halfAddRepeatTimes = ReduceOpInternal::CalculateFolds(CeilDivision(halfAddCount, sregLower));
        uint16_t halfAddBlock = static_cast<uint16_t>(CeilDivision(halfAddCount, sregLower));
        uint32_t count = 0;
        uint16_t currentHalfAddTimes;
        uint16_t aOffset = 0;
        __ubuf__ T *tmpBuf = AscendCUtils::GetTemporaryBufferAddr<T>(TMP_UB_OFFSET, rHeadLength / sregLower / 2);

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<T> src0Reg;
            MicroAPI::RegTensor<T> src1Reg;
            MicroAPI::RegTensor<T> src2Reg;
            MicroAPI::RegTensor<T> src3Reg;
            MicroAPI::RegTensor<T> dst0Reg;
            MicroAPI::RegTensor<T> dst1Reg;
            MicroAPI::RegTensor<T> meanReg;

            MicroAPI::UnalignReg src0Ureg;
            MicroAPI::UnalignReg src1Ureg;
            MicroAPI::UnalignReg src2Ureg;
            MicroAPI::UnalignReg src3Ureg;

            MicroAPI::MaskReg preg0;
            MicroAPI::MaskReg preg1;
            MicroAPI::MaskReg pregFull = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
            MicroAPI::MaskReg pregOne = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::VL1>();

            for (uint16_t j = 0; j < static_cast<uint16_t>(aLength); j++) {
                count = tailLength;
                aOffset = j * rLength;
                uint64_t hoistSrc0Addr = (uint64_t)src + static_cast<uint64_t>(aOffset * sizeof(T));
                uint64_t hoistSrc1Addr = (uint64_t)hoistSrc0Addr + static_cast<uint64_t>(sregLower * sizeof(T));
                uint64_t hoistSrc2Addr = (uint64_t)hoistSrc0Addr + static_cast<uint64_t>(rHeadLength * sizeof(T));
                uint64_t hoistSrc3Addr = (uint64_t)hoistSrc2Addr + static_cast<uint64_t>(sregLower * sizeof(T));

                // main tail block add to main block
                for (uint16_t i = 0; i < repeatTimes1; i++) {
                    MicroAPI::DataCopyUnAlignPre(src0Ureg, ((__ubuf__ T *&)hoistSrc0Addr));
                    MicroAPI::DataCopyUnAlignPre(src1Ureg, ((__ubuf__ T *&)hoistSrc1Addr));
                    MicroAPI::DataCopyUnAlignPre(src2Ureg, ((__ubuf__ T *&)hoistSrc2Addr));
                    MicroAPI::DataCopyUnAlignPre(src3Ureg, ((__ubuf__ T *&)hoistSrc3Addr));
                    MicroAPI::DataCopyUnAlign(src0Reg, src0Ureg, ((__ubuf__ T *&)hoistSrc0Addr), 2 * sregLower);
                    MicroAPI::DataCopyUnAlign(src1Reg, src1Ureg, ((__ubuf__ T *&)hoistSrc1Addr), 2 * sregLower);
                    MicroAPI::DataCopyUnAlign(src2Reg, src2Ureg, ((__ubuf__ T *&)hoistSrc2Addr), 2 * sregLower);
                    MicroAPI::DataCopyUnAlign(src3Reg, src3Ureg, ((__ubuf__ T *&)hoistSrc3Addr), 2 * sregLower);
                    preg0 = MicroAPI::UpdateMask<T>(count);
                    MicroAPI::Add(dst0Reg, src0Reg, src2Reg, preg0); // first main block adds first tail block
                    preg1 = MicroAPI::UpdateMask<T>(count);
                    MicroAPI::Add(dst1Reg, src1Reg, src3Reg, preg1); // second main block adds second tail block
                    Select(dst1Reg, dst1Reg, src1Reg, preg1);
                    Add(dst0Reg, dst0Reg, dst1Reg, pregFull);
                    ReduceSum(dst0Reg, dst0Reg, pregFull);
                    DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((tmpBuf + i), dst0Reg, pregOne);
                }

                // remaining tail block add to main block
                for (uint16_t i = 0; i < repeatTimes2; i++) {
                    MicroAPI::DataCopyUnAlignPre(src0Ureg, ((__ubuf__ T *&)hoistSrc0Addr));
                    MicroAPI::DataCopyUnAlignPre(src1Ureg, ((__ubuf__ T *&)hoistSrc1Addr));
                    MicroAPI::DataCopyUnAlignPre(src2Ureg, ((__ubuf__ T *&)hoistSrc2Addr));
                    MicroAPI::DataCopyUnAlign(src0Reg, src0Ureg, ((__ubuf__ T *&)hoistSrc0Addr), 2 * sregLower);
                    MicroAPI::DataCopyUnAlign(src1Reg, src1Ureg, ((__ubuf__ T *&)hoistSrc1Addr), 2 * sregLower);
                    MicroAPI::DataCopyUnAlign(src2Reg, src2Ureg, ((__ubuf__ T *&)hoistSrc2Addr), 2 * sregLower);
                    preg0 = MicroAPI::UpdateMask<T>(count);
                    Add(dst0Reg, src0Reg, src2Reg, preg0); // first main block adds first tail block
                    Select(dst0Reg, dst0Reg, src0Reg, preg0);
                    MicroAPI::Add(dst0Reg, dst0Reg, src1Reg, pregFull); // above add result adds second main block
                    ReduceSum(dst0Reg, dst0Reg, pregFull);
                    DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((tmpBuf + repeatTimes1 + i), dst0Reg,
                        pregOne);
                }

                // remaining main block add to another main block
                for (uint16_t i = 0; i < repeatTimes3; i++) {
                    MicroAPI::DataCopyUnAlignPre(src0Ureg, ((__ubuf__ T *&)hoistSrc0Addr));
                    MicroAPI::DataCopyUnAlignPre(src1Ureg, ((__ubuf__ T *&)hoistSrc1Addr));
                    MicroAPI::DataCopyUnAlign(src0Reg, src0Ureg, ((__ubuf__ T *&)hoistSrc0Addr), 2 * sregLower);
                    MicroAPI::DataCopyUnAlign(src1Reg, src1Ureg, ((__ubuf__ T *&)hoistSrc1Addr), 2 * sregLower);
                    Add(dst0Reg, src0Reg, src1Reg, pregFull); // first main block adds second main block
                    ReduceSum(dst0Reg, dst0Reg, pregFull);
                    DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((tmpBuf + repeatTimes1 + repeatTimes2 + i),
                        dst0Reg, pregOne);
                }

                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();

                // Processes the 2^(k-1) data in tmp buffer.
                currentHalfAddTimes = halfAddBlock;
                for (uint16_t k = 0; k < static_cast<uint16_t>(halfAddRepeatTimes); k++) {
                    currentHalfAddTimes = currentHalfAddTimes / ReduceOpInternal::NO_REUSE_FOLD_NUM;
                    for (uint16_t i = 0; i < currentHalfAddTimes; i++) {
                        DataCopy(src0Reg, tmpBuf + i * sregLower);
                        DataCopy(src1Reg, tmpBuf + (currentHalfAddTimes + i) * sregLower);
                        Add(dst0Reg, src0Reg, src1Reg, pregFull);
                        DataCopy(tmpBuf + i * sregLower, dst0Reg, pregFull);
                    }
                    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
                }

                count = halfAddCount;
                preg0 = MicroAPI::UpdateMask<T>(count);
                DataCopy(src0Reg, tmpBuf);
                ReduceSum(dst0Reg, src0Reg, preg0);
                DataCopy<T, MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>((dst + j), dst0Reg, pregOne);
                MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
            }
        }
        AscendCUtils::FreeTemporaryBuffer<T>(tmpBuf);
    }
}

template <class T> __aicore__ inline void ReduceAR(__ubuf__ T *dst, __ubuf__ T *src, uint32_t dimA, uint32_t dimR)
{
    if ((dimR * sizeof(T)) % 32 == 0) {
        ReduceARAligned(dst, src, dimA, dimR);
    } else {
        ReduceARUnAligned(dst, src, dimA, dimR);
    }
}
} // namespace AscendC
#endif // DETAIL_REDUCE_REDUCE_COMMON_AR_NO_REUSE_C310_IMPL_H