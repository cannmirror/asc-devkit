/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file pad_c310_impl.h
 * \brief
 */
#ifndef AICORE_ADV_API_DETAIL_PAD_PAD_PAD_C310_IMPL_H
#define AICORE_ADV_API_DETAIL_PAD_PAD_PAD_C310_IMPL_H

namespace AscendC {
namespace PadInternal {

template <typename T>
__aicore__ inline void SetLeftPadMask(MicroAPI::MaskReg& mask, uint16_t leftPad)
{
    uint32_t scalar = leftPad;
    mask = MicroAPI::UpdateMask<T>(scalar);
}

template <typename T>
__aicore__ inline void SetRightPadMask(MicroAPI::MaskReg& mask, uint32_t srcOriWidth, uint16_t rightPad)
{
    MicroAPI::MaskReg maskAll = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

    uint32_t srcOriWidthForMask = srcOriWidth;
    MicroAPI::MaskReg maskSrcOri = MicroAPI::UpdateMask<T>(srcOriWidthForMask);
    MicroAPI::MaskReg maskNotSrcOri = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALLF>();
    MicroAPI::MaskNot(maskNotSrcOri, maskSrcOri, maskAll);

    uint32_t scalar = srcOriWidth + rightPad;
    MicroAPI::MaskReg maskWithPad = MicroAPI::UpdateMask<T>(scalar);
    MicroAPI::MaskAnd(mask, maskWithPad, maskNotSrcOri, maskAll);
}

template <typename T>
__aicore__ inline void UnAlignedPad(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, PadParams& padParams, PadTiling& padTiling)
{
    MicroAPI::RegTensor<T> regT;

    uint32_t regBlockElementCnt = CUBE_MAX_SIZE / sizeof(T);
    uint32_t lastRegBlockPerRowElementCnt = padTiling.srcWidth % regBlockElementCnt;
    uint32_t regBlockCntPerRow = padTiling.srcWidth / regBlockElementCnt;

    MicroAPI::MaskReg leftPadask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALLF>();
    SetLeftPadMask<T>(leftPadask, padParams.leftPad);
    MicroAPI::MaskReg rightPadask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALLF>();
    SetRightPadMask<T>(rightPadask, padTiling.srcOriWidth - regBlockCntPerRow * regBlockElementCnt, padParams.rightPad);

    uint32_t lastRegBlockPerRowCopyOutCnt = lastRegBlockPerRowElementCnt + padParams.rightPad;

    __local_mem__ T* srcStartPerBlock = srcUb;
    __local_mem__ T* dstStartPerBlock = dstUb;
    for (uint32_t i = 0; i < padTiling.srcHeight; i++) {
        MicroAPI::UnalignReg unalignRegCpyIn;
        MicroAPI::UnalignReg unalignRegCpyOut;

        MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::MERGING>(regT, static_cast<T>(padParams.padValue), leftPadask);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dstStartPerBlock, regT, unalignRegCpyOut, padParams.leftPad);

        for (uint32_t j = 0; j < regBlockCntPerRow; ++j) {
            MicroAPI::DataCopyUnAlignPre(unalignRegCpyIn, srcStartPerBlock);
            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                regT, unalignRegCpyIn, srcStartPerBlock, regBlockElementCnt);

            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstStartPerBlock, regT, unalignRegCpyOut, regBlockElementCnt);
        }

        MicroAPI::DataCopyUnAlignPre(unalignRegCpyIn, srcStartPerBlock);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            regT, unalignRegCpyIn, srcStartPerBlock, lastRegBlockPerRowElementCnt);

        MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::MERGING>(regT, static_cast<T>(padParams.padValue), rightPadask);

        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dstStartPerBlock, regT, unalignRegCpyOut, lastRegBlockPerRowCopyOutCnt);
        MicroAPI::DataCopyUnAlignPost(dstStartPerBlock, unalignRegCpyOut, 0);
    }
}

template <typename T>
__aicore__ inline void AlignedPad(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, PadParams& padParams, PadTiling& tiling)
{
    MicroAPI::RegTensor<T> regT;

    uint32_t regBlockElementCnt = CUBE_MAX_SIZE / sizeof(T);
    uint32_t regBlockCntPerRow = tiling.srcWidth / regBlockElementCnt;
    uint32_t lastRegBlockPerRowElementCnt = tiling.srcWidth % regBlockElementCnt;
    if (lastRegBlockPerRowElementCnt == 0) {
        regBlockCntPerRow = regBlockCntPerRow - 1;
        lastRegBlockPerRowElementCnt = regBlockElementCnt;
    }

    uint32_t regBlockElementCntForMask = regBlockElementCnt;
    uint32_t lastRegBlockPerRowElementCntForMask = lastRegBlockPerRowElementCnt;
    MicroAPI::MaskReg regBlockPerRowCopyOutMask = MicroAPI::UpdateMask<T>(regBlockElementCntForMask);
    MicroAPI::MaskReg lastRegBlockPerRowCopyOutMask = MicroAPI::UpdateMask<T>(lastRegBlockPerRowElementCntForMask);

    MicroAPI::MaskReg rightPadask = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALLF>();
    SetRightPadMask<T>(rightPadask, tiling.srcOriWidth - regBlockCntPerRow * regBlockElementCnt, padParams.rightPad);

    __local_mem__ T* srcStartPerBlock = srcUb;
    __local_mem__ T* dstStartPerBlock = dstUb;

    for (uint32_t i = 0; i < tiling.srcHeight; i++) {
        for (uint32_t j = 0; j < regBlockCntPerRow; j++) {
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regT, srcStartPerBlock, regBlockElementCnt);
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstStartPerBlock, regT, regBlockElementCnt, regBlockPerRowCopyOutMask);
        }

        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            regT, srcStartPerBlock, lastRegBlockPerRowElementCnt);
        MicroAPI::Duplicate<T, MicroAPI::MaskMergeMode::MERGING>(regT, static_cast<T>(padParams.padValue), rightPadask);
        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dstStartPerBlock, regT, lastRegBlockPerRowElementCnt, lastRegBlockPerRowCopyOutMask);
    }
}

template <typename T>
__aicore__ inline void UnPad(
    __local_mem__ T* dstUb, __local_mem__ T* srcUb, UnPadParams& padParams, UnPadTiling& tiling)
{
    MicroAPI::RegTensor<T> regT;
    MicroAPI::UnalignReg unalignRegCpyOut;

    uint32_t regBlockElementCnt = CUBE_MAX_SIZE / sizeof(T);
    uint32_t regBlockCntPerRow = tiling.srcWidth / regBlockElementCnt;
    uint32_t lastRegBlockPerRowElementCnt = tiling.srcWidth % regBlockElementCnt;
    if (lastRegBlockPerRowElementCnt == 0) {
        regBlockCntPerRow = regBlockCntPerRow - 1;
        lastRegBlockPerRowElementCnt = regBlockElementCnt;
    }

    __local_mem__ T* srcStartPerBlock = srcUb;
    __local_mem__ T* dstStartPerBlock = dstUb;
    for (uint32_t i = 0; i < tiling.srcHeight; i++) {
        for (uint32_t j = 0; j < regBlockCntPerRow; ++j) {
            MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(regT, srcStartPerBlock, regBlockElementCnt);

            MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                dstStartPerBlock, regT, unalignRegCpyOut, regBlockElementCnt);
        }

        MicroAPI::DataCopy<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            regT, srcStartPerBlock, lastRegBlockPerRowElementCnt);
        MicroAPI::DataCopyUnAlign<T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            dstStartPerBlock, regT, unalignRegCpyOut, lastRegBlockPerRowElementCnt - padParams.rightPad);
        MicroAPI::DataCopyUnAlignPost(dstStartPerBlock, unalignRegCpyOut, 0);
    }
}

} // namespace PadInternal

template <typename T>
__aicore__ inline void PadCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    PadParams& padParams, const LocalTensor<uint8_t>& sharedTmpBuffer, PadTiling& tiling)
{
    // 32B aligned
    if (tiling.srcWidth * sizeof(T) % ONE_BLK_SIZE == 0) {
        VF_CALL<PadInternal::AlignedPad<T>>(
            (__local_mem__ T*)dstTensor.GetPhyAddr(), (__local_mem__ T*)srcTensor.GetPhyAddr(), padParams, tiling);
    } else {
        VF_CALL<PadInternal::UnAlignedPad<T>>(
            (__local_mem__ T*)dstTensor.GetPhyAddr(), (__local_mem__ T*)srcTensor.GetPhyAddr(), padParams, tiling);
    }
}

/* **************************************************************************************************
 * UnPad                                             *
 * ************************************************************************************************* */
/*
 * @ingroup UnPad
 * @brief unpad from src to dst, applicable to vector data
 * @param [out] dstTensor output LocalTensor
 * @param [in] srcTensor input LocalTensor
 * @param [in] sharedTmpBuffer tmp buffer LocalTensor
 * @param [in] unPadParams.leftPad number of left unpad
 * @param [in] unPadParams.rightPad number of right unpad
 */
template <typename T>
__aicore__ inline void UnPadCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor,
    UnPadParams& unPadParams, LocalTensor<uint8_t>& sharedTmpBuffer, UnPadTiling& tiling)
{
    VF_CALL<PadInternal::UnPad<T>>(
        (__local_mem__ T*)dstTensor.GetPhyAddr(), (__local_mem__ T*)srcTensor.GetPhyAddr(), unPadParams, tiling);
}
} // namespace AscendC

#endif // AICORE_ADV_API_DETAIL_PAD_PAD_PAD_C310_IMPL_H
