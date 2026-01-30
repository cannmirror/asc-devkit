/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/* !
 * \file ascend_quant_l300_impl.h
 * \brief
 */
#ifndef LIB_ASCEND_QUANT_ASCEND_QUANT_PER_GROUP_L300_IMPL_H
#define LIB_ASCEND_QUANT_ASCEND_QUANT_PER_GROUP_L300_IMPL_H
#include "kernel_tensor.h"
#include "kernel_tiling/kernel_tiling.h"
#include "include/adv_api/quantization/ascend_quant_utils.h"
#include "../../common/check.h"

namespace AscendC {
constexpr uint32_t ASCENDC_QUANT_PER_GROUP_B32_VF_LEN = GetVecLen() / sizeof(uint32_t);

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__simd_vf__ inline void QuantPerTokenForS8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, const AscendQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        GetPerTokenScaleAndOffset<scaleT, config>(scaleUb + i, offsetUb + i, scaleVreg, offsetVreg);
        uint32_t sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<scaleT>(sreg);
            if constexpr (SupportType<srcT, half>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempVreg, srcUb + i * para.n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
            }
            TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerTokenForS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                          const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                          const AscendQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    QuantPerTokenForS8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__simd_vf__ inline void QuantPerTokenForS8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ scaleT* scaleUb, const scaleT offset, const AscendQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        GetPerTokenScale<scaleT>(scaleUb + i, scaleVreg);
        uint32_t sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<scaleT>(sreg);;
            if constexpr (SupportType<srcT, half>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempVreg, srcUb + i * para.n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                    srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
            }
            TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerTokenForS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                          const LocalTensor<scaleT>& scaleTensor, const scaleT offset,
                                          const AscendQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    QuantPerTokenForS8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offset, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__simd_vf__ inline void QuantPerGroupForKColS8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, const AscendQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        uint32_t sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<scaleT>(sreg);;
            GetPerGroupScale(scaleUb  + i * scaleK, j * vecLen, para, config, scaleVreg);
            if constexpr (config.hasOffset) {
                GetPerGroupOffset(offsetUb + i * scaleK, j * vecLen, para, config, offsetVreg);
            }
            if constexpr (SupportType<srcT, half>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempVreg, srcUb + i * para.n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * para.n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
            }
            TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKColS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                              const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                              const AscendQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    QuantPerGroupForKColS8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__simd_vf__ inline void QuantPerGroupForKColS8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ scaleT* scaleUb, const scaleT offset, const AscendQuantParam para)
{
    uint16_t rowNum = para.calCount / para.n;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    uint16_t scaleK = CeilDivision(para.n, para.groupSize);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    for (uint16_t i = 0; i < rowNum; ++i) {
        uint32_t sreg = para.n;
        for (uint16_t j = 0; j < repeat; ++j) {
            preg = MicroAPI::UpdateMask<scaleT>(sreg);;
            GetPerGroupScale(scaleUb  + i * scaleK, j * vecLen, para, config, scaleVreg);
            if constexpr (SupportType<srcT, half>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    tempVreg, srcUb + i * para.n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(
                    srcVreg, srcUb + i * para.n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                    srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
            }
            TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * para.n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKColS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                              const LocalTensor<scaleT>& scaleTensor, const scaleT offset,
                                              const AscendQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    QuantPerGroupForKColS8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offset, para);
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__simd_callee__ inline void QuantPerGroupForKRowS8TailBlock(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
                                                       __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb,
                                                       uint16_t repeat, uint16_t tailRow, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * vecLen);
            if constexpr (config.hasOffset) {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(offsetVreg, offsetUb + j * vecLen);
            }
            preg = MicroAPI::UpdateMask<scaleT>(sreg);;
            if constexpr (SupportType<srcT, half>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, srcUb + i * n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
            }
            TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__simd_vf__ inline void QuantPerGroupForKRowS8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ scaleT* scaleUb, __ubuf__ scaleT* offsetUb, const AscendQuantParam para,
    uint16_t rowNum, uint16_t tailRow)
{
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> offsetVreg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    for (uint16_t i = 0; i < mainRowGroup; ++i) {
        for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
            uint32_t sreg = para.n;
            for (uint16_t k = 0; k < repeat; ++k) {
                LoadContinousScaleAndOffset<scaleT, config>(scaleUb + i * para.n + k * vecLen,
                    offsetUb + i * para.n + k * vecLen, scaleVreg, offsetVreg);
                preg = MicroAPI::UpdateMask<scaleT>(sreg);;
                if constexpr (SupportType<srcT, half>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg,
                        srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Add<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, offsetVreg, preg);
                }
                TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, dstVreg, preg);
            }
        }
    }
    QuantPerGroupForKRowS8TailBlock<dstT, srcT, scaleT, config>(
        dstUb + mainRowGroup * para.groupSize * para.n, srcUb + mainRowGroup * para.groupSize * para.n,
        scaleUb + mainRowGroup * para.n, offsetUb + mainRowGroup * para.n, repeat, tailRow, para.n, vecLen);
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                              const LocalTensor<scaleT>& scaleTensor, const LocalTensor<scaleT>& offsetTensor,
                                              const AscendQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    __ubuf__ scaleT* offsetUb = (__ubuf__ scaleT*)offsetTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t tailRow = rowNum % para.groupSize;
    QuantPerGroupForKRowS8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offsetUb, para, rowNum, tailRow);
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__simd_callee__ inline void QuantPerGroupForKRowS8TailBlock(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
                                                       __ubuf__ scaleT* scaleUb, const scaleT& offset,
                                                       uint16_t repeat, uint16_t tailRow, uint32_t n, uint32_t vecLen)
{
    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};
    for (uint16_t i = 0; i < tailRow; ++i) {
        uint32_t sreg = n;
        for (uint16_t j = 0; j < repeat; ++j) {
            MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + j * vecLen);
            preg = MicroAPI::UpdateMask<scaleT>(sreg);;
            if constexpr (SupportType<srcT, half>() && SupportType<scaleT, float>()) {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(tempVreg, srcUb + i * n + j * vecLen);
                MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
            } else {
                MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(srcVreg, srcUb + i * n + j * vecLen);
            }
            MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
            if constexpr (config.hasOffset) {
                MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                    srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
            }
            TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
            StoreRes<dstT, scaleT>(dstUb + i * n + j * vecLen, dstVreg, preg);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__simd_vf__ inline void QuantPerGroupForKRowS8VF(__ubuf__ dstT* dstUb, __ubuf__ srcT* srcUb,
    __ubuf__ scaleT* scaleUb, const scaleT offset, const AscendQuantParam para,
    uint16_t rowNum, uint16_t tailRow)
{
    uint16_t mainRowGroup = rowNum / para.groupSize;
    uint32_t vecLen = GetVecLen() / sizeof(scaleT);
    uint16_t repeat = CeilDivision(para.n, vecLen);
    static constexpr MicroAPI::CastTrait castTrait = {
        MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, config.roundMode};

    MicroAPI::MaskReg preg;
    MicroAPI::RegTensor<scaleT> scaleVreg;
    MicroAPI::RegTensor<srcT> tempVreg;
    MicroAPI::RegTensor<dstT> dstVreg;
    MicroAPI::RegTensor<scaleT> srcVreg;
    for (uint16_t i = 0; i < mainRowGroup; ++i) {
        for (uint16_t j = 0; j < static_cast<uint16_t>(para.groupSize); ++j) {
            uint32_t sreg = para.n;
            for (uint16_t k = 0; k < repeat; ++k) {
                MicroAPI::DataCopy<scaleT, MicroAPI::LoadDist::DIST_NORM>(scaleVreg, scaleUb + i * para.n + k * vecLen);
                preg = MicroAPI::UpdateMask<scaleT>(sreg);;
                if constexpr (SupportType<srcT, half>() && SupportType<scaleT, float>()) {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        tempVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                    MicroAPI::Cast<float, srcT, layoutZMrgZ>(srcVreg, tempVreg, preg);
                } else {
                    MicroAPI::DataCopy<srcT, MicroAPI::LoadDist::DIST_NORM>(
                        srcVreg, srcUb + (i * para.groupSize + j) * para.n + k * vecLen);
                }
                MicroAPI::Mul<scaleT, MicroAPI::MaskMergeMode::ZEROING>(srcVreg, srcVreg, scaleVreg, preg);
                if constexpr (config.hasOffset) {
                    MicroAPI::Adds<scaleT, scaleT, MicroAPI::MaskMergeMode::ZEROING>(
                        srcVreg, srcVreg, static_cast<scaleT>(offset), preg);
                }
                TransRegForS8<dstT, scaleT, castTrait>(srcVreg, dstVreg, preg);
                StoreRes<dstT, scaleT>(dstUb + (i * para.groupSize + j) * para.n + k * vecLen, dstVreg, preg);
            }
        }
    }
    QuantPerGroupForKRowS8TailBlock<dstT, srcT, scaleT, config>(
        dstUb + mainRowGroup * para.groupSize * para.n, srcUb + mainRowGroup * para.groupSize * para.n,
        scaleUb + mainRowGroup * para.n, offset, repeat, tailRow, para.n, vecLen);
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void QuantPerGroupForKRowS8(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                              const LocalTensor<scaleT>& scaleTensor, const scaleT offset,
                                              const AscendQuantParam& para)
{
    __ubuf__ dstT* dstUb = (__ubuf__ dstT*)dstTensor.GetPhyAddr();
    __ubuf__ srcT* srcUb = (__ubuf__ srcT*)srcTensor.GetPhyAddr();
    __ubuf__ scaleT* scaleUb = (__ubuf__ scaleT*)scaleTensor.GetPhyAddr();
    uint16_t rowNum = para.calCount / para.n;
    uint16_t tailRow = rowNum % para.groupSize;
    QuantPerGroupForKRowS8VF<dstT, srcT, scaleT, config>(dstUb, srcUb, scaleUb, offset, para, rowNum, tailRow);
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerToken(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                           const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                           const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerTokenForS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerToken(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                           const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                           const scaleT offset, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerTokenForS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerGroupForKCol(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                  const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                                  const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerGroupForKColS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerGroupForKCol(const LocalTensor<dstT>& dstTensor,
    const LocalTensor<srcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer,
    const LocalTensor<scaleT>& scaleTensor, const scaleT offset, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerGroupForKColS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerGroupForKRow(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                  const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                                  const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerGroupForKRowS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offsetTensor, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config>
__aicore__ inline void AscendQuantPerGroupForKRow(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
                                                  const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
                                                  const scaleT offset, const AscendQuantParam& para)
{
    if constexpr (SupportType<dstT, int8_t>()) {
        QuantPerGroupForKRowS8<dstT, srcT, scaleT, config>(dstTensor, srcTensor, scaleTensor, offset, para);
    } else {
        ASCENDC_ASSERT((false), { KERNEL_LOG(KERNEL_ERROR, "unsupport dstT for AscendQuant!"); });
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config,
    const AscendQuantPolicy& policy>
__aicore__ inline void AscendQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor,
    const LocalTensor<scaleT>& offsetTensor, const AscendQuantParam& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(scaleTensor, "scaleTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(offsetTensor, "offsetTensor", "VECIN, VECOUT, VECCALC");
    static_assert(SupportType<srcT, half, float>(),
        "AscendQuant only support half/float input dtype");
    static_assert(SupportType<scaleT, half, float>(),
        "AscendQuant only support half/float scale dtype");
    static_assert((policy == AscendQuantPolicy::PER_TOKEN || policy == AscendQuantPolicy::PER_GROUP),
        "unsupported policy for AscendQuant in current device!");
    ASCENDC_ASSERT((para.calCount <= srcTensor.GetSize() && para.calCount <= dstTensor.GetSize() && para.calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            para.calCount, srcTensor.GetSize(), dstTensor.GetSize());
    });
    ASCENDC_ASSERT(
        (para.calCount % para.n == 0), { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of n!"); });
    if constexpr (policy == AscendQuantPolicy::PER_TOKEN) {
        static_assert(
            SupportType<dstT, int8_t>(),
            "AscendQuant PerToken only support int8_t output dtype");
        AscendQuantPerToken<dstT, srcT, scaleT, config>(
            dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
    } else if constexpr (policy == AscendQuantPolicy::PER_GROUP) {
        static_assert(
            SupportType<dstT, int8_t>(),
            "AscendQuant PerGroup only support int8_t output dtype");
        static_assert(
            ((config.kDim == 1) || (config.kDim == 0)), "AscendAntiQuant PerGroup only support K is axis 0/1!");
        ASCENDC_ASSERT((para.groupSize > 0 && para.groupSize % 32 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "groupSize must be an integer multiple of 32 and greater than 0 !"); });
        if constexpr (config.kDim == 1) {
            AscendQuantPerGroupForKCol<dstT, srcT, scaleT, isReuseSource, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
        } else {
            AscendQuantPerGroupForKRow<dstT, srcT, scaleT, isReuseSource, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offsetTensor, para);
        }
    }
}

template <typename dstT, typename srcT, typename scaleT, bool isReuseSource = false, const AscendQuantConfig& config,
    const AscendQuantPolicy& policy>
__aicore__ inline void AscendQuantImpl(const LocalTensor<dstT>& dstTensor, const LocalTensor<srcT>& srcTensor,
    const LocalTensor<uint8_t>& sharedTmpBuffer, const LocalTensor<scaleT>& scaleTensor, const scaleT offset,
    const AscendQuantParam& para)
{
    if ASCEND_IS_AIC {
        return;
    }
    CheckTensorPosition(dstTensor, "dstTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(srcTensor, "srcTensor", "VECIN, VECOUT, VECCALC");
    CheckTensorPosition(scaleTensor, "scaleTensor", "VECIN, VECOUT, VECCALC");
    static_assert(SupportType<srcT, half, float>(),
        "AscendQuant only support half/float input dtype");
    static_assert(SupportType<scaleT, half, float>(),
        "AscendQuant only support half/float scale dtype");
    static_assert((policy == AscendQuantPolicy::PER_TOKEN || policy == AscendQuantPolicy::PER_GROUP),
        "unsupported policy for AscendQuant in current device!");
    ASCENDC_ASSERT((para.calCount <= srcTensor.GetSize() && para.calCount <= dstTensor.GetSize() && para.calCount >= 0), {
        KERNEL_LOG(KERNEL_ERROR, "calCount is %u, which should be in [0, min(%u, %u)]",
            para.calCount, srcTensor.GetSize(), dstTensor.GetSize());
    });
    ASCENDC_ASSERT(
        (para.calCount % para.n == 0), { KERNEL_LOG(KERNEL_ERROR, "calCount must be an integer multiple of n!"); });
    if constexpr (policy == AscendQuantPolicy::PER_TOKEN) {
        static_assert(SupportType<dstT, int8_t>(),
            "AscendQuant PerToken only support int8_t output dtype");
        AscendQuantPerToken<dstT, srcT, scaleT, config>(
            dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset, para);
    } else if constexpr (policy == AscendQuantPolicy::PER_GROUP) {
        static_assert(
            SupportType<dstT, int8_t>(),
            "AscendQuant PerGroup only support int8_t output dtype");
        static_assert(
            ((config.kDim == 1) || (config.kDim == 0)), "AscendAntiQuant PerGroup only support K is axis 0/1!");
        ASCENDC_ASSERT((para.groupSize > 0 && para.groupSize % 32 == 0),
            { KERNEL_LOG(KERNEL_ERROR, "groupSize must be an integer multiple of 32 and greater than 0 !"); });
        if constexpr (config.kDim == 1) {
            AscendQuantPerGroupForKCol<dstT, srcT, scaleT, isReuseSource, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset, para);
        } else {
            AscendQuantPerGroupForKRow<dstT, srcT, scaleT, isReuseSource, config>(
                dstTensor, srcTensor, sharedTmpBuffer, scaleTensor, offset, para);
        }
    }
}
}  //  namespace AscendC
#endif  // LIB_ASCEND_QUANT_ASCEND_QUANT_PER_GROUP_L300_IMPL_H
