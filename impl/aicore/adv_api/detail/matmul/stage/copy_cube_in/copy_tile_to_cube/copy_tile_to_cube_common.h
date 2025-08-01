/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*!
 * \file copy_tile_to_cube_common.h
 * \brief
 */

#ifndef AICORE_ADV_API_DETAIL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_TILE_TO_CUBE_COMMON_H
#define AICORE_ADV_API_DETAIL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_TILE_TO_CUBE_COMMON_H

#include "copy_tile_to_cube_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {

template <typename IMPL, const auto& MM_CFG, class INPUT_TYPE>
class CopyTileToCubeWrapper<IMPL, MM_CFG, INPUT_TYPE,
    enable_if_t<!MatmulFeatureTrait<MM_CFG>::IsNeedUB()
                && !(INPUT_TYPE::TAG == InputTypeTag::scaleA || INPUT_TYPE::TAG == InputTypeTag::scaleB)>> {
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename INPUT_TYPE::T;

    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(DataCopyWrapper, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    MATMUL_USE_MODULE(MatmulUserDefineInfo);

public:
    __aicore__ inline CopyTileToCubeWrapper() = default;
    __aicore__ inline ~CopyTileToCubeWrapper() = default;

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyTileToCube(
        const LocalTensor<TransT>& dst, int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth)
    {
#ifdef ASCENDC_CPU_DEBUG
        if ((INPUT_TYPE::TAG == InputTypeTag::A && IMPL::CallBack::CopyA1Ptr)
            || (INPUT_TYPE::TAG == InputTypeTag::B && IMPL::CallBack::CopyB1Ptr)) {
#else
        if constexpr ((INPUT_TYPE::TAG == InputTypeTag::A && IMPL::CallBack::CopyA1Ptr)
                      || (INPUT_TYPE::TAG == InputTypeTag::B && IMPL::CallBack::CopyB1Ptr)) {
#endif
            CopyTileToCubeByCallBack(dst, curRow, curCol, tileHeight, tileWidth);
        } else {
            constexpr int32_t widthFactor =
                IsSupportB4<TransT>() && INPUT_TYPE::format == CubeFormat::ND ? INT4_TWO : 1;
            if (IsTranspose<IS_INTRA_BLOCK>()) {
                if constexpr (IsCopyFromUB<INPUT_TYPE, MM_CFG>()) {
                    LocalTensor<SrcT> src;
                    src.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
                    CopyTileToCubeFromUB<true, IS_INTRA_BLOCK>(
                        dst, src, curCol, curRow, tileWidth, tileHeight / widthFactor, widthFactor);
                } else {
                    GlobalTensor<SrcT> src;
                    src.SetGlobalBuffer(
                        MATMUL_MODULE(MatmulTensorInfo)->template GetGlobalTensor<IS_INTRA_BLOCK>().address_);
                    if constexpr (HasSparseIndex<INPUT_TYPE>() && INPUT_TYPE::TAG == InputTypeTag::B) {
                        CopyTileToCubeFromGM(dst, src, curCol, curRow, tileWidth, tileHeight,
                            MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<true>(),
                            MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<true>() >> 1,
                            MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<true, IS_INTRA_BLOCK>(),
                            MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<true, IS_INTRA_BLOCK>() >> 1,
                            MATMUL_MODULE(CopyCubeInParams)->template GetStepCol<false>(),
                            MATMUL_MODULE(CopyCubeInParams)->template IsKRowDirec<IS_INTRA_BLOCK>());
                    } else {
                        CopyTileToCubeFromGM(dst, src, curCol, curRow, tileWidth, tileHeight / widthFactor,
                            MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<true>(),
                            MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<true>(),
                            MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<true, IS_INTRA_BLOCK>(),
                            MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<true, IS_INTRA_BLOCK>() / widthFactor,
                            MATMUL_MODULE(CopyCubeInParams)->template GetStepCol<false>(),
                            MATMUL_MODULE(CopyCubeInParams)->template IsKRowDirec<IS_INTRA_BLOCK>());
                    }
                }
            } else {
#if __CCE_AICORE__ == 220
                Barrier();
#endif
                if constexpr (IsCopyFromUB<INPUT_TYPE, MM_CFG>()) {
                    LocalTensor<SrcT> src;
                    src.SetAddr(MATMUL_MODULE(MatmulTensorInfo)->GetLocalTensor().address_);
                    CopyTileToCubeFromUB<false, IS_INTRA_BLOCK>(
                        dst, src, curRow, curCol, tileHeight, tileWidth / widthFactor, widthFactor);
                } else {
                    GlobalTensor<SrcT> src;
                    src.SetGlobalBuffer(
                        MATMUL_MODULE(MatmulTensorInfo)->template GetGlobalTensor<IS_INTRA_BLOCK>().address_);
                    CopyTileToCubeFromGM(dst, src, curRow, curCol, tileHeight, tileWidth / widthFactor,
                        MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<false>(),
                        MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<false>(),
                        MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<false, IS_INTRA_BLOCK>(),
                        MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<false, IS_INTRA_BLOCK>() / widthFactor,
                        MATMUL_MODULE(CopyCubeInParams)->template GetStepCol<false>(),
                        MATMUL_MODULE(CopyCubeInParams)->template IsKRowDirec<IS_INTRA_BLOCK>());
                }
            }
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopySparseIdxToCubeFromGM(const LocalTensor<uint8_t>& dst, const GlobalTensor<uint8_t>& src,
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth)
    {
        ASCENDC_ASSERT(IsTranspose<IS_INTRA_BLOCK>(),
            { KERNEL_LOG(KERNEL_ERROR, "SparseMatmul only support B matrix transpose is true."); });
        int32_t baseHeight = MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<true>();
        int32_t baseWidth = MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<true>() >> 3;
        int32_t orgHeight = MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<true, IS_INTRA_BLOCK>();
        int32_t row = curCol * baseHeight;
        int32_t col = curRow * baseWidth;
        int32_t height = tileWidth;
        int32_t width = tileHeight >> 2;
        constexpr int32_t c0Size = AuxGetC0Size<int32_t>(); // Idx Matrix c0Size=8
        ASCENDC_ASSERT((orgHeight >= height), {
            KERNEL_LOG(KERNEL_ERROR,
                "NZ2NZ height larger than origin matrix height, orgHeight is %d, which should be no less than height %d.",
                orgHeight, height);
        });
        int32_t alignedGRow = Ceil(orgHeight, BLOCK_CUBE) * BLOCK_CUBE;
        int64_t srcOffset = (int64_t)row * (int64_t)c0Size + (int64_t)col * (int64_t)alignedGRow;
        // height direction need to be 16 aligned
        auto alignedHeight = Ceil(height, BLOCK_CUBE) * BLOCK_CUBE;
        int32_t blockLen = (alignedHeight * c0Size * sizeof(uint8_t)) / ONE_BLK_SIZE;
        int32_t srcStride = ((alignedGRow - alignedHeight) * c0Size * sizeof(uint8_t)) / ONE_BLK_SIZE;
        uint16_t nburst = Ceil(width, c0Size);
        if (srcStride >= UINT16_MAX) {
            for (int32_t i = 0; i < nburst; ++i) {
                DataCopy(dst[i * alignedHeight * c0Size], src[srcOffset], {1, static_cast<uint16_t>(blockLen), 0, 0});
                srcOffset += orgHeight * c0Size;
            }
        } else {
            DataCopy(
                dst, src[srcOffset], {nburst, static_cast<uint16_t>(blockLen), static_cast<uint16_t>(srcStride), 0});
        }
    }

private:
    constexpr static int32_t c0Size_ = AuxGetC0Size<SrcT>();

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline bool IsTranspose()
    {
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            if constexpr (INPUT_TYPE::format == CubeFormat::COLUMN_MAJOR) {
                return !MATMUL_MODULE(MatmulShapeInfo)->template IsTransposeA<IS_INTRA_BLOCK>();
            } else {
                return MATMUL_MODULE(MatmulShapeInfo)->template IsTransposeA<IS_INTRA_BLOCK>();
            }
        } else {
            if constexpr (INPUT_TYPE::format == CubeFormat::COLUMN_MAJOR) {
                return !MATMUL_MODULE(MatmulShapeInfo)->template IsTransposeB<IS_INTRA_BLOCK>();
            } else {
                return MATMUL_MODULE(MatmulShapeInfo)->template IsTransposeB<IS_INTRA_BLOCK>();
            }
        }
    }

    template <bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyTileToCubeByCallBack(
        const LocalTensor<TransT>& dst, int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth)
    {
#ifdef ASCENDC_CPU_DEBUG
        if (INPUT_TYPE::TAG == InputTypeTag::A && IMPL::CallBack::CopyA1Ptr) {
            LocalTensor<int8_t> tmpDst = dst.template ReinterpretCast<int8_t>();
            (IMPL::CallBack::CopyA1Ptr)(tmpDst,
                reinterpret_cast<__gm__ void*>(
                    MATMUL_MODULE(MatmulTensorInfo)->template GetGlobalTensor<IS_INTRA_BLOCK>().address_),
                curRow, curCol, tileHeight, tileWidth, MATMUL_MODULE(MatmulUserDefineInfo)->GetUserDefineInfo(),
                MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData());
        } else if (INPUT_TYPE::TAG == InputTypeTag::B && IMPL::CallBack::CopyB1Ptr) {
            LocalTensor<int8_t> tmpDst = dst.template ReinterpretCast<int8_t>();
            (IMPL::CallBack::CopyB1Ptr)(tmpDst,
                reinterpret_cast<__gm__ void*>(
                    MATMUL_MODULE(MatmulTensorInfo)->template GetGlobalTensor<IS_INTRA_BLOCK>().address_),
                curRow, curCol, tileHeight, tileWidth, MATMUL_MODULE(MatmulUserDefineInfo)->GetUserDefineInfo(),
                MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData());
        }
#else
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A && IMPL::CallBack::CopyA1Ptr) {
            LocalTensor<int8_t> tmpDst = dst.template ReinterpretCast<int8_t>();
            (IMPL::CallBack::CopyA1Ptr)(tmpDst,
                reinterpret_cast<__gm__ void*>(
                    MATMUL_MODULE(MatmulTensorInfo)->template GetGlobalTensor<IS_INTRA_BLOCK>().address_),
                curRow, curCol, tileHeight, tileWidth, MATMUL_MODULE(MatmulUserDefineInfo)->GetUserDefineInfo(),
                MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData());
        } else if constexpr (INPUT_TYPE::TAG == InputTypeTag::B && IMPL::CallBack::CopyB1Ptr) {
            LocalTensor<int8_t> tmpDst = dst.template ReinterpretCast<int8_t>();
            (IMPL::CallBack::CopyB1Ptr)(tmpDst,
                reinterpret_cast<__gm__ void*>(
                    MATMUL_MODULE(MatmulTensorInfo)->template GetGlobalTensor<IS_INTRA_BLOCK>().address_),
                curRow, curCol, tileHeight, tileWidth, MATMUL_MODULE(MatmulUserDefineInfo)->GetUserDefineInfo(),
                MATMUL_MODULE(MatmulUserDefineInfo)->GetSelfDefineData());
        }
#endif
    }

    __aicore__ inline void CopyND2NZForInt8(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, int32_t baseHeight, int32_t baseWidth,
        int32_t orgHeight, int32_t orgWidth, int32_t stepCol, bool iskRowDirec)
    {
        if (tileWidth < baseWidth || baseWidth % c0Size_ == 0 || stepCol == 1) {
            MATMUL_MODULE(DataCopyWrapper)
                ->CopyND2NZ(dst, src, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgWidth, 1, 0, 0,
                    iskRowDirec);
            PadZeroForL1(dst, tileHeight, tileWidth, baseHeight, baseWidth, CeilAlign(tileWidth, B8_C0SIZE));
        } else {
            if ((stepCol - 1) * baseWidth > tileWidth) {
                stepCol = Ceil(tileWidth, baseWidth);
            }
            int32_t dstNzMatrixStride = CeilAlign(baseWidth, c0Size_) * CeilAlign(tileHeight, c0Size_);
            if constexpr (HasScalePosition<INPUT_TYPE>::value) {
                if constexpr (INPUT_TYPE::isTrans && INPUT_TYPE::TAG == InputTypeTag::A
                              || INPUT_TYPE::TAG == InputTypeTag::B) {
                    // A transpose [K, M], B not transpose [K, N] scenario,
                    // K needs to be aligned to 64 in the height direction
                    if (MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA()
                        || MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB()) {
                        dstNzMatrixStride = CeilAlign(baseWidth, B8_C0SIZE) * CeilAlign(tileHeight, MX_BASEK_FACTOR);
                    }
                }
            }
            if (likely(dstNzMatrixStride <= UINT16_MAX)) {
                MATMUL_MODULE(DataCopyWrapper)
                    ->CopyND2NZ(dst, src, curRow * baseHeight, curCol * baseWidth, tileHeight, baseWidth, orgWidth,
                        stepCol - 1, baseWidth, dstNzMatrixStride, iskRowDirec);
                MATMUL_MODULE(DataCopyWrapper)
                    ->CopyND2NZ(dst[(stepCol - 1) * dstNzMatrixStride], src, curRow * baseHeight,
                        (curCol + stepCol - 1) * baseWidth, tileHeight, tileWidth - (stepCol - 1) * baseWidth, orgWidth,
                        1, 0, 0, iskRowDirec);
                PadZeroForL1(
                    dst, tileHeight, tileWidth, baseHeight, baseWidth, CeilAlign(tileWidth, B8_C0SIZE) * stepCol);
            } else {
                int32_t colIndex = curCol * baseWidth;
                int32_t dstOffset = 0;
                for (int i = 0; i < stepCol; ++i) {
                    if (i == stepCol - 1) {
                        baseWidth = (tileWidth - (stepCol - 1) * baseWidth) == 0 ?
                                        baseWidth :
                                        tileWidth - (stepCol - 1) * baseWidth;
                    }
                    MATMUL_MODULE(DataCopyWrapper)
                        ->CopyND2NZ(dst[dstOffset], src, curRow * baseHeight, colIndex, tileHeight, baseWidth, orgWidth,
                            1, 0, 0, iskRowDirec);
                    colIndex += baseWidth;
                    dstOffset += dstNzMatrixStride;
                }
                PadZeroForL1(
                    dst, tileHeight, tileWidth, baseHeight, baseWidth, CeilAlign(tileWidth, B8_C0SIZE) * stepCol);
            }
        }
    }

    __aicore__ inline void PadZeroForL1(const LocalTensor<TransT>& dst, int32_t tileHeight, int32_t tileWidth,
        int32_t baseHeight, int32_t baseWidth, int32_t padZeroSize)
    {
        // [K, M] or [K, N] pad outer axis for K
        if constexpr (HasScalePosition<INPUT_TYPE>::value) {
            if constexpr (INPUT_TYPE::isTrans && INPUT_TYPE::TAG == InputTypeTag::A) {
                if (MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA()) {
                    PadZerosForOuterAxis(dst, tileHeight, padZeroSize,
                        MATMUL_MODULE(CopyCubeInParams)->template GetStepCol<true>() * baseHeight, baseWidth);
                }
            }

            if constexpr (INPUT_TYPE::TAG == InputTypeTag::B) {
                if (!MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB()) {
                    PadZerosForOuterAxis(dst, tileHeight, padZeroSize,
                        MATMUL_MODULE(CopyCubeInParams)->template GetStepCol<true>() * baseHeight, baseWidth);
                }
            }

            // When Ceil(K, 32) is odd, 32B zeros are padded at the end of the K direction after DataCopyND2NZ for B8.
            if constexpr (IsSupportB8<TransT>() && (INPUT_TYPE::TAG == InputTypeTag::A)) {
                if (!MATMUL_MODULE(MatmulShapeInfo)->IsTransposeA()) {
                    // [M, K]
                    PadOneBlockInnerAxis(dst, tileHeight, tileWidth);
                }
            }
            if constexpr (IsSupportB8<TransT>() && (INPUT_TYPE::TAG == InputTypeTag::B && INPUT_TYPE::isTrans)) {
                if (MATMUL_MODULE(MatmulShapeInfo)->IsTransposeB()) {
                    // [N, K]
                    PadOneBlockInnerAxis(dst, tileHeight, tileWidth);
                }
            }
        }
    }

    template <typename T>
    __aicore__ inline void PadZerosForOuterAxis(const LocalTensor<T>& dst, const int32_t height, const int32_t width,
        const int32_t baseHeight, const int32_t baseWidth)
    {
        if (height < baseHeight) {
            const uint32_t alignBaseHeight = CeilAlign(height, MX_BASEK_FACTOR);
            auto padTensor = dst.template ReinterpretCast<uint16_t>();
            InitConstValueParams<uint16_t> initConstValueParams;
            initConstValueParams.repeatTimes = Ceil(width, B8_C0SIZE);
            initConstValueParams.blockNum = alignBaseHeight - height;
            initConstValueParams.dstGap = height;
            initConstValueParams.initValue = 0;
            InitConstValue(padTensor[height * BLOCK_CUBE], initConstValueParams);
        }
    }

    template <typename T>
    __aicore__ inline void PadOneBlockInnerAxis(const LocalTensor<T>& dst, const int32_t height, const int32_t width)
    {
        // pad the K direction tail of matrix A/B with 32B zeros
        if (CeilDivision(width, MX_K_FACTOR) % 2 != 0) {
            const uint32_t heightAlign = CeilAlign(height, BLOCK_CUBE);
            auto padTensor = dst.template ReinterpretCast<uint16_t>();
            InitConstValueParams<uint16_t> initConstValueParams;
            initConstValueParams.repeatTimes = heightAlign;
            initConstValueParams.blockNum = 1;
            initConstValueParams.initValue = 0;
            uint32_t dstOffset = heightAlign * (CeilAlign(width, MX_K_FACTOR) / 2);
            InitConstValue(padTensor[dstOffset], initConstValueParams);
        }
    }

    __aicore__ inline void CopyDN2NZForInt8(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, int32_t baseHeight, int32_t baseWidth,
        int32_t orgHeight, int32_t orgWidth, int32_t stepCol, bool iskRowDirec)
    {
        if (tileWidth < baseWidth || baseWidth % c0Size_ == 0 || stepCol == 1) {
            MATMUL_MODULE(DataCopyWrapper)
                ->CopyDN2NZ(dst, src, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgWidth, 1, 0, 0,
                    iskRowDirec);
        } else {
            if ((stepCol - 1) * baseWidth > tileWidth) {
                stepCol = Ceil(tileWidth, baseWidth);
            }
            int32_t dstNzMatrixStride = CeilAlign(baseWidth, c0Size_) * CeilAlign(tileHeight, c0Size_);
            if (likely(dstNzMatrixStride <= UINT16_MAX)) {
                MATMUL_MODULE(DataCopyWrapper)
                    ->CopyDN2NZ(dst, src, curRow * baseHeight, curCol * baseWidth, tileHeight, baseWidth, orgWidth,
                        stepCol - 1, baseWidth, dstNzMatrixStride, iskRowDirec);
                MATMUL_MODULE(DataCopyWrapper)
                    ->CopyDN2NZ(dst[(stepCol - 1) * dstNzMatrixStride], src, curRow * baseHeight,
                        (curCol + stepCol - 1) * baseWidth, tileHeight, tileWidth - (stepCol - 1) * baseWidth, orgWidth,
                        1, 0, 0, iskRowDirec);
            } else {
                int32_t colIndex = curCol * baseWidth;
                int32_t dstOffset = 0;
                for (int i = 0; i < stepCol; ++i) {
                    if (i == stepCol - 1) {
                        baseWidth = tileWidth - (stepCol - 1) * baseWidth;
                    }
                    MATMUL_MODULE(DataCopyWrapper)
                        ->CopyDN2NZ(dst[dstOffset], src, curRow * baseHeight, colIndex, tileHeight, baseWidth, orgWidth,
                            1, 0, 0, iskRowDirec);
                    colIndex += baseWidth;
                    dstOffset += dstNzMatrixStride;
                }
            }
        }
    }

    __aicore__ inline void CopyTileToCubeFromGM(const LocalTensor<TransT>& dst, const GlobalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, int32_t baseHeight, int32_t baseWidth,
        int32_t orgHeight, int32_t orgWidth, int32_t stepCol, bool iskRowDirec)
    {
        if constexpr (INPUT_TYPE::format == CubeFormat::ND) {
            if constexpr (sizeof(TransT) == sizeof(int8_t)) {
                CopyND2NZForInt8(dst, src, curRow, curCol, tileHeight, tileWidth, baseHeight, baseWidth, orgHeight,
                    orgWidth, stepCol, iskRowDirec);
            } else {
                MATMUL_MODULE(DataCopyWrapper)
                    ->CopyND2NZ(dst, src, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgWidth);
            }
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
        } else if constexpr (INPUT_TYPE::format == CubeFormat::COLUMN_MAJOR) {
            if constexpr (sizeof(TransT) == sizeof(int8_t)) {
                CopyDN2NZForInt8(dst, src, curRow, curCol, tileHeight, tileWidth, baseHeight, baseWidth, orgHeight,
                    orgWidth, stepCol, iskRowDirec);
            } else {
                MATMUL_MODULE(DataCopyWrapper)
                    ->CopyDN2NZ(dst, src, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgWidth);
            }
#endif
        } else if constexpr (INPUT_TYPE::format == CubeFormat::NZ) {
            MATMUL_MODULE(DataCopyWrapper)
                ->CopyNZ2NZ(
                    dst, src, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgHeight, iskRowDirec);
        } else if constexpr (INPUT_TYPE::format == CubeFormat::VECTOR) {
            MATMUL_MODULE(DataCopyWrapper)->CopyVector2A1(dst, src, curCol * baseWidth, Ceil(tileWidth, c0Size_));
        } else if constexpr (INPUT_TYPE::format == CubeFormat::SCALAR) {
            return;
        } else {
            ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "Unsupported Matmul format type."); });
        }
    }

    template <bool IS_TRANS = false, bool IS_INTRA_BLOCK = false>
    __aicore__ inline void CopyTileToCubeFromUB(const LocalTensor<TransT>& dst, const LocalTensor<SrcT>& src,
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, int32_t widthFactor)
    {
#if __CCE_AICORE__ != 300
        ASCENDC_ASSERT(false, { KERNEL_LOG(KERNEL_ERROR, "CopyTileToCubeFromUB only support input from UB."); });
#else
        auto baseHeight = MATMUL_MODULE(CopyCubeInParams)->template GetBaseHeight<IS_TRANS>();
        auto baseWidth = MATMUL_MODULE(CopyCubeInParams)->template GetBaseWidth<IS_TRANS>();
        auto orgHeight = MATMUL_MODULE(CopyCubeInParams)->template GetOrgHeight<IS_TRANS, IS_INTRA_BLOCK>();
        auto orgWidth = MATMUL_MODULE(CopyCubeInParams)->template GetOrgWidth<IS_TRANS, IS_INTRA_BLOCK>() / widthFactor;
        auto iskRowDirec = MATMUL_MODULE(CopyCubeInParams)->template IsKRowDirec<IS_INTRA_BLOCK>();
        if constexpr (INPUT_TYPE::format == CubeFormat::ND) {
            MATMUL_MODULE(DataCopyWrapper)
                ->CopyND2NZ(dst, src, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgWidth);
        } else if constexpr (INPUT_TYPE::format == CubeFormat::NZ) {
            MATMUL_MODULE(DataCopyWrapper)
                ->CopyNZ2NZ(dst, src, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgHeight);
        } else if constexpr (INPUT_TYPE::format == CubeFormat::VECTOR) {
            ASCENDC_ASSERT(false, {
                KERNEL_LOG(KERNEL_ERROR, "When input format is VECTOR, only support A transpose and B untranspose.");
            });
            MATMUL_MODULE(DataCopyWrapper)->CopyVector2A1(dst, src, curCol * baseWidth, Ceil(tileWidth, c0Size_));
        } else if constexpr (INPUT_TYPE::format == CubeFormat::SCALAR) {
            return;
        } else {
            ASCENDC_ASSERT(
                false, { KERNEL_LOG(KERNEL_ERROR, "MatmulApi only support input format ND/NZ/VECTOR/SCALAR."); });
        }
#endif
    }
};
} // namespace Detail
} // namespace Impl
} // namespace AscendC
#endif // AICORE_ADV_API_DETAIL_MATMUL_STAGE_COPY_CUBE_IN_COPY_TILE_TO_CUBE_COPY_TILE_TO_CUBE_COMMON_H
