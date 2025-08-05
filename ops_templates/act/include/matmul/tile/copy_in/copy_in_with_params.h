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
 * \file copy_in_with_params.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_TILE_COPY_IN_COPY_IN_WITH_PARAMS_H
#define ACT_INCLUDE_MATMUL_TILE_COPY_IN_COPY_IN_WITH_PARAMS_H

#include "../tile_copy_policy.h"

namespace Act {
namespace Gemm {
namespace Tile {
template <class ArchTag, class InputType, const auto& COPY_CFG>
struct Copy<ArchTag, CopyWithParams, void, void, InputType,
            AscendC::Std::enable_if_t<InputType::pos == AscendC::TPosition::GM // input from GM
                                      && InputType::format == CubeFormat::ND>, // ND
            COPY_CFG> {
public:
    using TransT = typename InputType::TRANS_T;
    using SrcT = typename InputType::T;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    __aicore__ inline void operator()(const AscendC::LocalTensor<TransT>& dstLocal,
                                      AscendC::GlobalTensor<SrcT> srcGlobal, int curRow, int curCol, int tileHeight,
                                      int tileWidth, int baseHeight, int baseWidth, int orgHeight, int orgWidth,
                                      bool iskRowDirec)
    {
        if constexpr (sizeof(TransT) == sizeof(int8_t)) {
            CopyND2NZ(dstLocal, srcGlobal, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgWidth, 1,
                      0, 0, iskRowDirec);
        } else {
            CopyND2NZ(dstLocal, srcGlobal, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgWidth);
        }
    }

private:
    __aicore__ inline void CopyND2NZ(const AscendC::LocalTensor<TransT>& dst, const AscendC::GlobalTensor<SrcT>& src,
                                     const int32_t row, const int32_t col, const int32_t height, const int32_t width,
                                     const int32_t gCol, const int32_t ndNum = 1, const int32_t srcNdMatrixStride = 0,
                                     const int32_t dstNzMatrixStride = 0, const bool kAlignToC0Size = false)
    {
        ASCENDC_ASSERT((row >= 0), { KERNEL_LOG(KERNEL_ERROR, "row is %d, which should be no less than 0.", row); });
        ASCENDC_ASSERT((col >= 0), { KERNEL_LOG(KERNEL_ERROR, "col is %d, which should be no less than 0.", col); });
        ASCENDC_ASSERT((height > 0),
                       { KERNEL_LOG(KERNEL_ERROR, "height is %d, which should be no less than 0.", height); });
        ASCENDC_ASSERT((width > 0),
                       { KERNEL_LOG(KERNEL_ERROR, "width is %d, which should be no less than 0.", width); });
        ASCENDC_ASSERT((gCol >= width), {
            KERNEL_LOG(
                KERNEL_ERROR,
                "ND2NZ width larger than origin matrix width, gCol is %d, which should be no less than width %d.", gCol,
                width);
        });
        constexpr static int32_t c0Size = AscendC::AuxGetC0Size<SrcT>();
        int32_t dstNzC0Stride = 0;
        int64_t srcOffset;
        if constexpr (AscendC::IsSameTypeV<TransT, AscendC::int4b_t>) {
            srcOffset = ((int64_t)row * (int64_t)gCol * AscendC::INT4_TWO + (int64_t)col);
        } else {
            srcOffset = ((int64_t)row * (int64_t)gCol + (int64_t)col);
        }
        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = ndNum;
        nd2nzParams.nValue = height;
        nd2nzParams.dValue = width;
        nd2nzParams.srcNdMatrixStride = srcNdMatrixStride;
        nd2nzParams.srcDValue = gCol;

        if (dstNzC0Stride) {
            nd2nzParams.dstNzC0Stride = dstNzC0Stride;
        } else {
            // when k is row(height) axis, int8 type gm->l1 nd2nz should be aligned to 32(c0Size)
            // while float/half type should be aligned to 16
            if (kAlignToC0Size) {
                nd2nzParams.dstNzC0Stride = AscendC::Ceil(height, c0Size) * c0Size;
            } else {
                nd2nzParams.dstNzC0Stride = AscendC::Ceil(height, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;
            }
        }
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = dstNzMatrixStride;
        if constexpr (!AscendC::ToMatmulConfig(COPY_CFG).intrinsicsCheck) {
            AscendC::DataCopy(dst, src[srcOffset], nd2nzParams);
        } else {
            if (gCol >= UINT16_MAX) {
                nd2nzParams.nValue = 1;
                nd2nzParams.srcDValue = width;
                for (int32_t i = 0; i < height; ++i) {
                    AscendC::DataCopy(dst[i * c0Size], src[srcOffset + gCol * i], nd2nzParams);
                }
            } else {
                AscendC::DataCopy(dst, src[srcOffset], nd2nzParams);
            }
        }
    }
};

template <class ArchTag, class InputType, const auto& COPY_CFG>
struct Copy<ArchTag, CopyWithParams, void, void, InputType,
            AscendC::Std::enable_if_t<InputType::pos == AscendC::TPosition::GM // input from GM
                                      && InputType::format == CubeFormat::NZ>, // NZ
            COPY_CFG> {
public:
    using TransT = typename InputType::TRANS_T;
    using SrcT = typename InputType::T;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    __aicore__ inline void operator()(const AscendC::LocalTensor<TransT>& dstLocal,
                                      AscendC::GlobalTensor<SrcT> srcGlobal, int curRow, int curCol, int tileHeight,
                                      int tileWidth, int baseHeight, int baseWidth, int orgHeight, int orgWidth,
                                      bool iskRowDirec)
    {
        CopyNZ2NZ(dstLocal, srcGlobal, curRow * baseHeight, curCol * baseWidth, tileHeight, tileWidth, orgHeight,
                  iskRowDirec);
    }

private:
    __aicore__ inline void CopyNZ2NZ(const AscendC::LocalTensor<TransT>& dst, const AscendC::GlobalTensor<SrcT>& src,
                                     const int32_t row, const int32_t col, const int32_t height, const int32_t width,
                                     const int32_t gRow, const bool kAlignToC0Size = false)
    {
        ASCENDC_ASSERT((gRow >= height), {
            KERNEL_LOG(
                KERNEL_ERROR,
                "NZ2NZ height larger than origin matrix height, gRow is %d, which should be no less than height %d.",
                gRow, height);
        });
        constexpr static int32_t c0Size = AscendC::AuxGetC0Size<SrcT>();
        int32_t alignedGRow = AscendC::Ceil(gRow, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;
        int64_t srcOffset = (int64_t)row * (int64_t)c0Size + (int64_t)col * (int64_t)alignedGRow;
        // height direction need to be 16 aligned
        auto alignHeight = AscendC::Ceil(height, AscendC::BLOCK_CUBE) * AscendC::BLOCK_CUBE;
        int32_t blockLen = alignHeight * c0Size * sizeof(TransT) / AscendC::ONE_BLK_SIZE;
        int32_t srcStride = (alignedGRow - alignHeight) * (c0Size * sizeof(TransT) / AscendC::ONE_BLK_SIZE);

        if (srcStride >= UINT16_MAX) {
            for (int32_t i = 0; i < AscendC::Ceil(width, c0Size); ++i) {
                AscendC::DataCopy(dst[i * alignHeight * c0Size], src[srcOffset + i * gRow * c0Size],
                                  {1, static_cast<uint16_t>(blockLen), 0, 0});
            }
        } else {
            uint16_t nburst = AscendC::Ceil(width, c0Size);
            int32_t dstStride = 0;
            if constexpr (AscendC::IsSameTypeV<TransT, int8_t>) {
                if (kAlignToC0Size) {
                    auto alignHeightC0Size = AscendC::Ceil(height, c0Size) * c0Size;
                    dstStride = alignHeightC0Size - alignHeight;
                }
            }
            AscendC::DataCopy(dst, src[srcOffset],
                              {nburst, static_cast<uint16_t>(blockLen), static_cast<uint16_t>(srcStride),
                               static_cast<uint16_t>(dstStride)});
        }
    }
};

template <class ArchTag, class InputType, const auto& COPY_CFG>
struct Copy<ArchTag, CopyWithParams, void, void, InputType,
            AscendC::Std::enable_if_t<InputType::pos == AscendC::TPosition::GM     // input from GM
                                      && InputType::format == CubeFormat::VECTOR>, // VECTOR
            COPY_CFG> {
public:
    using TransT = typename InputType::TRANS_T;
    using SrcT = typename InputType::T;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    __aicore__ inline void operator()(const AscendC::LocalTensor<TransT>& dstLocal,
                                      AscendC::GlobalTensor<SrcT> srcGlobal, int curRow, int curCol, int tileHeight,
                                      int tileWidth, int baseHeight, int baseWidth, int orgHeight, int orgWidth,
                                      bool iskRowDirec)
    {
        constexpr static int32_t c0Size = AscendC::AuxGetC0Size<SrcT>();
        CopyVector2A1(dstLocal, srcGlobal, curCol * baseWidth, AscendC::Ceil(tileWidth, c0Size));
    }

private:
    __aicore__ inline void CopyVector2A1(const AscendC::LocalTensor<TransT>& dst,
                                         const AscendC::GlobalTensor<SrcT>& src, const int32_t col,
                                         const int32_t blockLen)
    {
        ASCENDC_ASSERT((col >= 0), { KERNEL_LOG(KERNEL_ERROR, "col is %d, which should be no less than 0.", col); });
        ASCENDC_ASSERT((InputType::format == CubeFormat::VECTOR),
                       { KERNEL_LOG(KERNEL_ERROR, "InputType::format should be CubeFormat::VECTOR."); });

        AscendC::DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = 1;
        dataCopyInfo.blockLen = blockLen;
        dataCopyInfo.srcStride = 0;
        dataCopyInfo.dstStride = 0;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_VECTOR;
        AscendC::DataCopy(dst, src[col], dataCopyInfo, enhancedParams);
    }
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
