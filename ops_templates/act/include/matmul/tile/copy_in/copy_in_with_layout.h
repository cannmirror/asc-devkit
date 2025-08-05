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
 * \file copy_in_with_layout.h
 * \brief
 */
#ifndef ACT_INCLUDE_MATMUL_TILE_COPY_IN_COPY_IN_WITH_LAYOUT_H
#define ACT_INCLUDE_MATMUL_TILE_COPY_IN_COPY_IN_WITH_LAYOUT_H

#include "../tile_copy_policy.h"

namespace Act {
namespace Gemm {
namespace Tile {
template <class ArchTag, class InputType, class DstTrait, class SrcTrait>
struct Copy<ArchTag, CopyWithLayout, InputType, DstTrait, SrcTrait,
            AscendC::Std::enable_if_t<
                SrcTrait::tPos == AscendC::TPosition::GM &&
                (DstTrait::tPos == AscendC::TPosition::A1 || DstTrait::tPos == AscendC::TPosition::B1)    // GM->A1/B1
                && (InputType::format == CubeFormat::ND || InputType::format == CubeFormat::ND_ALIGN)>> { // ND/ND_ALIGN
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::GlobalTensor<SrcTrait>;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord)
    {
        using TransT = typename InputType::T;
        using SrcT = typename SrcTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto srcStride = src.GetTensorTrait().GetLayout().GetStride();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();

        AscendC::Nd2NzParams nd2nzParams;
        nd2nzParams.ndNum = 1;
        nd2nzParams.nValue =
            AscendC::Std::min<int, int>(Get<0, 0>(dstShape) * Get<0, 1>(dstShape), Get<0>(srcShape) - Get<0>(coord));
        nd2nzParams.dValue =
            AscendC::Std::min<int, int>(Get<1, 0>(dstShape) * Get<1, 1>(dstShape), Get<1>(srcShape) - Get<1>(coord));
        nd2nzParams.srcNdMatrixStride = 0;
        nd2nzParams.srcDValue = Get<0>(srcStride);
        nd2nzParams.dstNzC0Stride = Get<0, 0>(dstShape) * Get<0, 1>(dstShape);
        nd2nzParams.dstNzNStride = 1;
        nd2nzParams.dstNzMatrixStride = 0;

        auto offset = src.GetTensorTrait().GetLayout()(coord);

        AscendC::GlobalTensor<SrcT> srcTensor;
        srcTensor.SetGlobalBuffer(src.address_);
        AscendC::LocalTensor<TransT> dstTensor;
        dstTensor.SetAddr(dst.address_);

        AscendC::DataCopy(dstTensor, srcTensor[offset], nd2nzParams);
    }
};

template <class ArchTag, class InputType, class DstTrait, class SrcTrait>
struct Copy<ArchTag, CopyWithLayout, InputType, DstTrait, SrcTrait,
            AscendC::Std::enable_if_t<SrcTrait::tPos == AscendC::TPosition::GM &&
                                      (DstTrait::tPos == AscendC::TPosition::A1 ||
                                       DstTrait::tPos == AscendC::TPosition::B1) // GM->A1/B1
                                      && InputType::format == CubeFormat::NZ>> { // NZ
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::GlobalTensor<SrcTrait>;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord)
    {
        using TransT = typename InputType::T;
        using SrcT = typename SrcTrait::LiteType;

        auto srcShape = src.GetTensorTrait().GetLayout().GetShape();
        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        int32_t alignedGRow = Get<0, 0>(srcShape) * Get<0, 1>(srcShape);
        // height direction need to be 16 aligned
        auto alignHeight = Get<0, 0>(dstShape) * Get<0, 1>(dstShape);
        int32_t blockLen = alignHeight * Get<1, 0>(dstShape) * sizeof(TransT) / 32;                  // 32: c0 byte size
        int32_t srcStride = (alignedGRow - alignHeight) * Get<1, 0>(srcShape) * sizeof(TransT) / 32; // 32: c0 byte size
        uint16_t nburst = Get<1, 1>(dstShape);
        auto offset = src.GetTensorTrait().GetLayout()(coord);
        AscendC::GlobalTensor<SrcT> srcTensor;
        srcTensor.SetGlobalBuffer(src.address_);
        AscendC::LocalTensor<TransT> dstTensor;
        dstTensor.SetAddr(dst.address_);
        AscendC::DataCopy(
            dstTensor, srcTensor[offset],
            {nburst, static_cast<uint16_t>(blockLen), static_cast<uint16_t>(srcStride), static_cast<uint16_t>(0)});
    }
};

template <class ArchTag, class InputType, class DstTrait, class SrcTrait>
struct Copy<ArchTag, CopyWithLayout, InputType, DstTrait, SrcTrait,
            AscendC::Std::enable_if_t<SrcTrait::tPos == AscendC::TPosition::GM &&
                                      (DstTrait::tPos == AscendC::TPosition::A1 ||
                                       DstTrait::tPos == AscendC::TPosition::B1)     // GM->A1/B1
                                      && InputType::format == CubeFormat::VECTOR>> { // VECTOR
    using DstTensor = AscendC::LocalTensor<DstTrait>;
    using SrcTensor = AscendC::GlobalTensor<SrcTrait>;

    __aicore__ Copy() = default;
    __aicore__ ~Copy() = default;

    template <class Coord>
    __aicore__ inline void operator()(DstTensor& dst, SrcTensor& src, const Coord& coord)
    {
        using TransT = typename InputType::T;
        using SrcT = typename SrcTrait::LiteType;

        auto dstShape = dst.GetTensorTrait().GetLayout().GetShape();
        AscendC::DataCopyParams dataCopyInfo;
        dataCopyInfo.blockCount = 1;
        dataCopyInfo.blockLen = Get<1, 1>(dstShape);
        dataCopyInfo.srcStride = 0;
        dataCopyInfo.dstStride = 0;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_VECTOR;
        auto offset = Get<1>(coord);
        AscendC::GlobalTensor<SrcT> srcTensor;
        srcTensor.SetGlobalBuffer(src.address_);
        AscendC::LocalTensor<TransT> dstTensor;
        dstTensor.SetAddr(dst.address_);
        AscendC::DataCopy(dstTensor, srcTensor[offset], dataCopyInfo, enhancedParams);
    }
};
} // namespace Tile
} // namespace Gemm
} // namespace Act
#endif
