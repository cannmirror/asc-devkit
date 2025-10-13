/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file copy_cube_in_ubtol1_singleshape.h
 * \brief
 */


#ifndef IMPL_MATMUL_MODULES_STAGE_COPY_CUBE_IN_UBTOL1_SINGLESHAPE_H
#define IMPL_MATMUL_MODULES_STAGE_COPY_CUBE_IN_UBTOL1_SINGLESHAPE_H

#include "../copy_tile_to_cube/copy_tile_to_cube.h"
#include "copy_cube_in_intf.h"

namespace AscendC {
namespace Impl {
namespace Detail {
/*
    CopyCubeIn is considered entirely experimental.
    We retain the freedom to make incompatible changes, but do not guarantee the stability.
    CopyCubeIn is only for internal usage, does not support extension or customized specialization!
*/
template <typename IMPL, class INPUT_TYPE, const auto& MM_CFG>
class CopyCubeIn<IMPL, INPUT_TYPE, MM_CFG, enable_if_t<
GetCopyCubeInType<INPUT_TYPE, MM_CFG>() == CopyCubeInType::UBTOL1_SINGLESHAPE>>
{
    MATMUL_USE_MODULE_ON(CubeInBuffer, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(CopyCubeInParams, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE_ON(MatmulTensorInfo, INPUT_TYPE::TAG);
    MATMUL_USE_MODULE(MatmulShapeInfo);
    using TransT = typename INPUT_TYPE::TRANS_T;
    using SrcT = typename Conditional<IsSameType<TransT, fp8_e8m0_t>::value, fp8_e8m0_t, typename INPUT_TYPE::T>::type;

public:
    __aicore__ inline CopyCubeIn() = default;
    __aicore__ inline ~CopyCubeIn() = default;

    __aicore__ inline void Init() {
        MATMUL_MODULE(CubeInBuffer)->Init(
            MATMUL_MODULE(CopyCubeInParams)->GetBufferSize(), MATMUL_MODULE(CopyCubeInParams)->GetDepth());
    }

    __aicore__ inline void Reset() {}

    __aicore__ inline void SetInput(const LocalTensor<SrcT>& localMatrix, bool isTranspose)
    {
        MATMUL_MODULE(MatmulTensorInfo)->SetLocalTensor(localMatrix, isTranspose);
    }

    __aicore__ inline void SetInput(const GlobalTensor<SrcT>& globalMatrix, bool isTranspose)
    {
        MATMUL_MODULE(MatmulTensorInfo)->SetGlobalTensor(globalMatrix, isTranspose);
    }

    template <typename ScheduleContext = int>
    __aicore__ inline LocalTensor<TransT> LoadData(
        int32_t curRow, int32_t curCol, int32_t tileHeight, int32_t tileWidth, const ScheduleContext& context = {})
    {
        LocalTensor<TransT> l1;

        TBuffAddr tbuffTmp;
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            tbuffTmp.logicPos = (uint8_t)(TPosition::A1);
        } else {
            tbuffTmp.logicPos = (uint8_t)(TPosition::B1);
        }
        tbuffTmp.bufferAddr = MATMUL_MODULE(CubeInBuffer)->GetBufferHeadAddr();

#ifdef ASCENDC_CPU_DEBUG
        if constexpr (INPUT_TYPE::TAG == InputTypeTag::A) {
            tbuffTmp.dataLen = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreM() * MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK() * sizeof(TransT);
        } else {
            tbuffTmp.dataLen = MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreK() * MATMUL_MODULE(MatmulShapeInfo)->GetSingleCoreN() * sizeof(TransT);
        }
        tbuffTmp.absAddr = GetTPipePtr()->GetBaseAddr(static_cast<uint8_t>(TPosition::A1)) + tbuffTmp.bufferAddr;
#endif

        l1.SetAddr(tbuffTmp);
        return l1;
    }

    __aicore__ inline void AllocTensor(int32_t iterIndex = 0) {}

    __aicore__ inline void ClearLoadData(const LocalTensor<TransT>& tensor = NULL_TENSOR<TransT>,
        int32_t curRow = 0, int32_t curCol = 0) {}

    __aicore__ inline void Destroy() {}
};
}  // namespace Detail
}  // namespace Impl
}  // namespace AscendC
#endif // IMPL_MATMUL_MODULES_STAGE_COPY_CUBE_IN_UBTOL1_SINGLESHAPE_H
