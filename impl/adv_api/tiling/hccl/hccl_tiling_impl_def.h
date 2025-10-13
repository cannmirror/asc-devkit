/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file hccl_tiling_impl_def.h
 * \brief
 */
#ifndef IMPL_HCCL_HCCL_TILING_IMPL_DEF_H
#define IMPL_HCCL_HCCL_TILING_IMPL_DEF_H

namespace AscendC {
class HcclTilingImpl {
public:
    uint32_t opType_;
    std::string groupName_;
    std::string algConfig_;
    uint32_t reduceType_ = 0U;
    uint8_t stepSize_ = 0U;
    uint8_t skipLocalRankCopy_ = 0U;
    uint8_t skipBufferWindowCopy_ = 0U;
    uint8_t debugMode_ = 0U;
    uint64_t initTilingAddr_ = 0UL;
    uint16_t queueNum_ = 0U;
    uint16_t commBlockNum_ = 0U;
    uint8_t srcDataType_;
    uint8_t dstDataType_;
};
}  // namespace AscendC
#endif  // IMPL_HCCL_HCCL_TILING_IMPL_DEF_H
