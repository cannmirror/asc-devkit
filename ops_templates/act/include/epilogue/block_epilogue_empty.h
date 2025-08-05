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
 * \file block_epilogue_empty.h
 * \brief
 */

#ifndef BLOCK_EPILOGUE_EMPTY_H
#define BLOCK_EPILOGUE_EMPTY_H
#include "kernel_operator.h"
#include "include/epilogue/fusion/default_fusion_op.h"
#include "include/utils/common_utils.h"
#include "include/utils/device_utils.h"
#include "include/utils/status_utils.h"

namespace Act {
namespace Gemm {
namespace Block {

class BlockEpilogueEmpty {
public:
    using BlockShape = AscendC::Shape<int64_t, int64_t, int64_t, int64_t>;
    using BlockCoord = AscendC::Coord<int64_t, int64_t, int64_t, int64_t>;

    struct Arguments {
        Arguments() = default;
    };

    struct Params {
        Params() = default;
    };

    __aicore__ inline BlockEpilogueEmpty() {}

    __aicore__ inline void run()
    {
        return;
    }

    __aicore__ inline void operator()(Arguments const& params)
    {
        run();
    }

    __host_aicore__ static Params InitParams(Arguments const& args, GM_ADDR workspaceGm)
    {
        Params params = {};
        return params;
    }

    __host_aicore__ static size_t GetWorkSpaceSize(int64_t blockNum, int64_t l1M, int64_t l1N)
    {
        return 0;
    }

    __host_aicore__ static Status CheckArgs(Arguments const& args)
    {
        return Status::success;
    }

    __aicore__ inline void operator()(BlockShape& blockShape, BlockCoord& blockCoord, int64_t dstStartOffset = 0,
                                      int64_t srcStartOffset = 0)
    {
        return;
    }
};
} // namespace Block
} // namespace Gemm
} // namespace Act
#endif
