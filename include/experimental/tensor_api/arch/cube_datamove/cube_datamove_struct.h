/**
* Copyright (c) 2026 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

/*!
 * \file cube_datamove_struct.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_ARCH_CUBE_DATAMOVE_CUBE_DATAMOVE_STRUCT_H
#define EXPERIMENTAL_TENSOR_API_ARCH_CUBE_DATAMOVE_CUBE_DATAMOVE_STRUCT_H

#include "include/experimental/tensor_api/utils/utils.h"

namespace AscendC {

struct DataCopyTrait {};
constexpr DataCopyTrait DEFAULT_DATA_COPY_TRAIT;


struct FixpipeTrait {
    __aicore__ constexpr FixpipeTrait() {}
    __aicore__ constexpr FixpipeTrait(
        QuantMode_t quantPreIn,
        bool enableReluIn,
        bool enableChannleSplitIn,
        uint8_t unitFlagIn,
        uint8_t dualDstCtlIn
    ) :
        quantPre(quantPreIn),
        enableRelu(enableReluIn),
        enableChannleSplit(enableChannleSplitIn),
        unitFlag(unitFlagIn),
        dualDstCtl(dualDstCtlIn)
    {}

    QuantMode_t quantPre = QuantMode_t::NoQuant;
    bool enableRelu = false;
    bool enableChannleSplit = false;
    uint8_t unitFlag = false;
    uint8_t dualDstCtl = false;
};
constexpr FixpipeTrait DEFAULT_FIXPIPE_TRAIT;

struct LoadDataTrait {
    __aicore__ constexpr LoadDataTrait() {}

    __aicore__ constexpr LoadDataTrait(const bool transposedIn) : transposed(transposedIn) {}

    bool transposed = false;
};
constexpr LoadDataTrait DEFAULT_LOAD_DATA_TRAIT{};
}

#endif // EXPERIMENTAL_TENSOR_API_ARCH_CUBE_DATAMOVE_CUBE_DATAMOVE_STRUCT_H