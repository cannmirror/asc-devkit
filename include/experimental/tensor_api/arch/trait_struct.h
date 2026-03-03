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
 * \file trait_struct.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_ARCH_TRAIT_STRUCT_H
#define EXPERIMENTAL_TENSOR_API_ARCH_TRAIT_STRUCT_H

namespace AscendC {
namespace Te {

struct MmadTrait {
   int32_t fmOffset = 0;
   bool kDirectionAlign = false;
   bool cmatrixSource = false;
   bool disableGemv = false;

   __aicore__ constexpr MmadTrait () {};

   __aicore__ constexpr MmadTrait (int32_t fmOffsetIn, bool kDirectionAlignIn, bool cmatrixSourceIn,
         bool disableGemvIn) 
      {
         fmOffset = fmOffsetIn;
         kDirectionAlign = kDirectionAlignIn;
         cmatrixSource = cmatrixSourceIn;
         disableGemv = disableGemvIn;
      };
};

struct MmadParams {
   uint8_t unitFlag;
   bool cmatrixInitVal;

   __aicore__ MmadParams() {};

   __aicore__ constexpr MmadParams(uint8_t unitFlagIn, bool cmatrixInitValIn) : 
      unitFlag(unitFlagIn), cmatrixInitVal(cmatrixInitValIn){};
};

struct DataCopyTrait {};

struct FixpipeTrait {
    __aicore__ constexpr FixpipeTrait() {}
    __aicore__ constexpr FixpipeTrait(
        QuantMode_t quantPreIn,
        bool enableReluIn,
        bool enableChannelSplitIn,
        uint8_t unitFlagIn,
        uint8_t dualDstCtlIn
    ) :
        quantPre(quantPreIn),
        enableRelu(enableReluIn),
        enableChannleSplit(enableChannelSplitIn),
        unitFlag(unitFlagIn),
        dualDstCtl(dualDstCtlIn)
    {}

    QuantMode_t quantPre = QuantMode_t::NoQuant;
    bool enableRelu = false;
    bool enableChannleSplit = false;
    uint8_t unitFlag = false;
    uint8_t dualDstCtl = false;
};

struct LoadDataTrait {
    __aicore__ constexpr LoadDataTrait() {}

    __aicore__ constexpr LoadDataTrait(const bool transposedIn) : transposed(transposedIn) {}

    bool transposed = false;
};

} // namespace Te
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_ARCH_TRAIT_STRUCT_H