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
 * \file cube_compute_struct.h
 * \brief
 */
#ifndef EXPERIMENTAL_TENSOR_API_ARCH_CUBE_COMPUTE_CUBE_COMPUTE_STRUCT_H
#define EXPERIMENTAL_TENSOR_API_ARCH_CUBE_COMPUTE_CUBE_COMPUTE_STRUCT_H

#include "include/experimental/tensor_api/utils/utils.h"

namespace AscendC {
namespace Te {

struct MmadTrait {
    int32_t fmOffset = 0;
    bool enSsparse = false;
    bool enWinogradA = false;
    bool enWinogradB = false;
    int8_t unitFlag = 0;
    bool kDirectionAlign = false;
    bool cmatrixSource = false;
    bool cmatrixInitVal = true;

   __aicore__ constexpr MmadTrait () {};

   __aicore__ constexpr MmadTrait (int32_t fmOffsetIn, bool enSsparseIn, bool enWinogradAIn, bool enWinogradBIn, 
      int8_t unitFlagIn, bool kDirectionAlignIn, bool cmatrixSourceIn, bool cmatrixInitValIn) 
   {
      fmOffset = fmOffsetIn;
      enSsparse = enSsparseIn;
      enWinogradA = enWinogradAIn;
      enWinogradB = enWinogradBIn;
      unitFlag = unitFlagIn;
      kDirectionAlign = kDirectionAlignIn;
      cmatrixSource = cmatrixSourceIn;
      cmatrixInitVal = cmatrixInitValIn;
   };
};
constexpr MmadTrait DEFAULT_MMAD_TRAIT; 

} // namespace Te
} // namespace AscendC

#endif // EXPERIMENTAL_TENSOR_API_ARCH_CUBE_COMPUTE_CUBE_COMPUTE_STRUCT_H