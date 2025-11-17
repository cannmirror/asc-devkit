/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
/*!
 * \file tile_copy.h
 * \brief
 */
#ifndef CONV_TILE_TILE_COPY_H
#define CONV_TILE_TILE_COPY_H
#include "../utils/conv_arch.h"

namespace Act {
namespace Conv {
namespace Tile {

template <class ArchTag, class DispatchPolicy = void>
struct TileCopy {};
// 定义SOC结构体
struct Ascend910_95 {};

} // namespace Tile
} // namespace Conv
} // namespace Act
#endif

