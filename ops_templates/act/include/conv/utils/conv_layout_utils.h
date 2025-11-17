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
 * \file conv_layout_utils.h
 * \brief
 */

#ifndef UTILS_CONV_LAYOUT_UTILS_H
#define UTILS_CONV_LAYOUT_UTILS_H

#include "./conv_integral_constant.h"

namespace Act {
namespace Conv {
enum class ConvFormat : std::uint8_t {
    NCHW = 0,
    NHWC = 1,
    ND = 2,
    MAX = 0xff
};

namespace layout {
struct NCHW {
    static constexpr ConvFormat format = ConvFormat::NCHW;
};

struct ND {
    static constexpr ConvFormat format = ConvFormat::ND;
};
} // namespace layout

enum ConvOutputOrder {
    HWMode = 0,
    MMode = 1,
    MAX = 0xff
};

namespace order {
struct OutputHWMode {
    static constexpr uint8_t order = static_cast<uint8_t>(ConvOutputOrder::HWMode);
};

struct OutputMMode {
    static constexpr uint8_t order = static_cast<uint8_t>(ConvOutputOrder::MMode);
};
} // namespace order

} // namespace Conv
} // namespace Act
#endif