/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
 
/* !
 * \file kernel_micro_histograms_intf.h
 * \brief
 */
#ifndef ASCENDC_MODULE_MICRO_HISTOGRAMS_INTERFACE_H
#define ASCENDC_MODULE_MICRO_HISTOGRAMS_INTERFACE_H
 
#include "kernel_micro_common_intf.h"

namespace AscendC {
namespace MicroAPI {

template <typename T = DefaultType, typename U = DefaultType, HistogramsBinType mode, HistogramsType type,
    typename S, typename V>
__simd_callee__ inline void Histograms(V &dstReg, S &srcReg, MaskReg &mask);

}
}  // namespace AscendC

#include "impl/micro_api/kernel_micro_histograms_intf_impl.h"
#endif // ASCENDC_MODULE_MICRO_HISTOGRAMS_INTERFACE_H
