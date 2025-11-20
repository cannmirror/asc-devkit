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
 * \file kernel_operator_selectwithbytesmask_intf.h
 * \brief
 */
#ifndef LIB_SELECT_KERNEL_OPERATOR_SELECT_WITH_BYTES_MASK_INTF_H
#define LIB_SELECT_KERNEL_OPERATOR_SELECT_WITH_BYTES_MASK_INTF_H
#if (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201 || __NPU_ARCH__ == 2002 || __NPU_ARCH__ == 3101 || \
    __NPU_ARCH__ == 5102)) || defined(__DAV_L311__)
#include "selectwithbytesmask.h"

namespace AscendC {
[[deprecated(__FILE__
    " is deprecated, please use sekectwithbytesmask.h instead!")]] typedef void SelectWithBytesMaskDeprecatedHeader;
using AscendCModuleSelectWithBytesMaskInterface = SelectWithBytesMaskDeprecatedHeader;
} // namespace AscendC
#endif
#endif // LIB_SELECT_KERNEL_OPERATOR_SELECT_WITH_BYTES_MASK_INTF_H
