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
 * \file kernel_operator_vec_vpadding_impl.h
 * \brief
 */
#if !defined(_ASCENDC_INCLUDE_INTERNAL_HEADERS_)
#pragma message("impl/basic_api/dav_c220/kernel_operator_vec_vpadding_impl.h is an internal header file and must not be used directly. Functions or variables defined in this file may be removed in the future. Please use \"#include \"basic_api/kernel_vec_intf.h\"\" and use public functions or variables defined in interface headers files.")
#define _ASCENDC_INCLUDE_INTERNAL_HEADERS_
#define UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_VPADDING_IMPL_H
#endif
#ifndef ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H
#define ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H
#include "kernel_utils.h"
#include "kernel_struct_unary.h"

namespace AscendC {
template <typename T, bool isSetMask>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    const uint64_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "VectorPadding");
}

template <typename T, bool isSetMask>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "VectorPadding");
}

template <typename T>
__aicore__ inline void VectorPaddingImpl(__ubuf__ T* dst, __ubuf__ T* src, uint8_t padMode, bool padSide,
    const uint32_t count)
{
    ASCENDC_REPORT_NOT_SUPPORT(false, "VectorPadding");
}
} // namespace AscendC
#endif // ASCENDC_MODULE_OPERATOR_VEC_VPADDING_IMPL_H
#if defined(UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_VPADDING_IMPL_H)
#undef _ASCENDC_INCLUDE_INTERNAL_HEADERS_
#undef UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_KERNEL_OPERATOR_VEC_VPADDING_IMPL_H
#endif
