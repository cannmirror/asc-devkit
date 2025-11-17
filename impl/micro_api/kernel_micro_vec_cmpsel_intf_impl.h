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
 * \file kernel_micro_vec_cmpsel_intf_impl.h
 * \brief
 */
#ifndef ASCENDC_KERNEL_MICRO_VEC_CMPSEL_INTERFACE_IMPL_H
#define ASCENDC_KERNEL_MICRO_VEC_CMPSEL_INTERFACE_IMPL_H

#if __NPU_ARCH__ == 2103
#include "micro_api/dav_l210/kernel_micro_vec_cmpsel_impl.h"
#elif __NPU_ARCH__ == 3003
#include "micro_api/dav_l300/kernel_micro_vec_cmpsel_impl.h"
#elif __NPU_ARCH__ == 3103
#include "micro_api/dav_l310/kernel_micro_vec_cmpsel_impl.h"
#elif __NPU_ARCH__ == 3113
#include "micro_api/dav_l311/kernel_micro_vec_cmpsel_impl.h"
#else
#include "micro_api/dav_c310/kernel_micro_vec_cmpsel_impl.h"
#endif

namespace AscendC {
namespace MicroAPI {
template <typename T, CMPMODE mode, typename U>
__simd_callee__ inline void Compare(MaskReg &dstMask, U &srcReg0, U &srcReg1, MaskReg &mask)
{
    CompareImpl<T, mode, U>(dstMask, srcReg0, srcReg1, mask);
}

template <typename T, CMPMODE mode, typename U, typename S>
__simd_callee__ inline void CompareScalar(MaskReg &dstMask, U &srcReg, S scalar, MaskReg &mask)
{
    CompareScalarImpl<T, mode, U, S>(dstMask, srcReg, scalar, mask);
}

template <typename T, typename U>
__simd_callee__ inline void Select(U &dstReg, U &srcReg0, U &srcReg1, MaskReg &mask)
{
    SelectImpl<T, U>(dstReg, srcReg0, srcReg1, mask);
}
}  // namespace MicroAPI
}  // namespace AscendC
#endif  // ASCENDC_KERNEL_MICRO_VEC_CMPSEL_INTERFACE_IMPL_H