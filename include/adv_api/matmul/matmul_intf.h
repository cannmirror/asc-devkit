/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_intf.h
 * \brief
 */
#ifndef LIB_MATMUL_MATMUL_INTF_H
#define LIB_MATMUL_MATMUL_INTF_H

#include "../../impl/adv_api/detail/kfc/kfc_register_obj.h"
#if __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__)
#include "../../impl/adv_api/detail/kfc/kernel_kfc.h"
#else
#include "include/adv_api/matmul/matmul.h"
#endif

namespace AscendC {
#define REGIST_MATMUL_OBJ_STATIC REGIST_CUBE_OBJ
#define REGIST_MATMUL_OBJ REGIST_CUBE_OBJ
#ifdef ASCENDC_CPU_DEBUG
#if __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__)
#ifdef ASCENDC_CUBE_ONLY
template <
    class A_TYPE, 
    class B_TYPE, 
    class C_TYPE, 
    class BIAS_TYPE = C_TYPE, 
    const auto& MM_CFG = CFG_NORM,
    class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
    MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)
>
using Matmul = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

#else
// Otherwise, use MatmulClient
template <
    class A_TYPE, 
    class B_TYPE, 
    class C_TYPE, 
    class BIAS_TYPE = C_TYPE, 
    const auto& MM_CFG = CFG_NORM,
    class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
    MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)
>

using Matmul = MatmulClient<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

#endif // ASCENDC_CUBE_ONLY

#else
// Use MatmulImpl in other cases
template <
    class A_TYPE, 
    class B_TYPE, 
    class C_TYPE, 
    class BIAS_TYPE = C_TYPE, 
    const auto& MM_CFG = CFG_NORM,
    class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
    MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)
>

using Matmul = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

#endif // __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__)

#else // Not in ASCENDC_CPU_DEBUG mode

#ifdef SPLIT_CORE_CUBE
#ifdef ASCENDC_CUBE_ONLY
// If SPLIT_CORE_CUBE and ASCENDC_CUBE_ONLY modes are enabled, use MatmulImpl
template <
    class A_TYPE, 
    class B_TYPE, 
    class C_TYPE, 
    class BIAS_TYPE = C_TYPE, 
    const auto& MM_CFG = CFG_NORM,
    class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
    MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)
>
using Matmul = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

#else
// Otherwise, use MatmulServiceAux
template <
    class A_TYPE, 
    class B_TYPE, 
    class C_TYPE, 
    class BIAS_TYPE = C_TYPE, 
    const auto& MM_CFG = CFG_NORM,
    class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
    MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)
>
using Matmul = MatmulServiceAux<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

#endif // ASCENDC_CUBE_ONLY

#elif defined(SPLIT_CORE_VEC)
// If SPLIT_CORE_VEC mode is enabled, use MatmulClient
template <
    class A_TYPE, 
    class B_TYPE, 
    class C_TYPE, 
    class BIAS_TYPE = C_TYPE, 
    const auto& MM_CFG = CFG_NORM,
    class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
    MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)
>
using Matmul = MatmulClient<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

#else
// Default case: use MatmulImpl
template <
    class A_TYPE, 
    class B_TYPE, 
    class C_TYPE, 
    class BIAS_TYPE = C_TYPE, 
    const auto& MM_CFG = CFG_NORM,
    class MM_CB = MatmulCallBackFunc<nullptr, nullptr, nullptr>,
    MATMUL_POLICY_DEFAULT_OF(MatmulPolicy)
>
using Matmul = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE, MM_CFG, MM_CB, MATMUL_POLICY>;

#endif // SPLIT_CORE_CUBE || SPLIT_CORE_VEC

#endif // ASCENDC_CPU_DEBUG
} //namespace AscendC
#endif
