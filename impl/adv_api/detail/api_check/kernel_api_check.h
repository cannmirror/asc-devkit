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
 * \file kernel_api_check.h
 * \brief
 */
#ifndef IMPL_API_CHECK_KERNEL_API_CHECK_H_
#define IMPL_API_CHECK_KERNEL_API_CHECK_H_

#ifdef ASCENDC_CPU_DEBUG

#define EXPAND(...) __VA_ARGS__
#define MERGE_ARGS(first, args) (first, EXPAND args)
#define CHECK_FUNC_HIGHLEVEL_API(API, templateArgs, inputArgs)\
    HighLevelApiCheck::CheckFunc##API<EXPAND templateArgs> MERGE_ARGS(#API, inputArgs)
#else
#define CHECK_FUNC_HIGHLEVEL_API(...)
#endif

#endif // IMPL_API_CHECK_KERNEL_API_CHECK_H_
