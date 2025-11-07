/*
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file asc_interface.h
 * \brief
 */


#ifndef __INCLUDE_EXTERNAL_ASC_INTERFACE_H__
#define __INCLUDE_EXTERNAL_ASC_INTERFACE_H__

#include "asc_struct.h"

namespace AscPlugin {

/**
 * @brief Get include compile options
 * @param[in]   result            PreCompileResult contains -I options
 * @return int32_t that indicates success or not: ASC_SUCCESS / ASC_FAILURE / ASC_NULLPTR
 */
extern "C" __attribute__((visibility("default"))) int32_t PluginGetPreCompileOpts(const char** result);

/**
 * @brief Generate binaryRegisterCode
 * @param[in]   result            PrologueResult containing <binary register code, origin function Prefix, device stub
 * function Prefix>
 * @param[in]   config            config compile options received from compiler
 * @return int32_t that indicates success or not: ASC_SUCCESS / ASC_FAILURE / ASC_NULLPTR
 */
extern "C" __attribute__((visibility("default"))) int32_t PluginPrologue(const char** result, const char* config);

/**
 * @brief Generate ​​device/host stub functions​​, ​​metadata segments​​, and ​​kernel type
 * descriptors​​ from input kernel function specifications.
 * @param[in]   result          Json Format Info C-style String, GenKernelResult containing <host stub, device stub,
 *                                  meta info, kernel type>
 * @param[in]   info            Json Format Info C-style String, Kernel function specifications
 * @return int32_t that indicates success or not: ASC_SUCCESS / ASC_FAILURE / ASC_NULLPTR
 */
extern "C" __attribute__((visibility("default"))) int32_t PluginGenKernel(const char** result, const char* info);


/**
 * @brief Generate function register code and compile options for device and host
 * @param[in]   result          char** for EpilogueResult containing <function register code, preinclude paths,
 device / host compile options>
 * @return int32_t that indicates success or not: ASC_SUCCESS / ASC_FAILURE / ASC_NULLPTR
 */
extern "C" __attribute__((visibility("default"))) int32_t PluginEpilogue(const char** result);

/**
 * @brief return default link options for compiling .so / executable
 * @param[in]   result          char** for FatbinLinkResult json string containing link paths, link options
 * @return int32_t that indicates success or not: ASC_SUCCESS / ASC_FAILURE / ASC_NULLPTR
 */
extern "C" __attribute__((visibility("default"))) int32_t PluginFatbinLink(const char** result);

} // namespace AscPlugin
#endif // __INCLUDE_EXTERNAL_ASC_INTERFACE_H__