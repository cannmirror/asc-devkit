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
 * \file asc_ast_device_analyzer.h
 * \brief
 */
#ifndef __INCLUDE_ASC_AST_DEVICE_ANALYZER_H__
#define __INCLUDE_ASC_AST_DEVICE_ANALYZER_H__
#include <stdexcept>
#include <string>
#include "asc_ast_utils.h"

namespace AscPlugin {
class AscAstDeviceAnalyzer {
public:
    explicit AscAstDeviceAnalyzer(const std::string &source);
    int32_t Process();

private:
    void InitCompileDeviceArgs(const std::string &source);
    CompileArgs astDeviceArgs_;
};
}  // namespace AscPlugin
#endif