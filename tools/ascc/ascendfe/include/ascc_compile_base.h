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
 * \file ascc_compile_base.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_COMPILE_BASE_H__
#define __INCLUDE_ASCC_COMPILE_BASE_H__
#include <memory>

#include "ascc_types.h"
#include "ascc_global_env_manager.h"
#include "ascc_argument_manager.h"

namespace Ascc {
class AsccCompileBase {
public:
    explicit AsccCompileBase(const CompileArgs& args) : args_(args) {};
    virtual ~AsccCompileBase() = default;
    virtual AsccStatus Compile() = 0;
    virtual void MergeOption() = 0;
    void MergeCommonOption(const CompileArgs& commonArgs);
    void AddOption(const std::string& option);
    void AddDefinition(const std::string& definition);
    void AddIncPath(const std::string& path);
    void AddIncFile(const std::string& file);
    void SetCustomOption(const std::string& option);

public:
    CompileArgs args_;
protected:
    AsccStatus ExecuteCompile(const std::string &cmd) const;
    std::string GetDependencyCmd() const;
};
}  // namespace Ascc
#endif // __INCLUDE_ASCC_COMPILE_BASE_H__