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
 * \file ascc_compile_host.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_COMPILE_HOST_H__
#define __INCLUDE_ASCC_COMPILE_HOST_H__
#include <vector>
#include <string>

#include "ascc_types.h"
#include "ascc_compile_base.h"
#include "ascc_global_env_manager.h"
#include "ascc_argument_manager.h"
#include "ascc_utils.h"

namespace Ascc {
class AsccCompileHost : public AsccCompileBase {
public:
    explicit AsccCompileHost(const CompileArgs& args) : AsccCompileBase(args) {};
    AsccStatus Compile() override
    {
        MergeOption();
        const auto& command = args_.GetCmd(AsccGlobalEnvManager::GetInstance().ascendCompiler);
        return AsccCompileBase::ExecuteCompile(command);
    };

    void MergeOption() override
    {
        const auto &inputArgs = Ascc::AsccArgumentManager::GetInstance().GetCompileOptions();
        MergeCommonOption(AsccGlobalEnvManager::GetInstance().commonHostArgs);
        args_.options.insert(args_.options.end(), options_.begin(), options_.end());
        args_.incPaths.insert(args_.incPaths.begin(), inputArgs.incPaths.begin(), inputArgs.incPaths.end());
        args_.definitions.insert(args_.definitions.end(), inputArgs.definitions.begin(), inputArgs.definitions.end());
        args_.options.insert(args_.options.end(), inputArgs.options.begin(), inputArgs.options.end());
        args_.incFiles.insert(args_.incFiles.end(), inputArgs.incFiles.begin(), inputArgs.incFiles.end());
        args_.incPaths.emplace_back(Ascc::GetFilePath(args_.file));
        args_.incPaths.emplace_back(Ascc::GetFilePath(AsccArgumentManager::GetInstance().GetInputFile()));
        args_.options.emplace_back(AsccCompileBase::GetDependencyCmd());

        // Current use bisheng host. Default of bisheng is O2. bisheng not support O1, thus use O2 instead
        const std::string optLevel = Ascc::AsccArgumentManager::GetInstance().GetOptimizeLevel();
        if (optLevel != "O1") {
            args_.options.emplace_back("-" + optLevel);
        } else {
            args_.options.emplace_back("-O2");
        }

        if (Ascc::AsccArgumentManager::GetInstance().DebugRequested()) {
            args_.options.emplace_back("-g");
        }
    }
private:
    const std::vector<std::string> options_ = {
        "-x cce --cce-host-only",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
        "-fPIC -std=c++17 -fcce-kernel-launch-custom",
        "-D__NPU_HOST__"
    };
};
}
#endif // __INCLUDE_ASCC_COMPILE_HOST_H__