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
 * \file ascc_compile_v220.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_COMPILE_V220_H__
#define __INCLUDE_ASCC_COMPILE_V220_H__
#include <vector>
#include <string>

#include "ascc_types.h"
#include "ascc_compile_base.h"
#include "ascc_global_env_manager.h"
#include "ascc_argument_manager.h"
#include "ascc_utils.h"

namespace Ascc {
class AsccCompileV220 : public AsccCompileBase {
public:
    explicit AsccCompileV220(const CompileArgs& args) : AsccCompileBase(args) {};
    AsccStatus Compile() override
    {
        MergeOption();
        const auto& command = args_.GetCmd(AsccGlobalEnvManager::GetInstance().ascendCompiler);
        return AsccCompileBase::ExecuteCompile(command);
    };

    void MergeOption() override
    {
        const auto &inputArgs = Ascc::AsccArgumentManager::GetInstance().GetCompileOptions();
        MergeCommonOption(AsccGlobalEnvManager::GetInstance().commonDeviceArgs);
        args_.incPaths.insert(args_.incPaths.begin(), inputArgs.incPaths.begin(), inputArgs.incPaths.end());
        args_.definitions.insert(args_.definitions.end(), inputArgs.definitions.begin(), inputArgs.definitions.end());
        args_.options.insert(args_.options.end(), inputArgs.options.begin(), inputArgs.options.end());
        args_.incFiles.insert(args_.incFiles.end(), inputArgs.incFiles.begin(), inputArgs.incFiles.end());
        args_.incPaths.emplace_back(Ascc::GetFilePath(args_.file));
        args_.incPaths.emplace_back(Ascc::GetFilePath(AsccArgumentManager::GetInstance().GetInputFile()));
        if (args_.customOption == std::string("aiv")) {
            args_.options.insert(args_.options.end(), aivOptions_.begin(), aivOptions_.end());
        }
        if (args_.customOption == std::string("aic")) {
            args_.options.insert(args_.options.end(), aicOptions_.begin(), aicOptions_.end());
        }

        // default of bisheng is O2. bisheng not support O1, thus use O2 instead
        const std::string optLevel = Ascc::AsccArgumentManager::GetInstance().GetOptimizeLevel();
        if (optLevel != "O1") {
            args_.options.emplace_back("-" + optLevel);
        } else {
            args_.options.emplace_back("-O2");
        }

        if (Ascc::AsccArgumentManager::GetInstance().DebugRequested()) {
            args_.options.emplace_back("-g");
        }
        if (Ascc::AsccArgumentManager::GetInstance().SanitizerRequested()) {
            args_.options.emplace_back("--cce-enable-sanitizer");
        }
    }
    const std::vector<std::string> aivOptions_ = {
        "-x cce",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-aicore-only",
        "--cce-auto-sync",
        "-mllvm -cce-aicore-stack-size=0x8000",
        "-mllvm -cce-aicore-function-stack-size=0x8000",
        "-mllvm -cce-aicore-record-overflow=true",
        "-mllvm -cce-aicore-addr-transform",
        "-mllvm -cce-aicore-dcci-insert-for-scalar=false",
        "-D__NPU_DEVICE__"
    };
    const std::vector<std::string> aicOptions_ = {
        "-x cce",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-aicore-only",
        "-mllvm -cce-aicore-stack-size=0x8000",
        "-mllvm -cce-aicore-function-stack-size=0x8000",
        "-mllvm -cce-aicore-record-overflow=true",
        "-mllvm -cce-aicore-addr-transform",
        "-mllvm -cce-aicore-dcci-insert-for-scalar=false",
        "-D__NPU_DEVICE__"
    };
};
}
#endif // __INCLUDE_ASCC_COMPILE_V220_H__