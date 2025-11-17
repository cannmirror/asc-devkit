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
 * \file asc_auto_identify_ktype.h
 * \brief auto identify kerneltype
 */
#ifndef __INCLUDE_INTERNAL_ASC_AUTO_IDENTIFY_KTYPE_H__
#define __INCLUDE_INTERNAL_ASC_AUTO_IDENTIFY_KTYPE_H__

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>

#include "asc_struct.h"
#include "asc_utils.h"
#include "asc_log.h"
#include "asc_info_manager.h"

namespace AscPlugin {

constexpr uint8_t CORE_TYPE_MIX = 0;
constexpr uint8_t CORE_TYPE_CUBE = 1;
constexpr uint8_t CORE_TYPE_VEC = 2;
constexpr uint8_t HEX_BASE = 16;

class AscCompileV220 {
public:
    AscCompileV220(std::string& coreTypeStr) : coreType_(coreTypeStr) {};
    int32_t Compile() {
        MergeCompileOpt();
        const auto& command = args_.GetCmd(AscPlugin::InfoManager::GetInstance().GetPathInfo().bishengPath);
        return AscPlugin::ExecuteCompile(command);
    }
    CompileArgs GetCompileArgs() const {
        return args_;
    }
    void MergeCompileOpt();

private:
    const std::vector<std::string> aivOptions_ = {
        "-x cce",
        "-std=c++17",
        "-DTILING_KEY_VAR=0",
        "-D__NO_KFC_SERVER__",
        "-D__NPU_DEVICE__",
        "-DASCENDC_DUMP=0",
        "-UASCENDC_ACC_DUMP",
        "-UASCENDC_TIME_STAMP_ON",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-aicore-only",
        "-O3",
        "-mllvm -cce-aicore-stack-size=0x8000",
        "-mllvm -cce-aicore-function-stack-size=0x8000",
        "-mllvm -cce-aicore-dcci-insert-for-scalar=false"
    };
    const std::vector<std::string> aicOptions_ = {
        "-x cce",
        "-std=c++17",
        "-DTILING_KEY_VAR=0",
        "-D__NPU_DEVICE__",
        "-DASCENDC_DUMP=0",
        "-UASCENDC_ACC_DUMP",
        "-UASCENDC_TIME_STAMP_ON",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-aicore-only",
        "-O3",
        "-mllvm -cce-aicore-stack-size=0x8000",
        "-mllvm -cce-aicore-function-stack-size=0x8000",
        "-mllvm -cce-aicore-dcci-insert-for-scalar=false"
    };
    CompileArgs args_;
    std::string coreType_;
};

std::unordered_map<std::string, std::string> GetTextMap(const std::string& command, const std::string& outputFile);
uint8_t GetV220CoreMode(const std::string& output);
KernelMetaType IdentifyKtypeImpl(const AscPlugin::KernelFuncInfo kernelKey,
    const std::vector<TemplateInstance>& templateInstances);

}  // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_AST_UTILS_H__