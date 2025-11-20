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
 * \file asc_auto_identify_ktype.cpp
 * \brief
 */

#include "asc_auto_identify_ktype.h"

namespace AscPlugin {

std::unordered_map<std::string, std::string> GetTextMap(const std::string& command, const std::string& outputFile) {
    std::unordered_map<std::string, std::string> result;
    auto [output, status] = AscPlugin::ExecuteCommand(command.c_str());
    if (status != 0) {
        ASC_LOGE("Failed to execute command: %s, status: %d", command.c_str(), status);
        return result;
    }
    if (output.empty()) {
        ASC_LOGE("No symbols found in %s", outputFile.c_str());
        return result;
    }
    auto lines = SplitLines(output);
    for (const auto& line : lines) {
        std::istringstream iss(line);
        std::string addr, flags, type, section, size, symbol;

        if (!(iss >> addr >> flags >> type >> section >> size)) {
            ASC_LOGE("Invalid line format: %s", line.c_str());
            continue;
        }
        std::getline(iss, symbol);
        symbol.erase(0, symbol.find_first_not_of(" \t"));
        if (section.substr(0, 5) == ".text") { // 5 is size of ".text"
            result[symbol] = section;
            ASC_LOGD("Section: %s, symbol: %s store successfully", section.c_str(), symbol.c_str());
        }
    }
    return result;
}

uint8_t GetV220CoreMode(const std::string& output) {
    uint8_t mode = 0;
    std::istringstream iss(output);
    std::string line;

    while (std::getline(iss, line)) {
        std::istringstream lineStream(line);
        std::vector<std::string> tokens;
        std::string token;

        while (lineStream >> token) {
            tokens.push_back(token);
        }
        if (tokens.size() < 5) continue; // 5 is Exceed the field limit

        for (size_t  i = 1; i <= 4 && i < tokens.size(); ++i) {  // 4 is max fileds
            const std::string& inst = tokens[i];
            if (inst.length() != 8) continue;     // 8 is expected inst length

            if (inst[6] == 'f' || inst[6] == 'c') {
                mode |= CORE_TYPE_CUBE;
            } else if (inst[6] == '8' || inst[6] == '9') {
                if (!(inst[4] == '4' && inst[5] == '0' && inst[6] == '8' && inst[7] == '0')) {
                    mode |= CORE_TYPE_VEC;
                }
            } else if (inst[6] == '6') {
                if (inst[7] == 'b' && (std::stoul(inst.substr(4, 1), nullptr, HEX_BASE) & 0x8) == 0x8) { // 4 is the index
                    mode |= CORE_TYPE_VEC;
                } else {
                    mode |= CORE_TYPE_CUBE;
                }
            } else if (inst[6] == '7') {
                if (inst[7] == '0' && (std::stoul(inst.substr(4, 1), nullptr, HEX_BASE) & 0x8) == 0x8) { // 4 is the index
                    mode |= CORE_TYPE_VEC;
                } else if ((std::stoul(inst.substr(0, 1), nullptr, HEX_BASE) & 0x7) == 0 &&
                          (std::stoul(inst.substr(1, 1), nullptr, HEX_BASE) & 0x8) == 0x8) {
                    mode |= CORE_TYPE_VEC;
                } else {
                    mode |= CORE_TYPE_CUBE;
                }
            }
        }
    }
    return mode;
}

KernelMetaType IdentifyKtypeImpl(const AscPlugin::KernelFuncInfo kernelKey,
    const std::vector<TemplateInstance>& templateInstances) {
    bool isTemplate = false;
    std::string instanceMangledNameKey;
    if (!templateInstances.empty()) {
        instanceMangledNameKey = templateInstances[0].instanceMangledName;  // No matter how many times it is instantiated, the kernel type remains the same.
        isTemplate = true;
    }
    ASC_LOGD("Start identify kernel type for instanceMangledNameKey %s", instanceMangledNameKey.c_str());

    uint8_t mode = 0;
    for (std::string coreTypeStr: {"aic", "aiv"}) {
        AscCompileV220 compileUnit(coreTypeStr);
        if (compileUnit.Compile() != ASC_SUCCESS) {
            ASC_LOGE("Execute compile failed, please check log, using default KERNEL_TYPE_AIV_ONLY");
            return KernelMetaType::KERNEL_TYPE_AIV_ONLY;
        }
        std::string outputFile = compileUnit.GetCompileArgs().outputPath;
        std::string objdumpPath= AscPlugin::InfoManager::GetInstance().GetPathInfo().objdumpPath;
        std::string command;
        std::string textName = ".text";
        if (isTemplate) {
            command = objdumpPath + " -t " + outputFile + " | grep F"; // the template kernel func is not in .text section
            ASC_LOGD("objdump command is %s", command.c_str());

            std::unordered_map<std::string, std::string> textMap = GetTextMap(command, outputFile);
            for (const auto& [key, value] : textMap) {
                ASC_LOGD("textMap has key %s, value %s", key.c_str(), value.c_str());
            }

            if (textMap.find(instanceMangledNameKey) != textMap.end()) {
                textName = textMap[instanceMangledNameKey];
                ASC_LOGD("During kernel type auto identification, found kernel mangled name: %s with text: %s",
                    instanceMangledNameKey.c_str(), textName.c_str());
            } else {
                ASC_LOGE("During kernel type auto identification, received unknown kernel mangled name: %s, "
                    "please check log", instanceMangledNameKey.c_str());
                return KernelMetaType::KERNEL_TYPE_AIV_ONLY;
            }
        }
        command = objdumpPath + " -s -j " + textName + " " + outputFile;
        auto [output, status] = AscPlugin::ExecuteCommand(command.c_str());
        if (status != 0) {
            ASC_LOGE("Failed to execute command: %s, status: %d", command.c_str(), status);
            return KernelMetaType::KERNEL_TYPE_AIV_ONLY;
        }
        if (output.empty()) {
            ASC_LOGE("No symbols found in %s", outputFile.c_str());
            return KernelMetaType::KERNEL_TYPE_AIV_ONLY;
        }
        mode |= GetV220CoreMode(output);
        if (!AscPlugin::InfoManager::GetInstance().SaveTempRequested()) {
           AscPlugin::ExecuteCommand(("rm -rf " + outputFile).c_str());
        }
    }
    if (mode == 0 || mode == CORE_TYPE_VEC) {
        ASC_LOGD("kernel func mangled name: %s at %s:%u, col:%u, kernel type is aiv_only by automatically identified",
            kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(), kernelKey.lineNum, kernelKey.colNum);
        return KernelMetaType::KERNEL_TYPE_AIV_ONLY;
    } else if (mode == CORE_TYPE_CUBE) {
        ASC_LOGD("kernel func mangled name: %s at %s:%u, col:%u, kernel type is aic_only by automatically identified",
            kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(), kernelKey.lineNum, kernelKey.colNum);
        return KernelMetaType::KERNEL_TYPE_AIC_ONLY;
    } else if (mode == 3) {   // 3 means 1 | 2
        ASC_LOGD("kernel func mangled name: %s at %s:%u, col:%u, kernel type is mix_aic_1_2 by automatically identified",
            kernelKey.mangledName.c_str(), kernelKey.fileName.c_str(), kernelKey.lineNum, kernelKey.colNum);
        return KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2;
    }
    return KernelMetaType::KERNEL_TYPE_AIV_ONLY;
}

void AscPlugin::AscCompileV220::MergeCompileOpt() {
    args_.customOption = coreType_;
    const CompileArgs &inputArgs = AscPlugin::InfoManager::GetInstance().GetCompileArgs();
    args_.file = AscPlugin::InfoManager::GetInstance().GetSourceFile();
    const std::string intputFileDir = GetFilePath(args_.file);
    if (AscPlugin::InfoManager::GetInstance().SaveTempRequested()) {
        args_.outputPath = AscPlugin::InfoManager::GetInstance().GetTempPath() + "/identify_" + coreType_ + ".o";
    } else {
        std::string timestamp = GenerateTimestamp();
        std::string pid = std::to_string(getpid());
        std::string tid = std::to_string(syscall(SYS_gettid));
        args_.outputPath = "/tmp/AscPlugin_" + timestamp + "_" + pid + "_" + tid + "_identify_" + coreType_ + ".o";
    }

    PathInfo pathInfo = AscPlugin::InfoManager::GetInstance().GetPathInfo();
    args_.includePaths = {
        "-I" + intputFileDir,
        "-I" + pathInfo.cannIncludePath,
        "-I" + pathInfo.hostApiPath,
        "-I" + pathInfo.highLevelApiPath,
        "-I" + pathInfo.tikcfwPath,
        "-I" + pathInfo.tikcfwLibPath,
        "-I" + pathInfo.tikcfwLibMatmulPath,
        "-I" + pathInfo.tikcfwImplPath,
        "-I" + pathInfo.tikcfwInterfacePath,
        "-I" + pathInfo.ascendClangIncludePath
    };
    args_.definitions.insert(args_.definitions.end(), inputArgs.definitions.begin(), inputArgs.definitions.end());
    std::vector<std::string> removeOpts = {"-DASCENDC_DUMP", "-DASCENDC_DUMP=1"};
    args_.RemoveOptions(removeOpts);
    args_.includePaths.insert(args_.includePaths.end(), inputArgs.includePaths.begin(), inputArgs.includePaths.end());
    args_.options.insert(args_.options.end(), inputArgs.options.begin(), inputArgs.options.end());
    args_.includeFiles.insert(args_.includeFiles.end(), inputArgs.includeFiles.begin(), inputArgs.includeFiles.end());
    args_.linkFiles.insert(args_.linkFiles.end(), inputArgs.linkFiles.begin(), inputArgs.linkFiles.end());
    args_.linkPaths.insert(args_.linkPaths.end(), inputArgs.linkPaths.begin(), inputArgs.linkPaths.end());
    if(AscPlugin::InfoManager::GetInstance().IsAutoSyncOn()){
        args_.options.emplace_back("--cce-auto-sync");
    }
    if (args_.customOption == std::string("aiv")) {
        args_.options.insert(args_.options.end(), aivOptions_.begin(), aivOptions_.end());
    }
    if (args_.customOption == std::string("aic")) {
        args_.options.insert(args_.options.end(), aicOptions_.begin(), aicOptions_.end());
    }
}

} // namesapce AscPlugin