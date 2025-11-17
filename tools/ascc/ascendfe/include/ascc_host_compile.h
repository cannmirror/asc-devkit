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
 * \file ascc_host_compile.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_HOST_COMPILE_H__
#define __INCLUDE_ASCC_HOST_COMPILE_H__
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <utility>
#include <cstddef>
#include <iostream>
#include <cstring>

#include "ascc_log.h"
#include "ascc_info_aicore_function.h"
#include "ascc_host_stub.h"
#include "ascc_tmp_file_manager.h"
#include "ascc_compile_factory.h"
#include "ascc_argument_manager.h"
#include "ascc_utils.h"
#include "ascc_match_global_info.h"

namespace Ascc {

class AsccHostCompile {
public:
    AsccHostCompile() = default;
    ~AsccHostCompile() = default;

    void HostCompile() const
    {
        const auto &env = Ascc::AsccGlobalEnvManager::GetInstance();
        Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
        std::string fullDstAddr = Ascc::CheckAndGetFullPath(env.asccTmpHostGenPath);
        if (fullDstAddr.empty()) {
            Ascc::HandleError("Tmp file path is invalid! Please check log!");
        }
        if (argManager.GetPreTaskType() != PreTaskType::NONE) {
            fullDstAddr = argManager.GetModulePath();
        }
        std::string unitPath = argManager.GetInputFile();
        size_t lastSlash = unitPath.find_last_of('/');
        std::string unitName = unitPath.substr(lastSlash + 1);
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Compile unit file: [%s].", unitName.c_str());
        Ascc::CompileArgs args;
        args.incPaths.emplace_back(env.asccTmpAutoGenPath);
        args.incPaths.emplace_back(fullDstAddr);
        args.incPaths.emplace_back(env.ascendCannIncludePath);
        args.definitions.emplace_back("TILING_KEY_VAR=0");
        if (!AsccMatchGlobalInfo::GetInstance().HasKernelCall()) {
            args.outputPath = env.asccTmpAutoGenPath + "/" + unitName + ".o";
            args.file = argManager.GetInputFile();
            AsccGlobalEnvManager::GetInstance().hostCompileFile = args.file;
            Ascc::CompileRegister<Ascc::AsccCompileHost>(args);
            return;
        }
        AsccInfoAicoreFunc &instance = AsccInfoAicoreFunc::GetInstance();
        const auto &allKeys = instance.GetAllKeys();
        // isCppNotRegist = false -> use .cpp in tmp auto_gen to compile;
        //                = true  -> use .cpp from user
        bool isCppNotRegist = true;
        std::vector<std::string> includeFiles;
        std::vector<std::string> includePaths;
        for (const auto &key : allKeys) {
            ASC_LOG_ASC_DEBUG(PREPROCESS, "Extract host code process key: [%s].", key.c_str());
            std::vector<std::pair<uint32_t, uint32_t>> aicoreRanges;
            instance.GetKernelDefScope(key, aicoreRanges);

            std::vector<std::pair<uint32_t, std::string>> kernelCallLines;
            instance.GetKernelCallLineCode(key, kernelCallLines);

            std::vector<std::pair<uint32_t, std::string>> kernelDefLines;
            instance.GetkernelDefLineCode(key, kernelDefLines);

            size_t innerLastSlash = key.find_last_of('/');
            std::string newFileName = key.substr(innerLastSlash + 1);
            std::string dstAddress = fullDstAddr + "/" + newFileName;
            std::string replacedFilePath = key.substr(0, innerLastSlash);
            if (Ascc::CheckAndGetFullPath(replacedFilePath).empty()) {
                return;
            }
            includePaths.emplace_back(replacedFilePath);
            ShieldDeviceCode(key, dstAddress, aicoreRanges, kernelCallLines, kernelDefLines);
            if (dstAddress.find(unitName) != std::string::npos) {
                isCppNotRegist = false;
            } else {
                includeFiles.emplace_back(dstAddress);
            }
        }
        if (argManager.GetPreTaskType() == Ascc::PreTaskType::HOST) {
            return; // only for precompile, do not compile
        }
        args.incFiles.insert(args.incFiles.end(), includeFiles.begin(), includeFiles.end());
        args.incPaths.insert(args.incPaths.end(), includePaths.begin(), includePaths.end());
        if (isCppNotRegist) {
            args.incFiles.emplace_back(argManager.GetInputFile());
        } else {
            args.incFiles.emplace_back(fullDstAddr + "/" + unitName);
        }
        args.outputPath = env.asccTmpAutoGenPath + "/" + unitName + ".o";
        args.file = AsccGlobalEnvManager::GetInstance().kernelCallStubFile;
        AsccGlobalEnvManager::GetInstance().hostCompileFile = args.file;
        Ascc::CompileRegister<Ascc::AsccCompileHost>(args);
    }

private:
    void ProcessKernelCallLines(
        const std::vector<std::pair<uint32_t, std::string>> &kernelCallLines, std::vector<std::string> &lines) const
    {
        for (const auto &pair : kernelCallLines) {
            const auto& row = pair.first;
            const auto& line = pair.second;
            lines[row] = line;
        }
    }

    void ProcessKernelFuncLines(
        const std::vector<std::pair<uint32_t, uint32_t>> &ranges, std::vector<std::string> &lines) const
    {
        // 检查并收集有效范围，同时转换索引
        std::vector<std::pair<uint32_t, uint32_t>> validRanges;
        for (size_t i = 0; i < ranges.size(); ++i) {
            uint32_t start = ranges[i].first;
            uint32_t end = ranges[i].second;
            if (start < 1 || end > lines.size() || start > end) {
                ASC_LOG_ASC_WARN(PREPROCESS, "Invalid line range specified for range: [%zu]!", (i + 1));
                continue;
            }
            validRanges.emplace_back(start - 1, end - 1);  // 转换0-based索引
        }

        // 仅处理有效范围
        const uint32_t commentSymbolLength = 2;
        for (const auto &range : validRanges) {
            for (uint32_t lineIndex = range.first; lineIndex <= range.second; ++lineIndex) {
                if (!lines[lineIndex].empty() && lines[lineIndex].substr(0, commentSymbolLength) != "//") {
                    lines[lineIndex] = "//" + lines[lineIndex];
                }
            }
        }
    }

    void ShieldDeviceCode(const std::string &inputFile, const std::string &outputFile,
        const std::vector<std::pair<uint32_t, uint32_t>> &ranges,
        const std::vector<std::pair<uint32_t, std::string>> kernelCallLines,
        const std::vector<std::pair<uint32_t, std::string>> kernelDefLines) const
    {
        if (!Ascc::IsPathLegal(inputFile) || !Ascc::IsParentDirValid(inputFile)) {
            ASC_LOG_ASC_ERROR(PREPROCESS, "inputFile path [%s] does not exist!", inputFile.c_str());
            return;
        }
        std::ifstream infile(inputFile);
        if (!infile.is_open()) {
            ASC_LOG_ASC_ERROR(PREPROCESS, "Failed to open input file: [%s]!", inputFile.c_str());
            return;
        }

        std::vector<std::string> lines;
        std::string line;
        while (std::getline(infile, line)) {
            lines.push_back(line);
        }
        infile.close();

        if (!kernelCallLines.empty()) {
            ProcessKernelCallLines(kernelCallLines, lines);
        }

        if (!ranges.empty()) {
            ProcessKernelFuncLines(ranges, lines);
        }

        if (!kernelDefLines.empty()) {
            for (const auto &pair : kernelDefLines) {
                uint32_t row = pair.first;
                const std::string &kernelDefinition = pair.second;
                if (lines.size() >= (row + 1)) {
                    lines[row] = kernelDefinition;
                }
            }
        }

        if (!Ascc::IsPathLegal(outputFile) || !Ascc::IsParentDirValid(outputFile)) {
            ASC_LOG_ASC_ERROR(PREPROCESS, "outputFile path [%s] does not exist!", outputFile.c_str());
            return;
        }
        std::ofstream outfile(outputFile);
        if (!outfile.is_open()) {
            ASC_LOG_ASC_ERROR(PREPROCESS, "Failed to open output file: [%s]!", outputFile.c_str());
            return;
        }

        for (const auto &l : lines) {
            outfile << l << "\n";
        }
        outfile.close();

        ASC_LOG_ASC_DEBUG(PREPROCESS, "Successfully processed and wrote to [%s].", outputFile.c_str());
    }
};
}  // namespace Ascc
#endif