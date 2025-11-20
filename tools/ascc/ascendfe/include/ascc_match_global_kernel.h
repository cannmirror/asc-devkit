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
 * \file ascc_match_global_kernel.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_MATCH_GLOBAL_KERNEL_H__
#define __INCLUDE_ASCC_MATCH_GLOBAL_KERNEL_H__
#include <fstream>
#include <string>
#include <vector>
#include <stack>
#include <map>
#include <algorithm>
#include <sstream>

#include "ascc_utils.h"
#include "ascc_match_global_info.h"
#include "ascc_info_aicore_function.h"
#include "ascc_mangle.h"

namespace Ascc {
class AsccMatchGlobalKernel {
public:
    AsccMatchGlobalKernel() = default;
    ~AsccMatchGlobalKernel() = default;
    struct MatchPosition {
        size_t launchFrontStart;
        size_t launchFrontEnd;
        size_t launchRearStart;
        size_t launchRearEnd;
    };
    inline bool IsBracketBalanced(const std::string &str) const;
    bool IsMatchPair(const char &leftSymbol, const char &rightSymbol) const;
    bool IsFuncParamExist(const std::string &paramList) const;
    std::vector<MatchPosition> ExtractTripleAngleBracketsPositions(
        const std::string &input, const std::vector<size_t> &posMap) const;
    std::pair<std::string, std::vector<size_t>> RemoveSpacesAndMapPositions(const std::string &input) const;
    AsccStatus MatchAndReplaceGlobalKernel(std::vector<std::pair<uint32_t, std::string>>& callLines,
    std::vector<std::string> &ctx, size_t start, size_t cloumn) const;
    AsccStatus ReadFileToVector(const std::string &fileName, std::vector<std::string> &results) const;
    void MatchAndGenerateGlobalKernel();

private:
    std::map<std::string, std::vector<std::string>> files_;
    std::unordered_map<std::string, AsccInfoFunction::FunctionInfo> functions_;
};

inline bool AsccMatchGlobalKernel::IsBracketBalanced(const std::string &str) const
{
    std::stack<char> bracketStack;
    for (char c : str) {
        if (c == '<') {
            bracketStack.push(c);
        } else if (c == '>') {
            if (bracketStack.empty()) {
                return false;
            }
            bracketStack.pop();
        }
    }
    return bracketStack.empty();
}

inline std::vector<AsccMatchGlobalKernel::MatchPosition> AsccMatchGlobalKernel::ExtractTripleAngleBracketsPositions(
    const std::string &input, const std::vector<size_t> &posMap) const
{
    std::vector<AsccMatchGlobalKernel::MatchPosition> results;
    for (size_t j = 0; j < input.length(); ++j) {
        auto tripleAngleStart = input.find("<<<", j);
        if (tripleAngleStart == std::string::npos) {
            return results;
        }
        j = tripleAngleStart;
        auto tripleAngleStop = tripleAngleStart;
        do {
            tripleAngleStop = input.find(">>>", tripleAngleStop);
            if (tripleAngleStop == std::string::npos) {
                break;
            }
            const size_t tripleAngleLength = 0x3;
            std::string candidate = input.substr(
                tripleAngleStart + tripleAngleLength, tripleAngleStop - tripleAngleStart - tripleAngleLength);
            if (IsBracketBalanced(candidate) && (tripleAngleStart + tripleAngleLength) < posMap.size() &&
                (tripleAngleStop + tripleAngleLength - 1) < posMap.size()) {
                results.push_back({posMap[tripleAngleStart],
                    posMap[tripleAngleStart + tripleAngleLength],
                    posMap[tripleAngleStop - 1],
                    posMap[tripleAngleStop + tripleAngleLength - 1]});
            }
            tripleAngleStop++;
        } while (tripleAngleStop < input.length());
    }
    return results;
}

inline std::pair<std::string, std::vector<size_t>> AsccMatchGlobalKernel::RemoveSpacesAndMapPositions(
    const std::string &input) const
{
    std::string result;
    std::vector<size_t> posMap;
    for (size_t i = 0; i < input.length(); ++i) {
        if (input[i] != ' ') {
            result += input[i];
            posMap.push_back(i);
        }
    }
    return {result, posMap};
}

inline bool AsccMatchGlobalKernel::IsMatchPair(const char &leftSymbol, const char &rightSymbol) const
{
    if (leftSymbol == '(' && rightSymbol == ')') {return true;}
    if (leftSymbol == '<' && rightSymbol == '>') {return true;}
    if (leftSymbol == '{' && rightSymbol == '}') {return true;}
    if (leftSymbol == '[' && rightSymbol == ']') {return true;}
    return false;
}

inline bool AsccMatchGlobalKernel::IsFuncParamExist(const std::string &paramList) const
{
    const std::unordered_set<char> leftSide = {'(', '<', '{', '['};
    const std::unordered_set<char> rightSide = {')', '>', '}', ']'};
    std::stack<char> bracketStack;
    for (auto &c : paramList) {
        if (leftSide.find(c) != leftSide.end()) {
            bracketStack.push(c);
        } else if (rightSide.find(c) != rightSide.end()) {
            if (bracketStack.empty() || !IsMatchPair(bracketStack.top(), c)) {
                return false;
            }
            bracketStack.pop();
        } else if (c != ' ' && c != ';' && c != '\r') {
            return true;
        }
    }
    return !bracketStack.empty();
}

inline AsccStatus AsccMatchGlobalKernel::MatchAndReplaceGlobalKernel(
    std::vector<std::pair<uint32_t, std::string>> &callLines, std::vector<std::string> &ctx, size_t start,
    size_t cloumn) const
{
    ASC_LOG_ASC_DEBUG(PREPROCESS, "Start: [%zu], cloumn: [%zu].", start, cloumn);
    size_t kernelCallRow = start;
    std::string tripleLaunchGlobalKernel;
    while (kernelCallRow < ctx.size() && ctx[kernelCallRow].find(";") == std::string::npos) {
        tripleLaunchGlobalKernel.append(ctx[kernelCallRow] + "\n");
        ++kernelCallRow;
    }
    tripleLaunchGlobalKernel.append(ctx[kernelCallRow]);
    ASC_LOG_ASC_DEBUG(PREPROCESS, "tripleLaunchGlobalKernel: [%s], size: [%zu]",
        tripleLaunchGlobalKernel.c_str(), tripleLaunchGlobalKernel.size());

    const std::string tripleLess = "<<<";
    const std::string tripleGreat = ">>>(";
    const std::unordered_set<char> leftSide = {'(', '<', '{', '['};
    const std::unordered_set<char> rightSide = {')', '>', '}', ']'};
    size_t leftBracketUnmatchNum = 0;
    size_t tripleBegin = tripleLaunchGlobalKernel.find(tripleLess);
    if (tripleBegin == std::string::npos) {
        Ascc::HandleError("Kernel function call line can not find [<<<]!");
        return AsccStatus::FAILURE;
    }

    size_t tripleEnd = 0;
    for (tripleEnd = tripleBegin + tripleLess.size(); tripleEnd < tripleLaunchGlobalKernel.size(); ++tripleEnd) {
        const char& c = tripleLaunchGlobalKernel[tripleEnd];
        if (leftSide.find(c) != leftSide.end()) {
            ++leftBracketUnmatchNum;
            continue;
        }
        if (leftBracketUnmatchNum > 0 && rightSide.find(c) != rightSide.end()) {
            --leftBracketUnmatchNum;
            continue;
        }
        if (leftBracketUnmatchNum == 0 &&
            (tripleLaunchGlobalKernel.substr(tripleEnd, tripleGreat.size()) == tripleGreat)) {
            break;
        }
    }
    if (tripleEnd == tripleLaunchGlobalKernel.size()) {
        Ascc::HandleError("Kernel function call line can not find [>>>]!");
        return AsccStatus::FAILURE;
    }

    if (IsFuncParamExist(tripleLaunchGlobalKernel.substr(tripleEnd + tripleGreat.size() - 1))) {
        tripleLaunchGlobalKernel.replace(tripleEnd, tripleGreat.size(), ", ");
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Function parameters exist!");
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Function parameter list is: [%s], size: [%zu].",
            tripleLaunchGlobalKernel.substr(tripleEnd + tripleGreat.size() - 1).c_str(),
            tripleLaunchGlobalKernel.substr(tripleEnd + tripleGreat.size() - 1).size());
    } else {
        tripleLaunchGlobalKernel.replace(tripleEnd, tripleGreat.size(), "");
        ASC_LOG_ASC_DEBUG(PREPROCESS, "Function parameter does not exist!");
    }
    tripleLaunchGlobalKernel.replace(tripleBegin, tripleLess.size(), "(");

    std::istringstream ss(tripleLaunchGlobalKernel);
    std::string line;
    size_t replaceRow = start;
    while (getline(ss, line, '\n')) {
        ASC_LOG_ASC_INFO(PREPROCESS,
            "Exchange global call line [%zu] : [%s] -> [%s].",
            replaceRow + 1,
            ctx[replaceRow].c_str(),
            line.c_str());
        callLines.emplace_back(replaceRow, line);
        ctx[replaceRow] = line;
        ++replaceRow;
    }

    return AsccStatus::SUCCESS;
}

inline AsccStatus AsccMatchGlobalKernel::ReadFileToVector(
    const std::string &fileName, std::vector<std::string> &results) const
{
    if (!Ascc::IsPathLegal(fileName) || !Ascc::IsParentDirValid(fileName)) {
        ASC_LOG_ASC_ERROR(PREPROCESS, "file path [%s] does not exist!", fileName.c_str());
        return AsccStatus::FAILURE;
    }
    std::ifstream inputFile(fileName);
    if (!inputFile) {
        return AsccStatus::FAILURE;
    }
    std::string line;
    while (std::getline(inputFile, line)) {
        results.push_back(line);
    }
    inputFile.close();
    return AsccStatus::SUCCESS;
}

inline void AsccMatchGlobalKernel::MatchAndGenerateGlobalKernel()
{
    auto globalInfo = AsccMatchGlobalInfo::GetInstance().GetGlobalKernelCallExpr();
    std::unordered_map<std::string, std::string> &ascFixedMangleMap =
        Ascc::AsccMangle::GetInstance().GetOriginToFixedMangledNames();
    for (auto &info : globalInfo) {
        ASC_LOG_ASC_INFO(PREPROCESS, "Find <<<>>> call in file: [%s].", info.second.file.c_str());
        std::vector<std::string> fileCtx;
        std::vector<std::pair<uint32_t, std::string>> callLines;
        if (AsccMatchGlobalKernel::ReadFileToVector(info.second.file, fileCtx) != AsccStatus::SUCCESS) {
            ASC_LOG_ASC_WARN(PREPROCESS, "Can't read file to vector: [%s].", info.second.file.c_str());
            continue;
        }
        AsccStatus stat = AsccStatus::FAILURE;
        if (ascFixedMangleMap.count(info.second.manglingName) > 0) {
            ASC_LOG_ASC_INFO(PREPROCESS,
                "Match and replace template kernel call [%s].", info.second.funcName.c_str());
            stat = MatchAndReplaceGlobalKernel(callLines, fileCtx, info.second.line - 1, 0);
        } else {
            ASC_LOG_ASC_INFO(PREPROCESS,
                "Match and replace kernel call [%s].", info.second.funcName.c_str());
            stat = MatchAndReplaceGlobalKernel(callLines, fileCtx, info.second.line - 1, 0);
        }
        if (stat == AsccStatus::SUCCESS) {
            ASC_LOG_ASC_INFO(PREPROCESS, "Match file: [%s] <<<>>> success.", info.second.file.c_str());
            AsccInfoAicoreFunc &aicoreFunc = AsccInfoAicoreFunc::GetInstance();
            aicoreFunc.StoreKernelCallLineCode(info.second.file, callLines);
            files_[info.second.file] = fileCtx;
        }
    }
}
}  // namespace Ascc
#endif
