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
 * \file ascc_link.cpp
 * \brief
 */
#ifndef __INCLUDE_ASCC_LINK_H__
#define __INCLUDE_ASCC_LINK_H__
#include <fstream>
#include <iostream>

#include "ascc_log.h"
#include "ascc_host_stub.h"
#include "ascc_host_compile.h"
#include "ascc_mangle.h"
#include "ascc_match_global_kernel.h"
#include "ascc_utils.h"

namespace Ascc {
class AsccLink {
public:
    AsccLink(std::string linkerPath, std::string inputFileName, BuildType buildType);
    ~AsccLink() = default;
    AsccStatus AscendLink(AsccHostStub &hostStubGenerator);

private:
    const std::string SanitizerLinkProcess(const std::string& cannPath) const;
    AsccStatus LinkProcessForDeviceO(Ascc::AsccGlobalEnvManager& envVar, const std::string& coreType,
        const std::string& linkerPathBase) const;
    AsccStatus LinkProcessForDeviceOWithR(const std::string &linkCmdBase, const std::string &coreType,
        const std::string &outputPath, Ascc::KernelMode expectMode) const;
    AsccStatus PackProcessForDeviceO(Ascc::AsccGlobalEnvManager& envVar, const std::string& coreType,
        const std::string& packPrefix, const std::string& elfIn, const uint8_t coreTypeValue) const;
    AsccStatus MergeObjFinal(bool mergeFlag);
    AsccStatus MergeObj(const std::string &outputPath, bool mergeFlag);
    AsccStatus AsccPackKernel(
        const std::string &packTool, const std::string &elfIn, const std::string &addDirPreLink) const;
    void GetLibFileDependency();
    void GetLibPathDependency();
    std::string CommandForFileExec(
        Ascc::AsccGlobalEnvManager &envVar, std::string &commonPart, std::string &libSuffix) const;
    std::string CommandForFileSo(
        Ascc::AsccGlobalEnvManager &envVar, std::string &commonPart, std::string &libSuffix) const;
    std::string CommandForFileO(Ascc::AsccGlobalEnvManager &envVar, std::string &commonPart) const;
    const std::string GetOutputFileName();
    AsccStatus GenResFile(Ascc::AsccGlobalEnvManager& envVar);
    bool IsKernelFuncFound() const;

private:
    std::string linkerPath_;
    std::string inputFileName_;
    BuildType buildType_;
    std::vector<std::string> libFileDependency_{};
    std::vector<std::string> libPathDependency_{};
};
} // namespace Ascc
#endif