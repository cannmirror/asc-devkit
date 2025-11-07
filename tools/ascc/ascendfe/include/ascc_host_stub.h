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
 * \file ascc_host_stub.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_HOST_STUB_H__
#define __INCLUDE_ASCC_HOST_STUB_H__

#include <string>
#include <fstream>

#include "ascc_info_function.h"
#include "ascc_info_storage.h"

namespace Ascc {
class AsccHostStub {
public:
    AsccHostStub(PreTaskType preTaskType);
    ~AsccHostStub() = default;
    std::string GetHostStubFilePath() const;
    AsccStatus GenHostStubFile();
    AsccStatus UpdateHostStubByDevice();

private:
    // tool
    bool IsKernelFuncFound() const;
    void UpdateKernelTypeStatus();
    void UpdateDumpStatus();

    // file level
    void GenHostStubHeadCode();
    void GenAscendKernelStruct();
    void GenerateCodeForMode(KernelMode mode, std::vector<std::string> &ascendKernelStruct);
    void GenHostStubUsedIntfDecl();
    void GenUnregisterCode();
    void GenKernelHandleUnregisterCode(const std::string& handleSuffix);
    void GenBinaryRegisterCode();
    void GenKernelHandleRegisterCode(const std::string& suffix);
    void GenFunctionRegisterCode();
    void GenManglingRegisterBody(const std::shared_ptr<AsccInfoFunction> &funcsInfo);
    void GenSetExceptionDumpCode();

    // function level
    void GenLaunchProfilingBody(const std::shared_ptr<AsccInfoFunction> &functions);
    void GenLaunchProfilingCode();
    void GenKernelHandleCheck(const std::string& suffix);
    void GenStubFuncImpl(const AsccInfoFunction::FunctionInfo& info);
    std::string GenStubFuncDecl(const AsccInfoFunction::FunctionInfo& info, bool hasNameSpace = false) const;
    void GenManglingRegisterCode(const std::string &manglingName, const KernelMode &curKernelMode);
    std::string ManglingNameJudgeCode(const AsccInfoFunction::FunctionInfo& info);

private:
    std::string hostStubFilePath_;
    std::ofstream hostStubSource_;
    std::ofstream kernelCallStubFile_;
    size_t isTypeCounter_ = 0;
    bool hasMix_ = false;
    bool hasAic_ = false;
    bool hasAiv_ = false;
    bool hasPrintf_ = false;
    bool hasAssert_ = false;
    uint32_t typeNums_ = 0;
};
} // Ascc

#endif // __INCLUDE_ASCC_HOST_STUB_H__