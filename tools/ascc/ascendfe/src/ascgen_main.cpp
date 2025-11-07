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
 * \file ascgen_main.cpp
 * \brief
 */

#include <cstdlib>
#include <cstring>
#include <sys/types.h>
#include "ascc_host_stub.h"
#include "ascc_log.h"
#include "ascc_argument_manager.h"
#include "ascc_device_stub.h"
#include "ascc_link.h"
#include "ascc_tmp_file_manager.h"
#include "ascc_host_compile.h"
#include "ascc_types.h"
#include "ascc_ast_analyzer.h"
#include "ascc_ast_device_analyzer.h"
#include "ascc_utils.h"

#if !defined(UT_TEST) && !defined(ST_TEST)
int main(int argc, char *argv[])
{
#else
int EntryAscGenMain(int argc, char *argv[])
{
#endif
    std::vector<std::string> args(argv + 1, argv + argc);

    // environment variables set up
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    ASCC_CHECK_MAIN((envVar.initSuccess),
        {Ascc::HandleError("Environment Variables init failed, please [source setenv.bash].");});

    // Parse arguments
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    ASCC_CHECK_MAIN((argManager.ArgumentParse(args) == Ascc::AsccStatus::SUCCESS),
        { Ascc::HandleError("ASCC parser failed in parsing arguments."); });
    if (argManager.ManualRequested()) {  // when encountered --help in arguments
        argManager.PrintManual();
        return Ascc::ASCC_SUCCESS;
    }

    const std::string& file = argManager.GetInputFile();
    std::string fileName = Ascc::GetFileName(file);

    // create temporary directories to store preprocess stub files
    const auto& tmpFileManager = Ascc::AsccTmpFileManager::GetInstance();
    ASCC_CHECK_MAIN((tmpFileManager.Init(fileName) == Ascc::AsccStatus::SUCCESS),
        {Ascc::HandleError("AsccTmpFileManager init failed. Please check log.");});

    // Start ClangTooling process
    Ascc::AsccAstAnalyzer analyzer(file);
    ASCC_CHECK_MAIN((analyzer.Process() == Ascc::AsccStatus::SUCCESS),
        {Ascc::HandleError("AsccAstAnalyzer run failed. Please check log.");});
    Ascc::AsccAstDeviceAnalyzer deviceAnalyzer(file);
    ASCC_CHECK_MAIN((deviceAnalyzer.Process() == Ascc::AsccStatus::SUCCESS),
        {Ascc::HandleError("AsccAstAnalyzer run failed. Please check log.");});

    Ascc::AsccDeviceStub deviceStubGenerator(argManager.GetNpuArch(), envVar.asccTmpAutoGenPath);
    // Generate kernel json
    ASCC_CHECK_MAIN((deviceStubGenerator.GenerateJsonFiles() == Ascc::AsccStatus::SUCCESS),
        { Ascc::HandleError("Generate json files failed. Please check log."); });
    // Generate device stub files
    ASCC_CHECK_MAIN((deviceStubGenerator.Process() == Ascc::AsccStatus::SUCCESS),
        { Ascc::HandleError("Device Compile or Code generate failed. Please check log."); });
    // return in advance if task type means only need to generate device stub files, no need for following processes
    Ascc::PreTaskType preTaskType = Ascc::AsccArgumentManager::GetInstance().GetPreTaskType();
    if (preTaskType == Ascc::PreTaskType::DEVICE_AIV || preTaskType == Ascc::PreTaskType::DEVICE_AIC) {
        if (!envVar.needSaveTmpFile) {
            Ascc::ExecuteCommand((std::string("rm -rf ") + envVar.asccTmpPath).c_str());
        }
        return Ascc::ASCC_SUCCESS;
    }

    Ascc::AsccHostStub hostStubGenerator(preTaskType);
    ASCC_CHECK_MAIN((hostStubGenerator.GenHostStubFile() == Ascc::AsccStatus::SUCCESS),
        { Ascc::HandleError("Host Compile or Code generate failed. Please check log."); });
    Ascc::AsccLink asccLink(envVar.ascendLinker, Ascc::GetFileName(argManager.GetInputFile()), Ascc::BuildType::DEBUG);
    ASCC_CHECK_MAIN((asccLink.AscendLink(hostStubGenerator) == Ascc::AsccStatus::SUCCESS),
        { Ascc::HandleError("Host Compile or Link process failed. Please check log."); });

    if ((!envVar.needSaveTmpFile) && Ascc::PathCheck(envVar.asccTmpPath.c_str(), true) != Ascc::PathStatus::NOT_EXIST) {
        Ascc::ExecuteCommand((std::string("rm -rf ") + envVar.asccTmpPath).c_str());
    }

    return Ascc::ASCC_SUCCESS;
}
