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
 * \file ascc_main.cpp
 * \brief
 */
#include "ascc_args.h"
#include "ascc_utils.h"
#include "ascc_task_manager.h"

#if !defined(UT_TEST) && !defined(ST_TEST)
int main(int argc, char *argv[])
#else
int EntryAsccMain(int argc, char *argv[])
#endif
{
    // analyze and collects arguments
    if (!Ascc::ParseCommandLineOptions(argc, argv)) {
        Ascc::HandleErrorAndCheckLog("Parse command line options failed.");
        return Ascc::ASCC_FAILURE;
    }

    // collect executable path info
    Ascc::ArgInfo argInfo;
    if (Ascc::InitArgInfo(argInfo) != Ascc::AsccStatus::SUCCESS) {
        Ascc::HandleErrorAndCheckLog("Get argument values failed.");
        return Ascc::ASCC_FAILURE;
    }

    // get input files
    std::vector<std::string> files;
    Ascc::GetPositionalInputFiles(files);
    if (Ascc::CheckFiles(files, argInfo) != Ascc::AsccStatus::SUCCESS) {
        return Ascc::ASCC_FAILURE;
    }

    Ascc::PathInfo pathInfo;
    if (Ascc::InitPathInfo(pathInfo, argInfo) != Ascc::AsccStatus::SUCCESS) {
        Ascc::HandleErrorAndCheckLog("Init paths for executable failed.");
        return Ascc::ASCC_FAILURE;
    }
    if (Ascc::CreateTmpDirectory(files, pathInfo) != Ascc::AsccStatus::SUCCESS) {
        Ascc::HandleErrorAndCheckLog("Create directory for temporary files storage failed.");
        return Ascc::ASCC_FAILURE;
    }

    // execute tasks for input files
    if (Ascc::ProcessFiles(files, pathInfo, argInfo) != Ascc::AsccStatus::SUCCESS) {
        Ascc::HandleErrorAndCheckLog("Process files failed.");
        return Ascc::ASCC_FAILURE;
    }

    if (!argInfo.needSaveTemps) {
        Ascc::ExecuteCommand(("rm -rf " + pathInfo.tmpFilePath).c_str());
    }
    return Ascc::ASCC_SUCCESS;
}
