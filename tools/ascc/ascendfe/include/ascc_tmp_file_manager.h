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
 * \file ascc_tmp_file_manager.h
 * \brief
 */

#ifndef __INCLUDE_ASCC_TMP_FILE_MANAGER_H__
#define __INCLUDE_ASCC_TMP_FILE_MANAGER_H__
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fstream>
#include <ctime>
#include <unistd.h>
#include <iomanip>

#include "ascc_utils.h"
#include "ascc_log.h"
#include "ascc_compile_base.h"
#include "ascc_ast_kernel_stub.h"

namespace Ascc {

class AsccTmpFileManager {
public:
    static AsccTmpFileManager &GetInstance()
    {
        static AsccTmpFileManager instance;
        return instance;
    }

    // inputFilePath: xx.cpp
    AsccStatus Init(const std::string& inputFileName) const
    {
        Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();

        std::string inputFilePrefix = "main";
        size_t lastdot = inputFileName.find_last_of(".");
        if (lastdot != std::string::npos) {
            inputFilePrefix = inputFileName.substr(0, lastdot);  // get [xx]
        } else {
            inputFilePrefix = inputFileName;
        }

        std::stringstream ss;
        ss << getpid() << "_" << syscall(SYS_gettid);   // make tmp directories unique
        envVar.asccTmpPath += std::string("/") + inputFilePrefix + std::string("_") + GenerateTimestamp() +
            std::string("_") + ss.str();

        ASC_LOG_ASC_DEBUG(INIT, "ASCC TmpPath: [%s].", envVar.asccTmpPath.c_str());
        ASCC_CHECK((CreateDirectory(envVar.asccTmpPath) == AsccStatus::SUCCESS), {});
        envVar.asccTmpIncludePath = envVar.asccTmpPath + "/include";        // /tmp/ascc/时间戳/include 目录
        ASCC_CHECK((CreateDirectory(envVar.asccTmpIncludePath) == AsccStatus::SUCCESS), {});
        envVar.asccTmpAutoGenPath = envVar.asccTmpPath + "/auto_gen";        // /tmp/ascc/时间戳/
        envVar.asccTmpHostGenPath = envVar.asccTmpPath + "/auto_gen/host_files";        // /tmp/ascc/时间戳/host_files 目录
        ASCC_CHECK((CreateDirectory(envVar.asccTmpHostGenPath) == AsccStatus::SUCCESS), {});
        envVar.asccTmpDependPath = envVar.asccTmpPath + "/dependence";                 // /tmp/ascc/时间戳/dependence 目录
        ASCC_CHECK((CreateDirectory(envVar.asccTmpDependPath) == AsccStatus::SUCCESS), {});
        envVar.asccMergeObjPath = envVar.asccTmpPath + "/link_files/merge_obj";       // /tmp/ascc/时间戳/link_files/merge_obj 目录
        ASCC_CHECK((CreateDirectory(envVar.asccMergeObjPath) == AsccStatus::SUCCESS), {});
        envVar.asccMergeObjFinalPath = envVar.asccTmpPath + "/link_files/merge_obj_final";   // /tmp/ascc/时间戳/link_files/merge_obj_final 目录
        ASCC_CHECK((CreateDirectory(envVar.asccMergeObjFinalPath) == AsccStatus::SUCCESS), {});
        envVar.asccCompileLogPath = envVar.asccTmpPath + "/compile_log";                // /tmp/ascc/时间戳/compile_log/目录 存放compile,link等命令
        ASCC_CHECK((CreateDirectory(envVar.asccCompileLogPath) == AsccStatus::SUCCESS), {});

        ASCC_CHECK((GenerateStubFiles(envVar.asccTmpIncludePath) == AsccStatus::SUCCESS), {});

        ASC_LOG_ASC_DEBUG(INIT, "Kernel operator file parsed successfully.");

        if (envVar.needSaveTmpFile) {
            std::cout << "Temporary file storage path is set up as: " << envVar.asccTmpPath << std::endl;
        }
        return AsccStatus::SUCCESS;
    }

    AsccTmpFileManager(const AsccTmpFileManager &) = delete;
    AsccTmpFileManager &operator=(const AsccTmpFileManager &) = delete;

private:
    AsccTmpFileManager() {}
    ~AsccTmpFileManager() = default;

    // timestamp used for creating unique tmp directories
    static std::string GenerateTimestamp()
    {
        std::time_t now = std::time(nullptr);
        struct tm tmBuf;

        // 使用线程安全版本
        if (!localtime_r(&now, &tmBuf)) {
            ASC_LOG_ASC_ERROR(INIT, "Failed to get local time in GenerateTimestamp!");
        }

        char timestamp[20];
        if (strftime(timestamp, sizeof(timestamp), "%Y%m%d%H%M%S", &tmBuf) == 0) {
            ASC_LOG_ASC_ERROR(INIT, "Timestamp formatting failed in GenerateTimestamp!");
        }
        return std::string(timestamp);
    }

    // execute mkdir to create directories
    static AsccStatus ExecMkdir(struct stat& fileStatus, const char* dirPath) {
        if (stat(dirPath, &fileStatus) != 0) {
            // need execute permission for directories
            int mkdirRes = mkdir(dirPath, S_IRWXU | S_IRWXG );
            if (mkdirRes != 0 ) {
                if (errno == EEXIST) {
                    ASC_LOG_ASC_DEBUG(INIT, "Directory already exists: [%s].", dirPath);
                    return AsccStatus::SUCCESS;
                } else {
                    ASC_LOG_ASC_ERROR(INIT, "Failed to create directory: [%s].", dirPath);
                    return AsccStatus::FAILURE;
                }
            }
            ASC_LOG_ASC_DEBUG(INIT, "Create directory successfully: [%s].", dirPath);
        }
        return AsccStatus::SUCCESS;
    }

    // Example dir:  /tmp/ascc/tmp_log
    static AsccStatus CreateDirectory(const std::string& dirPath)
    {
        ASCC_CHECK((!dirPath.empty()), {ASC_LOG_ASC_ERROR(INIT, "dirPath is empty in CreateDirectory!");});

        size_t pos = 0;
        std::string currentDir = dirPath;
        std::string currentFullDir;

        // 处理绝对路径
        if (dirPath[0] == '/') {
            currentFullDir = "/";
            currentDir = dirPath.substr(1);  // 截取除根目录后的部分
            pos = 0;
        }

        struct stat fileStatus;
        // for loop to search "/": in Example dir above, deal with dir [tmp], dir [ascc]wenjia
        while ((pos = currentDir.find('/', pos)) != std::string::npos) {
            std::string subDir = currentDir.substr(0, pos);
            if (subDir.empty()) {
                pos++;
                continue;
            }

            currentFullDir += subDir + "/";
            ASCC_CHECK((ExecMkdir(fileStatus, currentFullDir.c_str()) == AsccStatus::SUCCESS), {});

            // 更新currentDir为剩余部分，并重置pos
            currentDir = currentDir.substr(pos + 1);
            pos = 0;
        }

        // in Example dir above, deal with dir [tmp_log]
        currentFullDir += currentDir;
        ASCC_CHECK((ExecMkdir(fileStatus, currentFullDir.c_str()) == AsccStatus::SUCCESS), {});

        return AsccStatus::SUCCESS;
    }

    // generate stub version of kernel_operator_stub.h
    static AsccStatus GenerateStubFiles(const std::string& tmpIncPath) {
        // stub version of kernel_operator_stub.h
        const std::string clangToolingStub = GetAstKernelStub();
        std::string clangToolingStubPathCheck = CheckAndGetFullPath(tmpIncPath);
        if (clangToolingStubPathCheck.empty()) {
            return AsccStatus::FAILURE;
        }
        std::string clangToolingStubPath = clangToolingStubPathCheck + "/kernel_operator_stub.h";
        if (!Ascc::IsPathLegal(clangToolingStubPath) || !Ascc::IsParentDirValid(clangToolingStubPath)) {
            ASC_LOG_ASC_ERROR(PREPROCESS, "clangToolingStubPath [%s] does not exist!", clangToolingStubPath.c_str());
            return AsccStatus::FAILURE;
        }
        std::ofstream outfileTooling(clangToolingStubPath);
        ASCC_CHECK(outfileTooling.is_open(),
            {Ascc::HandleError("Failed to open stub kernel_operator_stub.h for parsing. Please check log!");});
        outfileTooling << clangToolingStub;
        outfileTooling.close();
        return AsccStatus::SUCCESS;
    }
};
}  // namespace Ascc
#endif