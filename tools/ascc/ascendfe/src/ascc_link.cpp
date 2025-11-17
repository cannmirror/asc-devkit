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

#include "ascc_link.h"
#include "ascc_host_stub.h"
#include "ascc_host_compile.h"

namespace Ascc {
AsccLink::AsccLink(std::string linkerPath, std::string inputFileName, BuildType buildType)
    : linkerPath_(linkerPath), inputFileName_(inputFileName), buildType_(buildType) {}

const std::string AsccLink::SanitizerLinkProcess(const std::string& cannPath) const
{
    Ascc::AsccArgumentManager& argManager = Ascc::AsccArgumentManager::GetInstance();
    if (argManager.SanitizerRequested()) {
        const auto &npuArch = argManager.GetNpuArch();
        if (npuArch == Ascc::ShortSoCVersion::ASCEND910B) {
            return " --dependent-libraries " + cannPath + "/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-vec.a "
                + cannPath + "/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-cube.a " + cannPath +
                "/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220.a";
        } else if (npuArch == Ascc::ShortSoCVersion::ASCEND310P) {
            return " --dependent-libraries " + cannPath + "/tools/mssanitizer/lib64/libsanitizer_stub_dav-m200-vec.a "
                + cannPath + "/tools/mssanitizer/lib64/libsanitizer_stub_dav-m200.a";
        }
    }
    return "";
}

// without -r option in process MergeObjFinal
AsccStatus AsccLink::LinkProcessForDeviceO(Ascc::AsccGlobalEnvManager& envVar, const std::string& coreType,
    const std::string& linkerPathBase) const
{
    std::string inputOFilePath = envVar.asccTmpPath + "/link_files/merge_obj/device_" + coreType + ".o";
    if (!Ascc::IsPathLegal(inputOFilePath) || !Ascc::IsParentDirValid(inputOFilePath)) {
        ASC_LOG_ASC_ERROR(LINK, "inputOFilePath [%s] does not exist!", inputOFilePath.c_str());
        return AsccStatus::FAILURE;
    }
    std::ifstream file(inputOFilePath);
    if (file.good()) {
        file.close();
        ASC_LOG_ASC_DEBUG(LINK, "Start Link process for device_%s.o in MergeObjFinal", coreType.c_str());
        std::string deviceOPath = envVar.asccTmpPath + "/link_files/merge_obj_final/device_" + coreType + ".o";
        std::string linkerCmd = linkerPathBase + " " + inputOFilePath + " -static -o " + deviceOPath;
        linkerCmd += SanitizerLinkProcess(envVar.ascendCannPackagePath);
        ASC_LOG_ASC_DEBUG(LINK, "LinkCommand: [%s].", linkerPathBase.c_str());
        Ascc::SaveCompileLogFile("Link command for merging obj with " + coreType + " (final):", linkerCmd);
        std::string output = "DEFAULT";
        int32_t returnCode = 0;
        std::tie(output, returnCode) = ExecuteCommand((linkerCmd + " 2>&1").c_str());
        ASCC_CHECK((returnCode == 0), {ASC_LOG_ASC_ERROR(LINK, "Execute command failed: %s", linkerCmd.c_str());});
    }
    return AsccStatus::SUCCESS;
}

// with -r option in process MergeObj
AsccStatus AsccLink::LinkProcessForDeviceOWithR(const std::string &linkCmdBase, const std::string &coreType,
    const std::string &outputPath, Ascc::KernelMode expectMode) const
{
    std::string linkCmd = linkCmdBase;
    const std::string& tmpAutoGenPath = AsccGlobalEnvManager::GetInstance().asccTmpAutoGenPath;
    std::vector<std::string> linkObjs;
    auto addLinkObj = [&linkObjs](const std::string& objPath) {
        if (PathCheck(objPath.c_str(), false) != PathStatus::NOT_EXIST) {
            linkObjs.emplace_back(objPath);
        }
    };
    if (expectMode == KernelMode::MIX) {
        addLinkObj(tmpAutoGenPath + "/device_stub_mix_aic.o");
        addLinkObj(tmpAutoGenPath + "/device_stub_mix_aic_1_1.o");
        addLinkObj(tmpAutoGenPath + "/device_stub_mix_aiv.o");
        addLinkObj(tmpAutoGenPath + "/device_stub_mix_aiv_1_1.o");
    }
    if (expectMode == KernelMode::AIC) {
        addLinkObj(tmpAutoGenPath + "/device_stub_aic.o");
    }
    if (expectMode == KernelMode::AIV) {
        addLinkObj(tmpAutoGenPath + "/device_stub_aiv.o");
    }
    for (const auto& objFile : linkObjs) {
        linkCmd += " " + objFile;
    }
    if (linkObjs.size() > 0) {
        ASC_LOG_ASC_DEBUG(LINK, "Start Link process for device_%s.o in MergeObj", coreType.c_str());
        linkCmd += " -static -o " + outputPath + "device_" + coreType + ".o";
        ASC_LOG_ASC_DEBUG(LINK, "LinkCommand: [%s].", linkCmd.c_str());
        Ascc::SaveCompileLogFile("Link command for merging obj with " + coreType + ":", linkCmd);
        std::string output = "DEFAULT";
        int32_t returnCode = 0;
        std::tie(output, returnCode) = ExecuteCommand((linkCmd + " 2>&1").c_str());
        ASCC_CHECK((returnCode == 0), {ASC_LOG_ASC_ERROR(LINK, "Execute command failed: %s", linkCmd.c_str());});
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccLink::PackProcessForDeviceO(Ascc::AsccGlobalEnvManager& envVar, const std::string& coreType,
    const std::string& packPrefix, const std::string& elfIn, const uint8_t coreTypeValue) const
{
    std::string inputOFilePath = envVar.asccTmpPath + "/link_files/merge_obj_final/device_" + coreType + ".o";
    if (!Ascc::IsPathLegal(inputOFilePath) || !Ascc::IsParentDirValid(inputOFilePath)) {
        ASC_LOG_ASC_ERROR(LINK, "inputOFilePath [%s] does not exist!", inputOFilePath.c_str());
        return AsccStatus::FAILURE;
    }
    std::ifstream file(inputOFilePath);
    if (file.good()) {
        file.close();
        ASC_LOG_ASC_DEBUG(LINK, "Start Pack process for device_%s.o", coreType.c_str());
        std::string packCmd = packPrefix + "device_" + coreType + ".o " + std::to_string(coreTypeValue) + " " + elfIn;
        ASC_LOG_ASC_DEBUG(LINK, "PackCommand: [%s].", packCmd.c_str());
        Ascc::SaveCompileLogFile("Pack command for ascendc_pack_kernel:", packCmd);
        std::string output = "DEFAULT";
        int32_t returnCode = 0;
        std::tie(output, returnCode) = ExecuteCommand((packCmd + " 2>&1").c_str());
        ASCC_CHECK((returnCode == 0), {ASC_LOG_ASC_ERROR(LINK, "Execute command failed: %s", packCmd.c_str());});
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccLink::MergeObjFinal(bool mergeFlag)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    std::string rOption;
    std::string xOption;
    if (mergeFlag == false) {
        rOption = "-r";
    }
    if (mergeFlag == true && buildType_ == BuildType::RELEASE) {
        xOption = "-x";
    }
    std::string linkCommand = linkerPath_ + " " + xOption + " -m aicorelinux " + rOption + " -Ttext=0";
    ASCC_CHECK((LinkProcessForDeviceO(envVar, "mix", linkCommand) == AsccStatus::SUCCESS), {});
    ASCC_CHECK((LinkProcessForDeviceO(envVar, "aic", linkCommand) == AsccStatus::SUCCESS), {});
    ASCC_CHECK((LinkProcessForDeviceO(envVar, "aiv", linkCommand) == AsccStatus::SUCCESS), {});
    return AsccStatus::SUCCESS;
}

AsccStatus AsccLink::MergeObj(const std::string &outputPath, bool mergeFlag)
{
    std::string rOption = "";
    std::string nOption = "";
    std::string xOption = "";

    if (mergeFlag == false) {
        rOption = "-r";
    }
    if (mergeFlag == true && buildType_ == BuildType::RELEASE) {
        xOption = "-x";
    }

    std::string linkCmdBase = linkerPath_ + " " + xOption + " -m aicorelinux " + rOption + " -Ttext=0";
    ASCC_CHECK((LinkProcessForDeviceOWithR(linkCmdBase, "mix", outputPath, Ascc::KernelMode::MIX)
        == AsccStatus::SUCCESS), {});
    ASCC_CHECK((LinkProcessForDeviceOWithR(linkCmdBase, "aic", outputPath, Ascc::KernelMode::AIC)
        == AsccStatus::SUCCESS), {});
    ASCC_CHECK((LinkProcessForDeviceOWithR(linkCmdBase, "aiv", outputPath, Ascc::KernelMode::AIV)
        == AsccStatus::SUCCESS), {});
    return AsccStatus::SUCCESS;
}

AsccStatus AsccLink::AsccPackKernel(
    const std::string &packTool, const std::string &elfIn, const std::string &addDirPreLink) const
{
    ASCC_CHECK((Ascc::PathCheck(packTool.c_str(), true) != Ascc::PathStatus::NOT_EXIST),
        {ASC_LOG_ASC_ERROR(LINK, "Ascendc pack tool path:%s does not exist!", packTool.c_str());});

    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    std::string packCmdPrefix = packTool + " " + elfIn + " " + addDirPreLink;
    // relevant with value in generate_aclrtlaunch_for_normal in extract_host_stub.py
    ASCC_CHECK(
        (PackProcessForDeviceO(envVar, "mix", packCmdPrefix, elfIn, ASCC_CORE_TYPE_MIX) == AsccStatus::SUCCESS), {});
    ASCC_CHECK(
        (PackProcessForDeviceO(envVar, "aic", packCmdPrefix, elfIn, ASCC_CORE_TYPE_AIC) == AsccStatus::SUCCESS), {});
    ASCC_CHECK(
        (PackProcessForDeviceO(envVar, "aiv", packCmdPrefix, elfIn, ASCC_CORE_TYPE_AIV) == AsccStatus::SUCCESS), {});
    return AsccStatus::SUCCESS;
}

// -l(lower L) or --library
void AsccLink::GetLibFileDependency()
{
    // The order of libxxx.so file is critical, and switching orders can lead to errors.
    // 被依赖的库需要放在靠后
    const auto& inputArgs = Ascc::AsccArgumentManager::GetInstance().GetCompileOptions();
    for (const auto& linkFile : inputArgs.linkFiles) {
        libFileDependency_.emplace_back(linkFile);
    }
    libFileDependency_.emplace_back("graph_base");
    libFileDependency_.emplace_back("tiling_api");
    libFileDependency_.emplace_back("platform");
    libFileDependency_.emplace_back("ascendcl");
    libFileDependency_.emplace_back("ascendc_runtime");
    libFileDependency_.emplace_back("runtime");
    libFileDependency_.emplace_back("register");
    libFileDependency_.emplace_back("error_manager");
    libFileDependency_.emplace_back("profapi");
    libFileDependency_.emplace_back("ge_common_base");
    libFileDependency_.emplace_back("ascendalog");
    libFileDependency_.emplace_back("mmpa");
    libFileDependency_.emplace_back("dl");
    libFileDependency_.emplace_back("ascend_dump");
    libFileDependency_.emplace_back("c_sec");
}

// -L or --library-path
void AsccLink::GetLibPathDependency()
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    libPathDependency_.emplace_back(envVar.ascendCannPackagePath + "/lib64");
    libPathDependency_.emplace_back(envVar.ascendCannPackagePath + "/tools/simulator/" +
        Ascc::AsccArgumentManager::GetInstance().GetNpuArchStr()+ "/lib");

    const auto& inputArgs = Ascc::AsccArgumentManager::GetInstance().GetCompileOptions();
    for (const auto& linkPath : inputArgs.linkPath) {
        libPathDependency_.emplace_back(linkPath);
    }
}

// link all files together into executable files
std::string AsccLink::CommandForFileExec(
    Ascc::AsccGlobalEnvManager &envVar, std::string &commonPart, std::string &libSuffix) const
{
    std::string compileExecCommand = envVar.cppCompilerPath + " " + commonPart + libSuffix;
    if (Ascc::AsccArgumentManager::GetInstance().DebugRequested()) {
        compileExecCommand += " -g";
    }
    ASC_LOG_ASC_DEBUG(LINK, "Compile command for executable : [%s].", compileExecCommand.c_str());
    Ascc::SaveCompileLogFile("Compile command for executable :", compileExecCommand);
    return compileExecCommand;
}

// compile result file xxx.so
std::string AsccLink::CommandForFileSo(
    Ascc::AsccGlobalEnvManager &envVar, std::string &commonPart, std::string &libSuffix) const
{
    std::string compileExecCommand = envVar.cppCompilerPath + " -shared " + commonPart + libSuffix;
    if (Ascc::AsccArgumentManager::GetInstance().DebugRequested()) {
        compileExecCommand += " -g";
    }
    ASC_LOG_ASC_DEBUG(LINK, "Compile command for .so : [%s].", compileExecCommand.c_str());
    Ascc::SaveCompileLogFile("Compile command for .so :", compileExecCommand);
    return compileExecCommand;
}

// compile result file xxx.o
std::string AsccLink::CommandForFileO(Ascc::AsccGlobalEnvManager& envVar, std::string& commonPart) const
{
    // only combine multiple .o into one .o       Do not need to link library
    std::string compileExecCommand = envVar.ldPath + " -r " + commonPart;
    ASC_LOG_ASC_DEBUG(LINK, "Compile command for .o : [%s].", compileExecCommand.c_str());
    Ascc::SaveCompileLogFile("Compile command for .o :", compileExecCommand);
    return compileExecCommand;
}

// Ex: -o libadd.so => inputFileName: libadd.so
const std::string AsccLink::GetOutputFileName()
{
    if (!Ascc::AsccArgumentManager::GetInstance().GetOutputFile().empty()) {   // when has -o xxx during parsing
        return Ascc::AsccArgumentManager::GetInstance().GetOutputFile();
    }
    std::string outputExecPath = "a.out";
    Ascc::OutputFileType compileMode = Ascc::AsccArgumentManager::GetInstance().GetCompileMode();
    if (compileMode == Ascc::OutputFileType::FILE_O) {
        size_t lastdot = inputFileName_.find_last_of(".");
        if (lastdot != std::string::npos) {
            outputExecPath = inputFileName_.substr(0, lastdot) + ".o";  // add.cpp -> add.o
        } else {
            outputExecPath = inputFileName_;
        }
    }
    ASC_LOG_ASC_WARN(LINK, "The output path is not set. The default path %s is used.", outputExecPath.c_str());
    return outputExecPath;
}


// Generate .o / .so / executable
AsccStatus AsccLink::GenResFile(Ascc::AsccGlobalEnvManager& envVar)
{
    // compile exec
    std::string hostCodeObjPath = envVar.asccTmpAutoGenPath + "/" + inputFileName_ + ".o";
    const std::string outputExecPath = GetOutputFileName();

    const auto& inputArgs = Ascc::AsccArgumentManager::GetInstance().GetCompileOptions();
    std::vector<std::string> macros = inputArgs.definitions;
    GetLibFileDependency();
    GetLibPathDependency();
    std::string commonPart = hostCodeObjPath + " -o " + outputExecPath;
    std::string libSuffix = "";
    for (auto& libPath : libPathDependency_) {
        libSuffix += " -L" + libPath;
    }
    for (auto& libFile : libFileDependency_) {
        libSuffix += " -l" + libFile;
    }
    for (auto& macro : macros) {
        libSuffix += " -D" + macro;
    }

    std::string compileCommand = "";
    if (Ascc::AsccArgumentManager::GetInstance().GetCompileMode() == Ascc::OutputFileType::FILE_EXECUTABLE) {
        compileCommand = CommandForFileExec(envVar, commonPart, libSuffix);              // Generate executable
    } else if (Ascc::AsccArgumentManager::GetInstance().GetCompileMode() == Ascc::OutputFileType::FILE_SO) {
        compileCommand = CommandForFileSo(envVar, commonPart, libSuffix);                // Generate .so
    } else {                                                                     // Generate .o
        std::string commonPartForO = hostCodeObjPath + " -o " + outputExecPath;  // do not need elf + rts file
        compileCommand = CommandForFileO(envVar, commonPartForO);
    }

    std::string output = "DEFAULT";
    int32_t returnCode = 0;
    std::tie(output, returnCode) = ExecuteCommand((compileCommand + " 2>&1").c_str());
    ASCC_CHECK((returnCode == 0),
        {ASC_LOG_ASC_ERROR(LINK, "Execute command failed: [%s]!", compileCommand.c_str());});
    std::cout << "Compiled products linked successfully." << std::endl;
    std::cout << "The generated executable file path is: " << outputExecPath << std::endl;
    return AsccStatus::SUCCESS;
}

// whether find kernel function / kernel function with templates in input .cpp file
bool AsccLink::IsKernelFuncFound() const
{
    return AsccMatchGlobalInfo::GetInstance().HasKernelCall();
}

AsccStatus AsccLink::AscendLink(AsccHostStub &hostStubGenerator)
{
    Ascc::AsccGlobalEnvManager& envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    bool isKernelFuncFound = IsKernelFuncFound();
    if (isKernelFuncFound) {
        if (Ascc::AsccArgumentManager::GetInstance().GetPreTaskType() == PreTaskType::NONE) {
            // 2. Merge multiple device.o into one
            // (device_stub_xxx.cpp.aic.o, device_stub_xxx.cpp.aiv.o) -> (device_aic.o, device_aiv.o)
            ASCC_CHECK((Ascc::PathCheck(linkerPath_.c_str(), true) != Ascc::PathStatus::NOT_EXIST),
                {ASC_LOG_ASC_ERROR(LINK, "Linker path [%s] does not exist!", linkerPath_.c_str());});
            std::string outputPath = envVar.asccTmpPath + "/link_files/merge_obj/";
            ASCC_CHECK((MergeObj(outputPath, false) == AsccStatus::SUCCESS),
                {ASC_LOG_ASC_ERROR(LINK, "Merge device_stub_xxx.cpp.aix.o failed!");});
            // device_aic.o + device_aiv.o -> device.o
            ASCC_CHECK((MergeObjFinal(true) == AsccStatus::SUCCESS),
                {ASC_LOG_ASC_ERROR(LINK, "Merge mix device_xxx.o failed!");});
        }

        // 3. update host_stub.h some __replace param by device_xxx.o
        ASCC_CHECK((hostStubGenerator.UpdateHostStubByDevice() == AsccStatus::SUCCESS),
            {ASC_LOG_ASC_ERROR(LINK, "UpdateHostStubByDevice failed!");});
    }

    AsccMatchGlobalKernel kernel;
    ASC_LOG_ASC_INFO(LINK, "Begin to match and generate host stub function.");
    kernel.MatchAndGenerateGlobalKernel();            // replace kernel function callexpr
    // 集中替换全部所需文件，注册host编译任务
    Ascc::AsccHostCompile hostCompiler;
    hostCompiler.HostCompile();                       // replace kernel function definition

    // if for sub_module path task, do not need to compile host stub files
    if (Ascc::AsccArgumentManager::GetInstance().GetPreTaskType() == Ascc::PreTaskType::HOST) {
        return AsccStatus::SUCCESS;
    }
    if (isKernelFuncFound) {
        std::string taskFile = AsccGlobalEnvManager::GetInstance().hostCompileFile;
        auto task = Ascc::GetCompileTask<Ascc::AsccCompileHost>(taskFile);
        if (task != nullptr) {
            task->AddIncFile(hostStubGenerator.GetHostStubFilePath());
        }

        auto& factory = Ascc::AsccCompileFactory::GetInstance();
        ASCC_CHECK((factory.Produce() == AsccStatus::SUCCESS), {ASC_LOG_ASC_ERROR(COMPILE, "Compile failed!");});

        // host.o + device_aiv.o => host.o
        std::string packTool = envVar.ascendCannPackagePath + "/bin/ascendc_pack_kernel";
        std::string elfIn = envVar.asccTmpAutoGenPath + "/" + inputFileName_ + ".o";
        std::string addDirPreLink = envVar.asccTmpPath + "/link_files/merge_obj_final/";
        AsccStatus result = AsccPackKernel(packTool, elfIn, addDirPreLink);
        ASCC_CHECK((result == AsccStatus::SUCCESS), {ASC_LOG_ASC_ERROR(LINK, "AsccPackKernel failed!");});

        std::cout << "Host and device files compiled successfully." << std::endl;
    } else {
        // recompile
        auto& factory = Ascc::AsccCompileFactory::GetInstance();
        ASCC_CHECK((factory.Produce() == AsccStatus::SUCCESS), {ASC_LOG_ASC_ERROR(COMPILE, "Compile failed!");});
    }

    return GenResFile(envVar);
}
}