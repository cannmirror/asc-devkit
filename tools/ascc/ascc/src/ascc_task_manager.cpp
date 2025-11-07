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
 * \file task_manager.cpp
 * \brief
 */

#include "ascc_task_manager.h"

#include <chrono>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <unordered_set>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/JSON.h>
#include "llvm/Support/FileSystem.h"

#include "ascc_option.h"
#include "ascc_utils.h"
#include "task_executor.h"

namespace Ascc {

static constexpr char DEFAULT_ASCC_TMP_PATH[] = "/tmp/bishengcc";
static constexpr char DEFAULT_EXEC_NAME[] = "a.out";
static constexpr char DEFAULT_SHARED_LIB_NAME[] = "a.out";

const std::vector<std::string> HOST_COMMON_COMPILE_OPTIONS = {"-x", "cce" , "--cce-host-only",
    "-D_GLIBCXX_USE_CXX11_ABI=0", "-fPIC", "-std=c++17", "-fcce-kernel-launch-custom"
};
const std::vector<std::string> LINK_LIB = {"-lascendc_runtime", "-lgraph_base", "-ltiling_api", "-lplatform",
    "-lascendcl", "-lruntime", "-lregister", "-lerror_manager", "-lprofapi", "-lge_common_base", "-lascendalog",
    "-lmmpa", "-ldl", "-lascend_dump", "-lc_sec"
};

const std::vector<std::string> LDLLD_PROCESS_ARG = {"-m", "aicorelinux", "-r", "-Ttext=0"};
const std::vector<std::string> LDLLD_FINAL_PROCESS_ARG = {"-m", "aicorelinux", "-Ttext=0"};

// preprocess not need link, thus not inserted library related args.
// Optimize is inserted manually, thus not included here
const std::vector<std::string> COMMON_COMPILE_OPTIONS = {
    "I",
    "D",    // current not support define-macro in ascendfe
    "U",    // current not supported
    "g",
};

const std::vector<std::string> DEPENDENCY_COMPILE_OPTIONS = {"MD", "MMD", "MP", "MF", "MT"};
const std::vector<std::string> LINK_COMPILE_OPTIONS = {"L", "l"};

namespace {
// extra check after parsing arguments
AsccStatus CheckArgInfo(const ArgInfo& argInfo)
{
    ASCC_CHECK((SUPPORTED_SOC_LIST.find(argInfo.socVersion) != SUPPORTED_SOC_LIST.end()),
        {Ascc::HandleErrorAndCheckLog("Do not support npu architecture [" + argInfo.socVersion + "]!");});

    std::string optLevel = argInfo.optLevel;
    ASCC_CHECK((optLevel == "-O0" || optLevel == "-O1" || optLevel == "-O2" || optLevel == "-O3"),
        {Ascc::HandleError("Invalid optimize level [" + optLevel.substr(2) + "], only support level 0, 1, 2, 3.");});

    std::string outPath = Ascc::GetFilePath(argInfo.outputFileName);
    ASCC_CHECK((!outPath.empty()),
        {Ascc::HandleError("Output Path directory " + argInfo.outputFileName + " does not exist!");});

    if (argInfo.needSaveTemps) {
        ASCC_CHECK((!CheckAndGetFullPath(argInfo.tmpPath).empty()),
            {Ascc::HandleError("Tmp path " + argInfo.tmpPath + " does not exist!");});
    }
    return AsccStatus::SUCCESS;
}

void InitExecutor(TaskExecutor& executor, const ArgInfo& argInfo)
{
    if (argInfo.verboseStatus) {
        executor.EnablePrintTasks();
    }
    if (argInfo.timeStatus) {
        executor.EnableStatistics();
    }
}

void PrintMsgForExecResult(const std::string& outputFileName, OutputFileType outputMode)
{
    printf("========== Execution Result ==========\n");
    if (outputMode == OutputFileType::FILE_O) {
        printf("The generated object file path is: %s.\n", outputFileName.c_str());
    } else if (outputMode == OutputFileType::FILE_SO) {
        printf("The generated shared library file path is: %s.\n", outputFileName.c_str());
    } else {
        printf("The generated executable file path is: %s.\n", outputFileName.c_str());
    }
    printf("======================================\n");
}

void ErrorMsgForProcessFile(const std::string& file)
{
    Ascc::HandleErrorAndCheckLog("Task execution of " + file + " failed.");
}

// json file process
std::string ReadJsonFileToString(const std::string& jsonName)
{
    if (PathCheck(jsonName.c_str(), true) == Ascc::PathStatus::NOT_EXIST) {
        Ascc::HandleError("Json file " + jsonName + " not exists.");
        return "";
    }
    std::ifstream file(jsonName);
    if (!file.is_open()) {
        Ascc::HandleError("Cannot open json file " + jsonName);
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void ReportArgInfo(const ArgInfo& argInfo)
{
    ASCC_LOGI("argInfo.socVersion is %s.", argInfo.socVersion.c_str());
    ASCC_LOGI("argInfo.optLevel is %s.", argInfo.optLevel.c_str());
    ASCC_LOGI("argInfo.bishengOptLevel is %s.", argInfo.bishengOptLevel.c_str());
    ASCC_LOGI("argInfo.outputFileName is %s.", argInfo.outputFileName.c_str());
    ASCC_LOGI("argInfo.tmpPath is %s.", argInfo.tmpPath.c_str());
    ASCC_LOGI("argInfo.needSaveTemps is %d.", argInfo.needSaveTemps);
    ASCC_LOGI("argInfo.verboseStatus is %d.", argInfo.verboseStatus);
    ASCC_LOGI("argInfo.timeStatus is %d.", argInfo.timeStatus);
    ASCC_LOGI("argInfo.hasUsersDump is %d.", argInfo.hasUsersDump);
    ASCC_LOGI("argInfo.outputMode is %u.", static_cast<uint32_t>(argInfo.outputMode));
}

void ReportPathInfo(const PathInfo& pathInfo)
{
    ASCC_LOGI("pathInfo.ascendFePath is %s.", pathInfo.ascendFePath.c_str());
    ASCC_LOGI("pathInfo.bishengPath is %s.", pathInfo.bishengPath.c_str());
    ASCC_LOGI("pathInfo.gppPath is %s.", pathInfo.gppPath.c_str());
    ASCC_LOGI("pathInfo.ldlldPath is %s.", pathInfo.ldlldPath.c_str());
    ASCC_LOGI("pathInfo.cannPath is %s.", pathInfo.cannPath.c_str());
    ASCC_LOGI("pathInfo.ascPackKernelPath is %s.", pathInfo.ascPackKernelPath.c_str());
    ASCC_LOGI("pathInfo.tmpFilePath is %s.", pathInfo.tmpFilePath.c_str());
    ASCC_LOGI("pathInfo.includePath is %s.", pathInfo.includePath.c_str());
    ASCC_LOGI("pathInfo.objectFilePath is %s.", pathInfo.objectFilePath.c_str());
    ASCC_LOGI("pathInfo.preprocessDevicePath is %s.", pathInfo.preprocessDevicePath.c_str());
    ASCC_LOGI("pathInfo.preprocessHostPath is %s.", pathInfo.preprocessHostPath.c_str());
    ASCC_LOGI("pathInfo.deviceStubPath is %s.", pathInfo.deviceStubPath.c_str());
    ASCC_LOGI("pathInfo.hostStubPath is %s.", pathInfo.hostStubPath.c_str());
}
} // namespace

const std::string GetCoreTypeStr(const CoreType coreType)
{
    if (CORE_TYPE_STR_MAP.find(coreType) == CORE_TYPE_STR_MAP.end()) {
        ASCC_LOGE("GetCoreTypeStr with coreType %u failed!", static_cast<uint32_t>(coreType));
        return "";
    }
    return CORE_TYPE_STR_MAP.at(coreType);
}

AsccStatus GetJsonInfo(std::unordered_map<uint8_t, std::vector<std::string>>& kernelType,
    std::unordered_map<uint8_t, uint32_t>& mixNumLineMap, std::unordered_map<uint8_t, uint32_t>& dumpUintLineMap,
    std::unordered_map<uint8_t, uint32_t>& dumpWorkspaceLineMap, std::unordered_map<uint8_t, uint32_t>& dumpSizeMap,
    std::unordered_map<uint8_t, std::string>& stubFileMap, const std::string& jsonName)
{
    std::string jsonContent = ReadJsonFileToString(jsonName);
    ASCC_CHECK((!jsonContent.empty()), {ASCC_LOGE("json content from %s is empty.", jsonName.c_str());});

    llvm::Expected<llvm::json::Value> parsedJson = llvm::json::parse(jsonContent);
    ASCC_CHECK((parsedJson),
        {Ascc::HandleError("Parse json file " + jsonName + " error: " + toString(parsedJson.takeError()));});

    const llvm::json::Object *root = parsedJson->getAsObject();
    ASCC_CHECK((root), {Ascc::HandleError("Cannot get root json object from " + jsonName);});

    for (auto it = root->begin(); it != root->end(); ++it) {
        const llvm::json::Object* kernelTypeObj = it->second.getAsObject();
        ASCC_CHECK((kernelTypeObj), {Ascc::HandleError("Cannot get kernelTypeObj json object from " + jsonName);});
        llvm::Optional<int64_t> intPtr = kernelTypeObj->getInteger("kernel_type");
        std::string mangleName = it->first.str();
        uint8_t kernelTypeEnum = static_cast<uint8_t>(*intPtr);
        auto itKernel = kernelType.find(kernelTypeEnum);
        if (itKernel == kernelType.end()) {
            kernelType[kernelTypeEnum] = {mangleName};
        } else {
            itKernel->second.push_back(mangleName);
        }
        llvm::Optional<int64_t> numPtr = kernelTypeObj->getInteger("mix_num_lineno");
        mixNumLineMap[kernelTypeEnum] = static_cast<uint32_t>(*numPtr);
        numPtr = kernelTypeObj->getInteger("dump_uint_lineno");
        dumpUintLineMap[kernelTypeEnum] = static_cast<uint32_t>(*numPtr);
        numPtr = kernelTypeObj->getInteger("dump_workspace_lineno");
        dumpWorkspaceLineMap[kernelTypeEnum] = static_cast<uint32_t>(*numPtr);
        numPtr = kernelTypeObj->getInteger("dump_size");
        dumpSizeMap[kernelTypeEnum] = static_cast<uint32_t>(*numPtr);
        llvm::Optional<llvm::StringRef> stubFilePtr = kernelTypeObj->getString("stub_filename");
        stubFileMap[kernelTypeEnum] = stubFilePtr->str();
    }

    return AsccStatus::SUCCESS;
}

// return empty if not found executable
const std::string FindExecPath(std::string execName)
{
    std::string pathCmd = "which " + execName + " 2>/dev/null";
    FILE* pipe = popen(pathCmd.c_str(), "r");
    if (pipe) {
        char buffer[2048];  //  set up as 2048 to store path
        if (fgets(buffer, sizeof(buffer), pipe)) {
            std::string path(buffer);
            // remove \n symbol
            size_t pos = path.find_last_not_of("\n");
            if (pos != std::string::npos) {
                path.erase(pos + 1);
            }
            pclose(pipe);
            if (access(path.c_str(), X_OK) == 0) {
                ASCC_LOGI("Path of executable %s is: %s.", execName.c_str(), path.c_str());
                return path;
            }
        } else {
            pclose(pipe);
        }
    }
    ASCC_LOGE("Cannot find path of executable: [%s].", execName.c_str());
    return "";
}

// Get soc for /tools/simulator/Ascendxxxx/lib
const std::string GetSocForSimulator(const std::string& socVersion)
{
    if (socVersion == "Ascend910B4-1") {
        return "Ascend910B4";      // B4-1 reads the same lib as Ascend910B4
    }
    return socVersion;
}

// extra check for files
AsccStatus CheckFiles(const std::vector<std::string> files, const ArgInfo& argInfo)
{
    uint32_t srcFileSize = files.size();
    ASCC_CHECK((srcFileSize > 0), {Ascc::HandleError("No input files.");});

    for (auto& file : files) {
        ASCC_CHECK((!CheckAndGetFullPath(file).empty()),
            {Ascc::HandleError("Input path " + file + " does not exist!");});
        std::string inputSuffix = GetSuffix((file));
        ASCC_CHECK((inputSuffix == ".C" || inputSuffix == ".c" || inputSuffix == ".cc" || inputSuffix == ".cxx" ||
            inputSuffix == ".cpp" || inputSuffix == ".asc"), {Ascc::HandleError("Input file suffix " + inputSuffix
            + " is not supported, only support following suffix: .C / .c / .cc / .cxx / .cpp / .asc.");});
    }

    if (srcFileSize > 1 && argInfo.outputMode == OutputFileType::FILE_O) {
        ASCC_CHECK((argInfo.outputFileName.empty()),
            {Ascc::HandleError("cannot specify '-o' with '-c' with multiple files");}); // Note: also apply to -S / -E
    }
    return AsccStatus::SUCCESS;
}

AsccStatus InitArgInfo(ArgInfo& argInfo)
{
    std::vector<std::string> status;
    ASCC_CHECK((GetOptionValuesByArgName("arch", status)), {ASCC_LOGE("Cannot get argument value by arch");});
    argInfo.socVersion = status[0];

    status.clear();
    if (GetOptionValuesByArgName("optimize", status)) {
        argInfo.optLevel = "-O" + status[0];
    } else {
        argInfo.optLevel = "-O3";
    }
    if (argInfo.optLevel == "-O1") {        // bisheng does not support -O1, use -O2 instead
        argInfo.bishengOptLevel = "-O2";
        ASCC_LOGW("bisheng does not support optimize level 1. Thus use -O2 instead.");
    } else {
        argInfo.bishengOptLevel = argInfo.optLevel;
    }

    status.clear();
    if (GetOptionValuesByArgName("output-file", status)) {
        argInfo.outputFileName = status[0];
    }

    status.clear();
    if (GetOptionValuesByArgName("save-temps", status)) {
        argInfo.needSaveTemps = true;
        if (!status[0].empty()) {
            argInfo.tmpPath = status[0];
        } else {
            argInfo.tmpPath = DEFAULT_ASCC_TMP_PATH;
        }
    }
    argInfo.verboseStatus = GetOptionValuesByArgName("verbose", status);
    argInfo.timeStatus = GetOptionValuesByArgName("time", status);

    // when -c + -shared both appear at same time, always -c applies.
    if (GetOptionValuesByArgName("compile", status)) {
        argInfo.outputMode = OutputFileType::FILE_O;
    } else if (GetOptionValuesByArgName("shared", status)) {
        argInfo.outputMode = OutputFileType::FILE_SO;
    } else {
        argInfo.outputMode = OutputFileType::FILE_EXECUTABLE;
    }

    status.clear();
    if(GetOptionValuesByArgName("D", status)){
        for (const auto& value : status) {
            if (value.find("ASCENDC_DUMP") != std::string::npos) {
                argInfo.hasUsersDump = true;
                break;
            }
        }
    }
    ReportArgInfo(argInfo);
    return CheckArgInfo(argInfo);
}

AsccStatus InitPathInfo(PathInfo& pathInfo, const ArgInfo& argInfo)
{
    const std::string gppPath = Ascc::FindExecPath("g++");
    ASCC_CHECK((!gppPath.empty()), {Ascc::HandleError("Cannot find path for g++.");});
    const std::string ascendFePath = Ascc::FindExecPath("ascendfe");
    ASCC_CHECK((!ascendFePath.empty()), {Ascc::HandleError("Cannot find path for ascendfe.");});
    const std::string bishengPath = Ascc::FindExecPath("bisheng");
    ASCC_CHECK((!bishengPath.empty()), {Ascc::HandleError("Cannot find path for bisheng.");});
    const std::string ldlldPath = Ascc::FindExecPath("ld.lld");
    ASCC_CHECK((!ldlldPath.empty()), {Ascc::HandleError("Cannot find path for ld.lld.");});
    const char* ascHomePath = getenv("ASCEND_HOME_PATH");
    ASCC_CHECK((ascHomePath != nullptr),
        {Ascc::HandleError("ASCEND_HOME_PATH is not set. Please source setenv.bash");});
    const std::string cannPath(ascHomePath);

    pathInfo.ascendFePath = ascendFePath;
    pathInfo.bishengPath = bishengPath;
    pathInfo.gppPath = gppPath;
    pathInfo.ldlldPath = ldlldPath;
    pathInfo.cannPath = cannPath;
    pathInfo.ascPackKernelPath = cannPath + "/bin/ascendc_pack_kernel";

    std::string socSimulator = GetSocForSimulator(argInfo.socVersion);
    ASCC_CHECK((!socSimulator.empty()), {ASCC_LOGE("Cannot find soc version for simulator path.");});
    pathInfo.linkPath = {"-L" + pathInfo.cannPath + "/lib64",
        "-L" + pathInfo.cannPath + "/tools/simulator/" + socSimulator + "/lib"};

    std::stringstream ss;
    ss << getpid() << "_" << syscall(SYS_gettid);   // make tmp directories unique
    std::string tmpFullPath = std::string(DEFAULT_ASCC_TMP_PATH);
    if (argInfo.needSaveTemps) {
        if (!argInfo.tmpPath.empty()) {
            tmpFullPath = Ascc::CheckAndGetFullPath(argInfo.tmpPath);
            ASCC_CHECK((!tmpFullPath.empty()),
                {Ascc::HandleError("Cannot find path for save-temps: " + argInfo.tmpPath);});
        }
        pathInfo.tmpFilePath = tmpFullPath + std::string("/") + GenerateTimestamp() + std::string("_") + ss.str();
        printf("Temporary file storage path is set up as: %s\n", pathInfo.tmpFilePath.c_str());
    } else {
        pathInfo.tmpFilePath = tmpFullPath + std::string("/") + GenerateTimestamp() + std::string("_") + ss.str();
    }
    pathInfo.includePath = pathInfo.tmpFilePath + "/include";
    pathInfo.objectFilePath = pathInfo.tmpFilePath + "/object_files";
    pathInfo.preprocessDevicePath = pathInfo.tmpFilePath + "/preprocess_device";
    pathInfo.preprocessHostPath = pathInfo.tmpFilePath + "/preprocess_host";
    pathInfo.deviceStubPath = pathInfo.tmpFilePath + "/stub_files_device";
    pathInfo.hostStubPath = pathInfo.tmpFilePath + "/stub_files_host";
    ReportPathInfo(pathInfo);
    return AsccStatus::SUCCESS;
}

int64_t GetDeviceFileSize(const std::string& basePath, const std::string& suffix)
{
    const std::string filename = "/device_" + suffix + ".o";
    std::string fullPath = basePath + filename;
    if (PathCheck(fullPath.c_str(), false) == Ascc::PathStatus::NOT_EXIST) {
        ASCC_LOGW("Cannot find file %s in ReplaceStubFile.", fullPath.c_str());
        return 0LL;  // returns 0 for error
    }
    std::ifstream deviceFile(fullPath, std::ios::binary | std::ios::ate);
    return static_cast<int64_t>(deviceFile.tellg());
}

// Example: R"(sed -i '5s/.*/ffff/' sss.cpp)"   replace 5th line in sss.cpp to ffff
const std::string GetReplaceCmd(const std::string& filename, const std::string& replaceContent, const uint32_t lineNo)
{
    return R"(sed -i ')" + std::to_string(lineNo) + R"(s/.*/)" + replaceContent + R"(/' )" + filename;
}

// replace "constexpr int32_t MIX_NUM = " in .cpp file
// replace "constexpr size_t DUMP_UINTSIZE = " in .cpp file
// replace "const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * = " in .cpp file
AsccStatus ReplaceStubFile(const std::string& fileName, CodeMode kernelTypeEnum, const uint32_t mixNumLineNo,
    const uint32_t dumpUintLineNo, const uint32_t dumpWorkspaceLineNo, const uint32_t dumpSize)
{
    ASCC_CHECK((PathCheck(fileName.c_str(), true) != Ascc::PathStatus::NOT_EXIST),
        {ASCC_LOGE("Cannot find file %s in ReplaceStubFile.", fileName.c_str());});

    // when mix 1_1 ->  value is 1,  otherwise always 2
    uint32_t expectedValue = (kernelTypeEnum == CodeMode::KERNEL_TYPE_MIX_AIC_1_1) ? 1 : 2;
    std::string mixNumStr = "constexpr int32_t MIX_NUM = " + std::to_string(expectedValue) + ";";
    const std::string replaceMixNumCmd = GetReplaceCmd(fileName, mixNumStr, mixNumLineNo);
    Ascc::ExecuteCommand(replaceMixNumCmd.c_str());

    std::string dumpUintSizeStr = "constexpr size_t DUMP_UINTSIZE = " + std::to_string(dumpSize) + ";";
    const std::string replaceDumpUintSizeCmd = GetReplaceCmd(fileName, dumpUintSizeStr, dumpUintLineNo);
    Ascc::ExecuteCommand(replaceDumpUintSizeCmd.c_str());

    std::string dumpWorkspaceStr = "const uint32_t DUMP_WORKSPACE_SIZE = DUMP_CORE_COUNT * " + std::to_string(dumpSize)
        + ";";
    const std::string replaceDumpWorkspaceCmd = GetReplaceCmd(fileName, dumpWorkspaceStr, dumpWorkspaceLineNo);
    Ascc::ExecuteCommand(replaceDumpWorkspaceCmd.c_str());
    return AsccStatus::SUCCESS;
}

std::string GetDeviceStubMacro(const CoreType coreType, const std::string& mangleName, const CodeMode codeMode)
{
    const std::string coreTypeStr = GetCoreTypeStr(coreType);
    if (coreTypeStr == "vec" && (codeMode == CodeMode::KERNEL_TYPE_MIX_AIV_1_0 ||
        codeMode == CodeMode::KERNEL_TYPE_MIX_AIC_1_1 || codeMode == CodeMode::KERNEL_TYPE_MIX_AIC_1_2)) {
        return "-D" + mangleName + "=" + mangleName + "_mix_aiv";
    }
    if (coreTypeStr == "cube"  && (codeMode == CodeMode::KERNEL_TYPE_MIX_AIC_1_0 ||
        codeMode == CodeMode::KERNEL_TYPE_MIX_AIC_1_1 || codeMode == CodeMode::KERNEL_TYPE_MIX_AIC_1_2)) {
        return "-D" + mangleName + "=" + mangleName + "_mix_aic";
    }
    return "";
}

// get the tmp folder name of user input file
std::string GetInputFileStubDirName(const std::string& file)
{
    std::string filePath = RemoveSuffix(file);
    std::replace(filePath.begin(), filePath.end(), '/', '_');
    return filePath;
}

AsccStatus CreateTmpDirectory(const std::vector<std::string>& files, const PathInfo& pathInfo)
{
    ASCC_CHECK((CreateDirectory(pathInfo.tmpFilePath) == AsccStatus::SUCCESS), {});
    ASCC_CHECK((CreateDirectory(pathInfo.includePath) == AsccStatus::SUCCESS), {});
    ASCC_CHECK((CreateDirectory(pathInfo.objectFilePath) == AsccStatus::SUCCESS), {});
    ASCC_CHECK((CreateDirectory(pathInfo.preprocessHostPath) == AsccStatus::SUCCESS), {});
    ASCC_CHECK((CreateDirectory(pathInfo.preprocessDevicePath) == AsccStatus::SUCCESS), {});
    ASCC_CHECK((CreateDirectory(pathInfo.deviceStubPath) == AsccStatus::SUCCESS), {});
    ASCC_CHECK((CreateDirectory(pathInfo.hostStubPath) == AsccStatus::SUCCESS), {});
    for (auto file : files) {
        // Assume file is add/add_custom.cpp -> FilePath: add_add_custom
        std::string filePath = GetInputFileStubDirName(file);
        ASCC_CHECK((CreateDirectory(pathInfo.objectFilePath + "/" + filePath) == AsccStatus::SUCCESS), {});
        ASCC_CHECK((CreateDirectory(pathInfo.preprocessHostPath + "/" + filePath) == AsccStatus::SUCCESS), {});
        ASCC_CHECK((CreateDirectory(pathInfo.preprocessDevicePath  + "/" + filePath) == AsccStatus::SUCCESS), {});
        ASCC_CHECK((CreateDirectory(pathInfo.deviceStubPath + "/" + filePath + "/cube") == AsccStatus::SUCCESS), {});
        ASCC_CHECK((CreateDirectory(pathInfo.deviceStubPath + "/" + filePath + "/vec") == AsccStatus::SUCCESS), {});
        ASCC_CHECK((CreateDirectory(pathInfo.hostStubPath + "/" + filePath) == AsccStatus::SUCCESS), {});
    }
    return AsccStatus::SUCCESS;
}

// Example: input: {define-macro, include-path}
//          output: {"-DADD", "-Ipath"}
// Note: in the future, all options should be long arg str instead of short ones
const std::vector<std::string> GetCompileOptionContent(const std::vector<std::string>& compileOptions)
{
    std::vector<std::string> res;
    std::vector<std::string> tmpStatus;
    for (auto& opt : compileOptions) {
        tmpStatus.clear();
        if(GetOptionValuesByArgName(opt, tmpStatus)){
            for (auto& value : tmpStatus) {
                if (opt == "MF" || opt == "MT") {
                    res.emplace_back("-" + opt);
                    res.emplace_back(value);
                } else if (opt == "sanitizer") {
                    res.emplace_back("--cce-enable-sanitizer");
                } else {
                    res.emplace_back("-" + opt + value);
                }
            }
        }
    }
    return res;
}

const std::vector<std::string> GetDependencyCompileOptionContent(const std::string& filename)
{
    std::vector<std::string> compileOptions = GetCompileOptionContent(DEPENDENCY_COMPILE_OPTIONS);
    std::vector<std::string> tmpStatus;
    // only when user passed in -MD or -MMD, then need to process -MF and -MT option
    if(!(GetOptionValuesByArgName("MD", tmpStatus) || GetOptionValuesByArgName("MMD", tmpStatus))){
        return compileOptions;
    }

    tmpStatus.clear();
    if(!GetOptionValuesByArgName("MF", tmpStatus)){
       // need to set -MF to original path
        std::string srcFileName;
        if(GetOptionValuesByArgName("output-file", tmpStatus)){
            srcFileName = tmpStatus[0];
        } else {
            srcFileName = filename;
        }
        std::string expectedDFileName = Ascc::RemoveSuffix(GetFileName(srcFileName)) + ".d";
        compileOptions.emplace_back("-MF");
        compileOptions.emplace_back(expectedDFileName);
    }
    tmpStatus.clear();
    if(!GetOptionValuesByArgName("MT", tmpStatus)){
       // need to set -MT to original path
        std::string expectedFileName;
        if (GetOptionValuesByArgName("output-file", tmpStatus)) { // when with -o, filename = output filename
            expectedFileName = tmpStatus[0];
        } else {                              // when without -o, filename = replace input file suffix with .o
            expectedFileName = Ascc::RemoveSuffix(GetFileName(filename)) + ".o";
        }
        compileOptions.emplace_back("-MT");
        compileOptions.emplace_back(expectedFileName);
    }

    return compileOptions;
}

// for compile shared library / executable
const std::vector<std::string> GetLinkCompileOptionContent()
{
    std::vector<std::string> compileOptions = COMMON_COMPILE_OPTIONS;
    compileOptions.insert(compileOptions.end(), LINK_COMPILE_OPTIONS.begin(), LINK_COMPILE_OPTIONS.end());
    return GetCompileOptionContent(compileOptions);
}

// Compile options used in device compilation
std::vector<std::string> GetDeviceDefaultCompileOption(const ArgInfo& argInfo)
{
    return {"-DTILING_KEY_VAR=0", "-std=c++17", "--cce-aicore-block-local-init", "-x", "cce", "--cce-aicore-only",
        "-mllvm", "-cce-aicore-stack-size=0x8000", "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-record-overflow=true", "-mllvm", "-cce-aicore-addr-transform",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-D__NPU_DEVICE__", argInfo.bishengOptLevel, "-DONE_CORE_DUMP_SIZE=1048576", "-D__MIX_CORE_MACRO__=1"
    };
}

std::vector<std::string> GetHostDefaultIncludePath(const PathInfo& pathInfo)
{
    return {"-I" +  pathInfo.cannPath + "/include",
        "-I" +  pathInfo.cannPath + "/include/ascendc/host_api",
        "-I" +  pathInfo.cannPath + "/compiler/ascendc/include/highlevel_api",
        "-I" +  pathInfo.cannPath + "/compiler/tikcpp/tikcfw",
        "-I" +  pathInfo.cannPath + "/compiler/tikcpp/tikcfw/lib",
        "-I" +  pathInfo.cannPath + "/compiler/tikcpp/tikcfw/lib/matmul",
        "-I" +  pathInfo.cannPath + "/compiler/tikcpp/tikcfw/impl",
        "-I" +  pathInfo.cannPath + "/compiler/tikcpp/tikcfw/interface"
    };
}

std::vector<std::string> GetDeviceDefaultIncludePath(const PathInfo& pathInfo, const std::string& file)
{
    std::string fileDir = GetFilePath(file).c_str();
    return {"-I" + pathInfo.cannPath + "/include",
        "-I" + pathInfo.cannPath + "/compiler/ascendc/include/highlevel_api",
        "-I" + pathInfo.cannPath + "/compiler/tikcpp/tikcfw/lib",
        "-I" + pathInfo.cannPath + "/compiler/tikcpp/tikcfw",
        "-I" + pathInfo.cannPath + "/compiler/tikcpp/tikcfw/interface",
        "-I" + pathInfo.cannPath + "/compiler/tikcpp/tikcfw/impl",
        "-I" + fileDir,
        "-include", pathInfo.cannPath + "/include/version/cann_version.h"
    };
}

// coreType: vec / cube / host
const std::string GetPreprocessFileName(const PathInfo& pathInfo, const std::string& file, const std::string& coreType)
{
    std::string preprocessFileName = GetFileName(RemoveSuffix(file) + "_cpp_" + coreType + "_ii.cpp");
    std::string filePath = GetInputFileStubDirName(file);
    if (coreType == "host") {
        return pathInfo.preprocessHostPath + "/" + filePath + "/" + preprocessFileName;
    } else {
        return pathInfo.preprocessDevicePath + "/" + filePath + "/" +preprocessFileName;
    }
}

// Preprocess + Generate stub files
// Note: Include Path and link path/lib users first; otherwise default first
Task TaskPreprocessDevice(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo,
    const CoreType coreType)
{
    const std::string coreTypeStr = GetCoreTypeStr(coreType);
    std::string preprocessFileName = GetPreprocessFileName(pathInfo, file, coreTypeStr);
    std::vector<std::string> outFileStatus = {"-o", preprocessFileName};

    Task taskContent = Task(pathInfo.bishengPath).AddArg("-E");
    if (!argInfo.hasUsersDump) {
        taskContent.AddArg("-DASCENDC_DUMP=1");     // preprocess need to set to DUMP = 1 for AST analyze
    }
    taskContent.AddArg("-D__CHECK_FEATURE_AT_PRECOMPILE")
        .AddArg({"-D__ASCC_PRE__", "-D__NPU_DEVICE__", "-include", "cstdio"})
        .AddArg(GetDeviceDefaultCompileOption(argInfo))
        .AddArg(GetCompileOptionContent({"I", "D", "U", "g", "sanitizer"}))
        .AddArg(GetDeviceDefaultIncludePath(pathInfo, file))
        .AddArg("--cce-aicore-arch=dav-c220-" + coreTypeStr)
        .AddArg(outFileStatus)
        .AddArg(file);
    if (coreTypeStr == "vec") {
        taskContent.AddArg("--cce-auto-sync");
    }
    return taskContent;
}

Task TaskPreprocessHost(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo)
{
    std::string preprocessFileName = GetPreprocessFileName(pathInfo, file, "host");
    std::vector<std::string> outFileStatus = {"-o", preprocessFileName};

    Task taskContent = Task(pathInfo.bishengPath).AddArg("-E");
    if (!argInfo.hasUsersDump) {
        taskContent.AddArg("-DASCENDC_DUMP=1");     // preprocess need to set to DUMP = 1 for AST analyze
    }
    taskContent.AddArg("-D__CHECK_FEATURE_AT_PRECOMPILE")
        .AddArg("-D__ASCC_PRE__")
        .AddArg({"-D__NPU_HOST__", "-DTILING_KEY_VAR=0", "-include", "cstdio"})
        .AddArg(HOST_COMMON_COMPILE_OPTIONS)
        .AddArg(GetCompileOptionContent({"I", "D", "U", "g"}))
        .AddArg(GetHostDefaultIncludePath(pathInfo))
        .AddArg(argInfo.bishengOptLevel)
        .AddArg(outFileStatus)
        .AddArg(file);
    return taskContent;
}

// coreType: vec / cube / host.  Return the directory path of stub files
const std::string GetStubFileDirectory(const PathInfo& pathInfo, const std::string& file, const std::string& coreType)
{
    std::string filePath = GetInputFileStubDirName(file);
    if (coreType == "host") {
        return pathInfo.hostStubPath + "/" + filePath;
    } else {
        return pathInfo.deviceStubPath + "/" + filePath + "/" + coreType;
    }
}

// use ascendfe to generate host / device stub files
Task TaskHostStub(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo)
{
    std::string preprocessFileName = GetPreprocessFileName(pathInfo, file, "host");
    std::string hostStubDir = GetStubFileDirectory(pathInfo, file, "host");

    Task taskContent = Task(pathInfo.ascendFePath)
        .AddArg({"-arch", argInfo.socVersion})
        .AddArg(GetCompileOptionContent({"I", "D", "U", "g"}))
        .AddArg(argInfo.optLevel)
        .AddArg("--sub_module=host_pre")
        .AddArg("--module_path=" + hostStubDir)
        .AddArg(preprocessFileName);
    return taskContent;
}

Task TaskDeviceStub(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo, const CoreType coreType)
{
    const std::string coreTypeStr = GetCoreTypeStr(coreType);
    const std::string preprocessFileName = GetPreprocessFileName(pathInfo, file, coreTypeStr);
    const std::string deviceStubDir = GetStubFileDirectory(pathInfo, file, coreTypeStr);

    Task taskContent = Task(pathInfo.ascendFePath)
        .AddArg({"-arch", argInfo.socVersion})
        .AddArg(GetCompileOptionContent({"I", "D", "U", "g"}))
        .AddArg(argInfo.optLevel)
        .AddArg("--sub_module=device_pre_" + coreTypeStr)
        .AddArg("--module_path=" + deviceStubDir)
        .AddArg(preprocessFileName);
    return taskContent;
}

AsccStatus TaskPreprocessLaunch(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo)
{
    TaskExecutor executor;
    InitExecutor(executor, argInfo);

    TaskGroup hostTaskGroup(TaskStep::CODEGEN);
    TaskGroup deviceVecTaskGroup(TaskStep::CODEGEN);
    TaskGroup deviceCubeTaskGroup(TaskStep::CODEGEN);
    // generate preprocess files + stub files
    hostTaskGroup.AddTask(TaskPreprocessHost(file, pathInfo, argInfo));
    hostTaskGroup.AddTask(TaskHostStub(file, pathInfo, argInfo));
    deviceVecTaskGroup.AddTask(TaskPreprocessDevice(file, pathInfo, argInfo, CoreType::SPLIT_VEC));
    deviceVecTaskGroup.AddTask(TaskDeviceStub(file, pathInfo, argInfo, CoreType::SPLIT_VEC));
    deviceCubeTaskGroup.AddTask(TaskPreprocessDevice(file, pathInfo, argInfo, CoreType::SPLIT_CUBE));
    deviceCubeTaskGroup.AddTask(TaskDeviceStub(file, pathInfo, argInfo, CoreType::SPLIT_CUBE));

    executor.AddTaskGroup(hostTaskGroup);
    executor.AddTaskGroup(deviceVecTaskGroup);
    executor.AddTaskGroup(deviceCubeTaskGroup);
    ASCC_CHECK((executor.ExecuteTasks()), {ASCC_LOGE("TaskPreprocessLaunch of file %s failed!", file.c_str());});
    return AsccStatus::SUCCESS;
}

// fileName: file passed in by user.  Returned result not involving directory
// Example: add_custom/add_custom.cpp => add_custom_host_ii.cpp
std::string GetHostStubFileName(const std::string& fileName)
{
    return RemoveSuffix(GetFileName(fileName)) + "_cpp_host_ii.cpp";
}

// device stub files: device_stub_<coreType>.cpp
std::string GetDeviceStubFileName(CodeMode kernelTypeEnum, const CoreType coreType)
{
    if (kernelTypeEnum == CodeMode::KERNEL_TYPE_MIX_AIC_1_1 || kernelTypeEnum == CodeMode::KERNEL_TYPE_MIX_AIC_1_2) {
        if (coreType == CoreType::SPLIT_CUBE) {
            return KERNEL_TYPE_TO_FILE_NAME.at(kernelTypeEnum).at(0);
        } else {
            return KERNEL_TYPE_TO_FILE_NAME.at(kernelTypeEnum).at(1);
        }
    }
    return KERNEL_TYPE_TO_FILE_NAME.at(kernelTypeEnum).at(0);
}

const std::string GetDeviceJsonName(const std::string& fileName, const std::string& coreType)
{
    return RemoveSuffix(GetFileName(fileName)) + "_cpp_" + coreType + "_ii.json";
}

// srcFile: add_custom/add_custom.cpp
// stubFile: device_stub_xx.cpp / add_custom.cpp(host)
const std::string GetObjFileDirectory(const PathInfo& pathInfo, const std::string& srcFile)
{
    return pathInfo.objectFilePath + "/" + GetInputFileStubDirName(srcFile);
}

AsccStatus TaskHostStubCompileO(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo)
{
    TaskExecutor executor;
    InitExecutor(executor, argInfo);

    std::string hostStubDir = GetStubFileDirectory(pathInfo, file, "host");
    std::string hostStubFileName = hostStubDir + "/" + GetHostStubFileName(file);
    std::string kernelCallStubFileName = hostStubDir + "/" + "kernel_call_stub.cpp";

    std::string objFileDir = GetObjFileDirectory(pathInfo, file);
    std::string expectedObjFileName = GetFileName(RemoveSuffix(file)) + ".o";

    uint64_t mixFileSize = GetDeviceFileSize(objFileDir, "mix");
    uint64_t aicFileSize = GetDeviceFileSize(objFileDir, "aic");
    uint64_t aivFileSize = GetDeviceFileSize(objFileDir, "aiv");
    std::vector<std::string> deviceBinsizeInform;
    if (mixFileSize > 0) {
        deviceBinsizeInform.emplace_back("-D__replaced_mix_len=" + std::to_string(mixFileSize));
        deviceBinsizeInform.emplace_back("-D__replaced_mix_file_len=" + std::to_string(mixFileSize));
    }
    if (aicFileSize > 0) {
        deviceBinsizeInform.emplace_back("-D__replaced_aic_len=" + std::to_string(aicFileSize));
        deviceBinsizeInform.emplace_back("-D__replaced_aic_file_len=" + std::to_string(aicFileSize));
    }
    if (aivFileSize > 0) {
        deviceBinsizeInform.emplace_back("-D__replaced_aiv_len=" + std::to_string(aivFileSize));
        deviceBinsizeInform.emplace_back("-D__replaced_aiv_file_len=" + std::to_string(aivFileSize));
    }
    TaskGroup groupCompile(TaskStep::COMPILE);
    Task taskContent = Task(pathInfo.bishengPath).AddArg("-c")
        .AddArg(HOST_COMMON_COMPILE_OPTIONS)
        .AddArg("-D__aicore__=[aicore]")
        .AddArg("-D__host_aicore__=[host,aicore]")
        .AddArg("-undef")
        .AddArg(GetCompileOptionContent({"I", "D", "U", "g"}))
        .AddArg(argInfo.bishengOptLevel)
        .AddArg(deviceBinsizeInform)
        .AddArg({"-include", hostStubFileName})
        .AddArg({"-o", objFileDir + "/" + expectedObjFileName})
        .AddArg(GetDependencyCompileOptionContent(file))
        .AddArg(kernelCallStubFileName);
    groupCompile.AddTask(taskContent);
    executor.AddTaskGroup(groupCompile);

    ASCC_CHECK((executor.ExecuteTasks()), {ASCC_LOGE("TaskHostStubCompileO of file %s failed!", file.c_str());});
    return AsccStatus::SUCCESS;
}

AsccStatus TaskPureHostCompileO(std::vector<std::string>& objFile, const std::string& file, const PathInfo& pathInfo,
    const ArgInfo& argInfo)
{
    TaskExecutor executor;
    InitExecutor(executor, argInfo);

    std::string objFileDir = GetObjFileDirectory(pathInfo, file);
    std::string expectedObjFileName = GetFileName(RemoveSuffix(file)) + ".o";
    std::string hostObjFilePath = objFileDir + "/" + expectedObjFileName;
    objFile.emplace_back(hostObjFilePath);

    TaskGroup groupCompile(TaskStep::COMPILE);
    Task taskContent = Task(pathInfo.bishengPath).AddArg("-c")
        .AddArg("-D__NPU_HOST__")
        .AddArg(HOST_COMMON_COMPILE_OPTIONS)
        .AddArg(GetCompileOptionContent({"I", "D", "U", "g"}))
        .AddArg(GetHostDefaultIncludePath(pathInfo))
        .AddArg(argInfo.bishengOptLevel)
        .AddArg(file)
        .AddArg({"-o", hostObjFilePath})
        .AddArg(GetDependencyCompileOptionContent(file));
    groupCompile.AddTask(taskContent);
    executor.AddTaskGroup(groupCompile);

    ASCC_CHECK((executor.ExecuteTasks()), {ASCC_LOGE("TaskPureHostCompileO of file %s failed!", file.c_str());});
    if (argInfo.outputMode == OutputFileType::FILE_O) {
        if (!argInfo.outputFileName.empty()) {
            expectedObjFileName = argInfo.outputFileName;
        }
        Ascc::ExecuteCommand(("cp " + hostObjFilePath + " " + expectedObjFileName).c_str());
    }
    PrintMsgForExecResult(expectedObjFileName, OutputFileType::FILE_O);
    return AsccStatus::SUCCESS;
}

bool HasJsonInStubDir(const std::string& file, const PathInfo& pathInfo)
{
    const std::string vecStr = GetCoreTypeStr(CoreType::SPLIT_VEC);
    const std::string cubeStr = GetCoreTypeStr(CoreType::SPLIT_CUBE);
    const std::string deviceStubDirVec = GetStubFileDirectory(pathInfo, file, vecStr);
    const std::string jsonNameVec = deviceStubDirVec + "/" + GetDeviceJsonName(file, vecStr);
    const std::string deviceStubDirCube = GetStubFileDirectory(pathInfo, file, cubeStr);
    const std::string jsonNameCube = deviceStubDirCube + "/" + GetDeviceJsonName(file, cubeStr);
    return !(CheckAndGetFullPath(jsonNameVec).empty() && CheckAndGetFullPath(jsonNameCube).empty());
}

AsccStatus TaskDeviceStubCompileO(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo,
    const CoreType coreType)
{
    const std::string coreTypeStr = GetCoreTypeStr(coreType);
    const std::string deviceStubDir = GetStubFileDirectory(pathInfo, file, coreTypeStr);
    const std::string jsonName = deviceStubDir + "/" + GetDeviceJsonName(file, coreTypeStr);
    std::unordered_map<uint8_t, std::vector<std::string>> kernelType;
    std::unordered_map<uint8_t, uint32_t> mixNumLineMap;
    std::unordered_map<uint8_t, uint32_t> dumpUintLineMap;
    std::unordered_map<uint8_t, uint32_t> dumpWorkspaceLineMap;
    std::unordered_map<uint8_t, uint32_t> dumpSizeMap;
    std::unordered_map<uint8_t, std::string> stubFileMap;

    ASCC_CHECK((GetJsonInfo(kernelType, mixNumLineMap, dumpUintLineMap, dumpWorkspaceLineMap, dumpSizeMap, stubFileMap,
        jsonName) == AsccStatus::SUCCESS), {ASCC_LOGE("GetJsonInfo failed!");});
    for (const auto& pair : kernelType) {
        TaskExecutor executor;
        InitExecutor(executor, argInfo);
        CodeMode kernelTypeEnum = static_cast<CodeMode>(pair.first);
        std::vector<std::string> mangleNameVec = pair.second;

        // replace MIX_NUM with 1 for MIX_1_1, otherwise always 2
        std::string stubPreprocessFileName = stubFileMap[pair.first];
        uint32_t mixNumLineNum = mixNumLineMap[pair.first];
        uint32_t dumpUintLineNum = dumpUintLineMap[pair.first];
        uint32_t dumpWorkspaceNumLineNum = dumpWorkspaceLineMap[pair.first];
        uint32_t dumpSize = dumpSizeMap[pair.first];
        ASCC_CHECK((ReplaceStubFile(stubPreprocessFileName, kernelTypeEnum, mixNumLineNum, dumpUintLineNum,
            dumpWorkspaceNumLineNum, dumpSize) == AsccStatus::SUCCESS), {ASCC_LOGE("ReplaceStubFile failed!");});

        // vec or cube only compile kernel types that relate to it
        bool cubeIsCompilingAiv = (coreType == CoreType::SPLIT_CUBE) &&
            (kernelTypeEnum == CodeMode::KERNEL_TYPE_AIV_ONLY || kernelTypeEnum == CodeMode::KERNEL_TYPE_MIX_AIV_1_0);
        bool vecIsCompilingAic = (coreType == CoreType::SPLIT_VEC) &&
            (kernelTypeEnum == CodeMode::KERNEL_TYPE_AIC_ONLY || kernelTypeEnum == CodeMode::KERNEL_TYPE_MIX_AIC_1_0);
        if (cubeIsCompilingAiv || vecIsCompilingAic) {
            continue;
        }

        // device_stub_xxx.cpp
        std::string deviceStubFileName = GetDeviceStubFileName(kernelTypeEnum, coreType);
        std::string stubFileName = deviceStubDir + "/" + deviceStubFileName;
        // device_stub_xxx.o
        std::string stubObjFileDir = GetObjFileDirectory(pathInfo, file);
        std::string stubObjFileName = stubObjFileDir + "/" + RemoveSuffix(GetFileName(stubFileName)) + ".o";

        std::vector<std::string> stubMangle;
        for (auto mangleName : mangleNameVec) {
            std::string stubMacro = GetDeviceStubMacro(coreType, mangleName, kernelTypeEnum);
            if (!stubMacro.empty()) {
                stubMangle.push_back(stubMacro);
            }
        }

        TaskGroup groupCompile(TaskStep::COMPILE);
        Task taskContent = Task(pathInfo.bishengPath).AddArg("-c")
            .AddArg("-undef")
            .AddArg("-D__aicore__=[aicore]")
            .AddArg("-D__host_aicore__=[host,aicore]")
            .AddArg("-DGM_ADDR=__gm__ uint8_t*")
            .AddArg("-D__gm__=__attribute__((cce_global))")
            .AddArg(stubMangle)
            .AddArg(GetCompileOptionContent({"I", "D", "U", "g", "sanitizer"}))
            .AddArg(GetDeviceDefaultCompileOption(argInfo))
            .AddArg("--cce-aicore-arch=dav-c220-" + coreTypeStr)
            .AddArg(GetDeviceDefaultIncludePath(pathInfo, file))
            .AddArg({"-o", stubObjFileName})
            .AddArg(stubFileName);

        if (coreType == CoreType::SPLIT_VEC) {
            taskContent.AddArg("--cce-auto-sync").AddArg("-D__DAV_C220_VEC__");
        } else {
            taskContent.AddArg("-D__DAV_C220_CUBE__");
        }
        groupCompile.AddTask(taskContent);
        executor.AddTaskGroup(groupCompile);

        ASCC_CHECK((executor.ExecuteTasks()), {ASCC_LOGE("TaskDeviceStubCompileO of file %s with coreType %s failed!",
            file.c_str(), coreTypeStr.c_str());});
    }
    return AsccStatus::SUCCESS;
}

void UpdateSanitizerProcess(Task& task, const PathInfo& pathInfo)
{
    std::vector<std::string> status;
    if (GetOptionValuesByArgName("sanitizer", status)) {
        task.AddArg({"--dependent-libraries",
            pathInfo.cannPath + "/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-vec.a",
            pathInfo.cannPath + "/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220-cube.a",
            pathInfo.cannPath + "/tools/mssanitizer/lib64/libsanitizer_stub_dav-c220.a"
        });
    }
}

void AddLdLldTask(TaskGroup& taskGroup, const PathInfo& pathInfo, const std::vector<std::string> files,
    const std::string& objFilePath, const std::string& coreType)
{
    std::string objFileName = objFilePath + "/device_" + coreType + ".o";
    Task taskContent = Task(pathInfo.ldlldPath)
        .AddArg(LDLLD_PROCESS_ARG)
        .AddArg(files)
        .AddArg("-static")
        .AddArg({"-o", objFileName});
    taskGroup.AddTask(taskContent);

    Task taskMergeFinal = Task(pathInfo.ldlldPath)
        .AddArg(LDLLD_FINAL_PROCESS_ARG)
        .AddArg(objFileName)
        .AddArg("-static")
        .AddArg({"-o", objFileName});
    UpdateSanitizerProcess(taskMergeFinal, pathInfo);
    taskGroup.AddTask(taskMergeFinal);
}

AsccStatus TaskMergeDeviceObjFiles(const std::string& file, const PathInfo& pathInfo, const ArgInfo& argInfo)
{
    TaskExecutor executor;
    InitExecutor(executor, argInfo);

    TaskGroup mergeTaskGroup(TaskStep::DEVICE_LINK);

    std::string objFilePath = GetObjFileDirectory(pathInfo, file);
    // mix files: must pack with aic first, then aiv.   Filename related to KERNEL_TYPE_TO_FILE_NAME
    std::vector<std::string> potentialMixFiles = {"device_stub_mix_1_1_aic.o", "device_stub_mix_aic.o",
        "device_stub_mix_1_1_aiv.o", "device_stub_mix_aiv.o"};

    std::vector<std::string> mixFiles;
    for (auto mixFile : potentialMixFiles) {
        std::string mixFilePath = objFilePath + "/" + mixFile;
        if (PathCheck(mixFilePath.c_str(), false) != Ascc::PathStatus::NOT_EXIST) {
            mixFiles.emplace_back(mixFilePath);
        }
    }

    if (mixFiles.size() > 0) {
        AddLdLldTask(mergeTaskGroup, pathInfo, mixFiles, objFilePath, "mix");
    }

    std::string aivFilePath = objFilePath + "/device_stub_aiv.o";
    if (PathCheck(aivFilePath.c_str(), false) != Ascc::PathStatus::NOT_EXIST) {
        AddLdLldTask(mergeTaskGroup, pathInfo, {aivFilePath}, objFilePath, "aiv");
    }

    std::string aicFilePath = objFilePath + "/device_stub_aic.o";
    if (PathCheck(aicFilePath.c_str(), false) != Ascc::PathStatus::NOT_EXIST) {
        AddLdLldTask(mergeTaskGroup, pathInfo, {aicFilePath}, objFilePath, "aic");
    }

    executor.AddTaskGroup(mergeTaskGroup);
    ASCC_CHECK((executor.ExecuteTasks()), {ASCC_LOGE("TaskMergeDeviceObjFiles of file %s failed!", file.c_str());});
    return AsccStatus::SUCCESS;
}

AsccStatus TaskPackKernel(std::vector<std::string>& objFile, const std::string& file, const PathInfo& pathInfo,
    const ArgInfo& argInfo)
{
    TaskExecutor executor;
    InitExecutor(executor, argInfo);

    std::vector<std::string> kernelType = {"mix", "aic", "aiv"};  // device_mix.o, device_aic.o, device_aiv.o
    std::string objFileDir = GetObjFileDirectory(pathInfo, file);
    std::string hostObjFilePath = objFileDir + "/" + RemoveSuffix(GetFileName(file)) + ".o";
    objFile.emplace_back(hostObjFilePath);

    TaskGroup groupPack(TaskStep::PACK_KERNEL);
    for (auto fileType : kernelType) {
        uint8_t coreTypeValue = ASCC_CORE_TYPE_MIX;
        if (fileType == "aic") {
            coreTypeValue = ASCC_CORE_TYPE_AIC;
        } else if (fileType == "aiv") {
            coreTypeValue = ASCC_CORE_TYPE_AIV;
        }

        std::string deviceObjFilePath = objFileDir + "/device_" + fileType + ".o";
        if (PathCheck(deviceObjFilePath.c_str(), false) != Ascc::PathStatus::NOT_EXIST) {
            groupPack.AddTask(Task(pathInfo.ascPackKernelPath)
                .AddArg(hostObjFilePath)
                .AddArg(deviceObjFilePath)
                .AddArg(std::to_string(coreTypeValue))
                .AddArg(hostObjFilePath)
            );
        }
    }

    executor.AddTaskGroup(groupPack);
    ASCC_CHECK((executor.ExecuteTasks()), {ASCC_LOGE("TaskPackKernel of file %s failed!", file.c_str());});

    std::string expectedObjFileName = GetFileName(RemoveSuffix(file) + ".o");
    if (argInfo.outputMode == OutputFileType::FILE_O) {
        if (!argInfo.outputFileName.empty()) {
            expectedObjFileName = argInfo.outputFileName;
        }
        Ascc::ExecuteCommand(("cp " + hostObjFilePath + " " + expectedObjFileName).c_str());
    }
    PrintMsgForExecResult(expectedObjFileName, OutputFileType::FILE_O);
    return AsccStatus::SUCCESS;
}

AsccStatus TaskSharedLibrary(const std::vector<std::string>& objFile, const PathInfo& pathInfo, const ArgInfo& argInfo)
{
    TaskExecutor executor;
    InitExecutor(executor, argInfo);

    std::vector<std::string> outFileStatus = {"-o", std::string(DEFAULT_SHARED_LIB_NAME)};
    if (!argInfo.outputFileName.empty()) {
        outFileStatus = {"-o", argInfo.outputFileName};
    }

    TaskGroup groupLink(TaskStep::LINK);
    groupLink.AddTask(Task(pathInfo.gppPath).AddArg("-shared")
        .AddArg(objFile)
        .AddArg(GetLinkCompileOptionContent())         // users link content first
        .AddArg(pathInfo.linkPath).AddArg(LINK_LIB)
        .AddArg(outFileStatus).AddArg(argInfo.optLevel)
    );
    executor.AddTaskGroup(groupLink);

    ASCC_CHECK((executor.ExecuteTasks()), {ASCC_LOGE("TaskSharedLibrary failed.");});
    PrintMsgForExecResult(outFileStatus[1], OutputFileType::FILE_SO);
    return AsccStatus::SUCCESS;
}

AsccStatus TaskExecutable(const std::vector<std::string>& objFile, const PathInfo& pathInfo, const ArgInfo& argInfo)
{
    TaskExecutor executor;
    InitExecutor(executor, argInfo);

    std::vector<std::string> outFileStatus = {"-o", std::string(DEFAULT_EXEC_NAME)};
    if (!argInfo.outputFileName.empty()) {
        outFileStatus = {"-o", argInfo.outputFileName};
    }

    TaskGroup groupLink(TaskStep::LINK);
    groupLink.AddTask(Task(pathInfo.gppPath)
        .AddArg(objFile)
        .AddArg(GetLinkCompileOptionContent())           // users link content first
        .AddArg(pathInfo.linkPath).AddArg(LINK_LIB)
        .AddArg(outFileStatus).AddArg(argInfo.optLevel)
    );
    executor.AddTaskGroup(groupLink);

    ASCC_CHECK((executor.ExecuteTasks()), {ASCC_LOGE("TaskExecutable failed.");});
    PrintMsgForExecResult(outFileStatus[1], OutputFileType::FILE_EXECUTABLE);
    return AsccStatus::SUCCESS;
}

AsccStatus ProcessFiles(const std::vector<std::string>& files, const PathInfo& pathInfo, const ArgInfo& argInfo)
{
    std::vector<std::string> objFile;  // .o file storage
    // compile multiple .cpp files to multiple.o
    for (const auto &file : files) {
        // preprocess host + device
        ASCC_CHECK((TaskPreprocessLaunch(file, pathInfo, argInfo) == AsccStatus::SUCCESS),
            {ErrorMsgForProcessFile(file);});

        // has <<<>>>> and host
        if (HasJsonInStubDir(file, pathInfo)) {
            // compile device stub files
            ASCC_CHECK((TaskDeviceStubCompileO(file, pathInfo, argInfo, CoreType::SPLIT_VEC) == AsccStatus::SUCCESS),
                {ErrorMsgForProcessFile(file);});
            ASCC_CHECK((TaskDeviceStubCompileO(file, pathInfo, argInfo, CoreType::SPLIT_CUBE) == AsccStatus::SUCCESS),
                {ErrorMsgForProcessFile(file);});
            // link mix.o into one .o
            ASCC_CHECK((TaskMergeDeviceObjFiles(file, pathInfo, argInfo) == AsccStatus::SUCCESS),
                {ErrorMsgForProcessFile(file);});

            // compile host stub files
            ASCC_CHECK((TaskHostStubCompileO(file, pathInfo, argInfo) == AsccStatus::SUCCESS),
                {ErrorMsgForProcessFile(file);});

            // generate .o by pack device.o into host.o
            ASCC_CHECK((TaskPackKernel(objFile, file, pathInfo, argInfo) == AsccStatus::SUCCESS),
                {ErrorMsgForProcessFile(file);});
        } else {   // pure host
            ASCC_CHECK((TaskPureHostCompileO(objFile, file, pathInfo, argInfo) == AsccStatus::SUCCESS),
                {ErrorMsgForProcessFile(file);});
        }
    }

    // compile multiple .o into one .so / executable
    if (argInfo.outputMode == OutputFileType::FILE_SO) {
        ASCC_CHECK((TaskSharedLibrary(objFile, pathInfo, argInfo) == AsccStatus::SUCCESS), {});
    } else if (argInfo.outputMode == OutputFileType::FILE_EXECUTABLE){
        ASCC_CHECK((TaskExecutable(objFile, pathInfo, argInfo) == AsccStatus::SUCCESS), {});
    }
    return AsccStatus::SUCCESS;
}

} // namespace Ascc
