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
 * \file ascc_argument_manager.cpp
 * \brief
 */
#include "ascc_argument_manager.h"

#include <unordered_set>
#include <csignal>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <set>
#include <string>

#include "ascc_tmp_file_manager.h"
#include "ascc_compile_base.h"
#include "ascc_dump_flags.h"
#include "ascc_utils.h"

namespace Ascc {
static constexpr uint8_t HELP_INFO_SHIFT = 30;  // shift 30 to start description
static constexpr bool IS_CONTINUE = true;
static constexpr bool NOT_CONTINUE = false;
static constexpr bool IS_SUPPORT_EQUAL = true;
static constexpr bool NOT_SUPPORT_EQUAL = false;

// register options parser
AsccArgumentManager::AsccArgumentManager() {
    OptionRegister<AsccParserValue>("-arch", "Set Soc Version, sample -> -arch <soc>", NOT_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("--npu-architecture", "Set Soc Version, sample -> --npu-architecture <soc>",
        NOT_CONTINUE, IS_SUPPORT_EQUAL);

    OptionRegister<AsccParserValue>("-o", "Set output file, sample -> -o <file>", NOT_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("--output-file", "Set output file, sample -> --output-file <file>", NOT_CONTINUE,
        IS_SUPPORT_EQUAL);

    OptionRegister<AsccParserValue>("-I", "Set include Path, sample -> -I<path>", IS_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("--include-path", "Set include Path, sample -> --include-path <path>", NOT_CONTINUE,
        IS_SUPPORT_EQUAL);

    OptionRegister<AsccParserValue>("-l", "Set library, sample -> -l<lib>", IS_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("--library", "Set library, sample -> --library <lib>", NOT_CONTINUE,
        IS_SUPPORT_EQUAL);

    OptionRegister<AsccParserValue>("-L", "Set library Path, sample -> -L<libpath>", IS_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("--library-path", "Set library Path, sample -> --library-path <libpath>",
        NOT_CONTINUE, IS_SUPPORT_EQUAL);

    OptionRegister<AsccParserValue>("--save-temps", "Save Tmp File, sample -> --save-temps/--save-temps=<tmpPath>",
        NOT_CONTINUE, IS_SUPPORT_EQUAL);

    OptionRegister<AsccParserBool>("-h", "Display this information, sample -> -h", NOT_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserBool>("--help", "Display this information, sample -> --help", NOT_CONTINUE,
        NOT_SUPPORT_EQUAL);

    OptionRegister<AsccParserBool>("-shared", "Generate .so file. sample -> -shared -o add.so then generate add.so",
        NOT_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserBool>("-c", "Compile and assemble, but do not link. sample -> -c -o add.o", NOT_CONTINUE,
        NOT_SUPPORT_EQUAL);

    OptionRegister<AsccParserValue>("-D", "Define macros. sample -> -DDEBUG", IS_CONTINUE, NOT_SUPPORT_EQUAL);

    OptionRegister<AsccParserBool>("-g", "Enable debug mode. sample -> -g", NOT_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserBool>("--debug", "Enable debug mode. sample -> --debug", NOT_CONTINUE, NOT_SUPPORT_EQUAL);

    OptionRegister<AsccParserBool>("--sanitizer", "Enable sanitizer. sample -> --sanitizer", NOT_CONTINUE,
        NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserBool>("-sanitizer", "Enable sanitizer. sample -> -sanitizer", NOT_CONTINUE,
        NOT_SUPPORT_EQUAL);

    OptionRegister<AsccParserValue>("-O", "Set optimize level, sample -> -O<level>", IS_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("--optimize", "Set optimize level, sample -> --optimize <level>", NOT_CONTINUE,
        IS_SUPPORT_EQUAL);

    OptionRegister<AsccParserBool>("-fPIC", "Support -fPIC. sample -> -fPIC", NOT_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("-Wl", "Support -Wl. sample -> -Wl,-soname,libfoo.so.1", IS_CONTINUE,
        NOT_SUPPORT_EQUAL);

    OptionRegister<AsccParserBool>("-MD", "Generate a dependency file and compile the input file. sample -> -MD",
        NOT_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserBool>("-MMD", "Same as -MD, but skip header files found in system directories. "
        "sample -> -MMD", NOT_CONTINUE, NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserBool>("-MP", "Add an empty target for each dependency. sample -> -MP", NOT_CONTINUE,
        NOT_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("-MT", "Specify the target name of the generated rule when generating a dependency"
        "file. sample -> -MT <target name>", NOT_CONTINUE, IS_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("-MF", "Specify the output file for the dependency file generated with "
        "-M/-MM/-MD/-MMD. sample -> -MF <filename>", NOT_CONTINUE, IS_SUPPORT_EQUAL);

    // for preprocess
    OptionRegister<AsccParserValue>("--sub_module", "Specify the task module. sample -> --sub_module=host_pre",
        NOT_CONTINUE, IS_SUPPORT_EQUAL);
    OptionRegister<AsccParserValue>("--module_path", "Specify the path to store preprocess result. "
        "sample -> --module_path=<path>", NOT_CONTINUE, IS_SUPPORT_EQUAL);
}

bool AsccArgumentManager::IsArgValid(const std::string &arg) const
{
    for (const auto &[name, parser] : parserQueue_) {
        if (name.size() > arg.size()) {
            continue;
        }
        if (!(parser->IsContinue()) && arg == name) {
            return true;
        }
        if (!(parser->IsSupportEqual()) && arg == name) {
            return true;
        }
        const std::string &opt = arg.substr(0, name.size());
        if ((parser->IsContinue()) && opt == name) {
            if (opt == "-O") {
                return (arg == "-O0" || arg == "-O1" || arg == "-O2" || arg == "-O3");
            }
            return true;
        }
        if (parser->IsSupportEqual()) {
            size_t pos = arg.find('=');
            if (pos != std::string::npos && arg.substr(0, pos) == name) {
                return true;
            }
        }
    }
    return false;
}

AsccStatus AsccArgumentManager::ArgsCheck(const std::vector<std::string> &args)
{
    for (auto& arg : args) {
        if (arg == "--help" || arg == "-h") {
            helpRequested_ = true;
            continue;
        }
        if (arg == "--debug" || arg == "-g") {
            debugRequested_ = true;
            continue;
        }
        if (arg == "--sanitizer" || arg == "-sanitizer") {
            sanitizerRequested_ = true;
            continue;
        }
        if (arg == "-O0" || arg == "-O1" || arg == "-O2" || arg == "-O3") {
            optimizeLevel_ = arg.substr(1);
            continue;
        }

        if (parserQueue_.find(arg) != parserQueue_.end()) {
            continue;
        }
        if ((*arg.begin()) == '-' && (!IsArgValid(arg))) {
            Ascc::HandleError(std::string("Unrecognized command line option [" + arg + "]!"));
            return AsccStatus::FAILURE;
        }
    }
    return AsccStatus::SUCCESS;
}
AsccStatus AsccArgumentManager::InfoCollection(const std::vector<std::string> &args)
{
    std::unordered_set<uint32_t> parsedIndex;
    for (auto &[name, parser] : parserQueue_) { // option data collection
        ASCC_CHECK((parser->Parse(args, parsedIndex) == AsccStatus::SUCCESS),
            {HandleError(std::string("Option [" + name + "] parse failed!"));});
    }
    if (args.size() - parsedIndex.size() > 1) { // multiple input check
        Ascc::HandleError(std::string("Input files exceed the maximum limit! Maximum is [1]."));
        return AsccStatus::FAILURE;
    }
    for (size_t i = 0; i < args.size(); ++i) { // input path check
        if (parsedIndex.find(i) != parsedIndex.end()) {
            continue;
        }
        inputFile_ = Ascc::CheckAndGetFullPath(args[i]);
        if (inputFile_.empty()) {
            Ascc::HandleError(std::string("Input Path " + args[i] + " not exist!"));
            return AsccStatus::FAILURE;
        }
        std::string inputSuffix = Ascc::GetSuffix(inputFile_);
        if (!(inputSuffix == ".C" || inputSuffix == ".c" || inputSuffix == ".cc" || inputSuffix == ".cxx" ||
            inputSuffix == ".cpp" || inputSuffix == ".asc")) {
            Ascc::HandleError("Input file suffix " + inputSuffix + " is not supported, only support following suffix: "
                ".C / .c / .cc / .cxx / .cpp / .asc.");
            return AsccStatus::FAILURE;
        }
        ASC_LOG_ASC_INFO(OPTION, "Input file : [%s].", inputFile_.c_str());
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccArgumentManager::SaveTempsHandle(const std::string& tmpPath) const
{
    Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    envVar.asccTmpPath = Ascc::CheckAndGetFullPath(tmpPath);
    if (envVar.asccTmpPath.empty()) {
        Ascc::HandleError(std::string("Tmp path ") + tmpPath + " not exist!");
        return AsccStatus::FAILURE;
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccArgumentManager::OptimizeLevelHandle(const std::string& opLevel)
{
    if (!(opLevel == "0" || opLevel == "1" || opLevel == "2" || opLevel == "3")) {
        Ascc::HandleError("Invalid optimize level [" + opLevel + "], only support level 0, 1, 2, 3.");
        return AsccStatus::FAILURE;
    }
    optimizeLevel_ = "O" + opLevel;
    return AsccStatus::SUCCESS;
}

// -MD -MT -MF -MMD -MP -fPIC -Wl
void AsccArgumentManager::DependencyHandle(const std::string& argName, const std::string& argValue)
{
    dependencyOptions_ += " " + argName;
    if (argName == "-MD" || argName == "-MMD") {
        mfmtRequested_ = true;       // need to update mfmtRequested to set -MF -MT out of tmp directory
    } else if (argName == "-MT") {
        mtFile_ = argValue;
        dependencyOptions_ += " " + argValue;
    } else if (argName == "-MF") {
        mfFile_ = argValue;
        dependencyOptions_ += " " + argValue;
    } else if (argName == "-Wl") {
        dependencyOptions_ += argValue;
    }
}

AsccStatus AsccArgumentManager::PreprocessTaskHandle(const std::string& argValue)
{
    if (argValue == "host_pre") {
        preTaskType_ = PreTaskType::HOST;
    } else if (argValue == "device_pre_vec") {
        preTaskType_ = PreTaskType::DEVICE_AIV;
    } else if (argValue == "device_pre_cube") {
        preTaskType_ = PreTaskType::DEVICE_AIC;
    } else {
        Ascc::HandleError(std::string("Sub module only support [host_pre, device_pre_vec, device_pre_cube], "
            "but get value " + argValue));
        return AsccStatus::FAILURE;
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccArgumentManager::ModulePathHandle(const std::string& argValue)
{
    std::string pathRes = Ascc::CheckAndGetFullPath(argValue);
    ASCC_CHECK((!pathRes.empty()), {Ascc::HandleError("Module path " + argValue + " not exist!");});
    modulePath_ = pathRes;
    return AsccStatus::SUCCESS;
}

void AsccArgumentManager::SharedHandle(const std::string& name)
{
    if (compileMode_ == Ascc::OutputFileType::FILE_O) {
        return;
    }
    compileMode_ = (name == "-shared" ? Ascc::OutputFileType::FILE_SO : Ascc::OutputFileType::FILE_O);
}

AsccStatus AsccArgumentManager::ArchHandle(const std::string& archName)
{
    if (supportSoc_.find(archName) == supportSoc_.end()) {
        Ascc::HandleError(std::string("Do not support npu architecture [" + archName + "]!"));
        return AsccStatus::FAILURE;
    }

    npuArch_ = socToCore_.at(archName);
    if (archName == "Ascend910B4-1") {
        npuArchStr_ = "Ascend910B4";      // B4-1 reads the same lib as Ascend910B4
    } else {
        npuArchStr_ = archName;
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccArgumentManager::OutputHandle(const std::string& outputPath)
{
    const std::string& outName = Ascc::GetFileName(outputPath);
    std::string outPath = Ascc::GetFilePath(outputPath);
    if (outPath.empty()) {
        Ascc::HandleError(std::string("Output Path directory ") + outputPath + " not exist!");
        return AsccStatus::FAILURE;
    }
    outputFile_ = outPath + "/" + outName;
    return AsccStatus::SUCCESS;
}

AsccStatus AsccArgumentManager::GetArgument()
{
    ASC_LOG_ASC_DEBUG(OPTION, "Get argument start!");
    Ascc::AsccGlobalEnvManager &envVar = Ascc::AsccGlobalEnvManager::GetInstance();
    std::unordered_map<std::string, std::vector<std::string> *> argsPtrMap = {{"-I", &(inputArgs_.incPaths)},
        {"--include-path", &(inputArgs_.incPaths)},
        {"-l", &(inputArgs_.linkFiles)},
        {"--library", &(inputArgs_.linkFiles)},
        {"-L", &(inputArgs_.linkPath)},
        {"--library-path", &(inputArgs_.linkPath)},
        {"-D", &(inputArgs_.definitions)}};
    for (const auto &[name, parser] : parserQueue_) {
        if (!parser->IsValid()) {
            continue;
        }
        if (name == "--save-temps") {
            envVar.needSaveTmpFile = true;
            ASC_LOG_ASC_DEBUG(OPTION, "[%s] valid.", name.c_str());
            if (parser->GetValue().empty()) {
                continue;
            }
            if (SaveTempsHandle(parser->GetValue().back()) == AsccStatus::FAILURE) {
                Ascc::HandleError("--save-temps path not exist!");
                return AsccStatus::FAILURE;
            }
            continue;
        }
        if (name == "--optimize") {
            if (OptimizeLevelHandle(parser->GetValue().back()) == AsccStatus::FAILURE) {
                return AsccStatus::FAILURE;
            }
            ASC_LOG_ASC_DEBUG(OPTION, "[%s] valid.", name.c_str());
            continue;
        }
        if (name == "--sub_module") {
            ASCC_CHECK((PreprocessTaskHandle(parser->GetValue().back()) == AsccStatus::SUCCESS), {});
            ASC_LOG_ASC_DEBUG(OPTION, "[%s] valid.", name.c_str());
            continue;
        }
        if (name == "--module_path") {
            ASCC_CHECK((ModulePathHandle(parser->GetValue().back()) == AsccStatus::SUCCESS), {});
            ASC_LOG_ASC_DEBUG(OPTION, "[%s] valid.", name.c_str());
            continue;
        }
        if (name == "-shared" || name == "-c") {
            SharedHandle(name);
            ASC_LOG_ASC_DEBUG(OPTION, "[%s] valid.", name.c_str());
            continue;
        }

        if (name == "-MD" || name == "-MMD" || name == "-MP" || name == "-fPIC") {
            DependencyHandle(name, "");
            continue;
        }
        if (name == "-MT" || name == "-MF" || name == "-Wl") {
            DependencyHandle(name, parser->GetValue().back());
            continue;
        }
        if (name == "-arch" || name == "--npu-architecture") {
            if (ArchHandle(parser->GetValue().back()) == AsccStatus::FAILURE) {
                Ascc::HandleError("Npu architecture not exist!");
                return AsccStatus::FAILURE;
            }
            ASC_LOG_ASC_DEBUG(OPTION, "[%s] valid.", name.c_str());
            continue;
        }
        if (name == "-o" || name == "--output-file") {
            if (OutputHandle(parser->GetValue().back()) == AsccStatus::FAILURE) {
                Ascc::HandleError("Output path not exist!");
                return AsccStatus::FAILURE;
            }
            ASC_LOG_ASC_DEBUG(OPTION, "[%s] valid.", name.c_str());
            continue;
        }
        if (argsPtrMap.find(name) == argsPtrMap.end()) {
            continue;
        }
        auto argsPtr = argsPtrMap[name];
        const auto &parserValue = parser->GetValue();
        argsPtr->insert(argsPtr->end(), parserValue.begin(), parserValue.end());
    }
    return AsccStatus::SUCCESS;
}

AsccStatus AsccArgumentManager::ArgumentParse(std::vector<std::string> &args)
{
    if (ArgsCheck(args) == AsccStatus::FAILURE) {
        ASC_LOG_ASC_ERROR(OPTION, "Unrecognized command line option!");
        return AsccStatus::FAILURE;
    }
    if (InfoCollection(args) == AsccStatus::FAILURE) {
        ASC_LOG_ASC_ERROR(OPTION, "Option parse failed!");
        return AsccStatus::FAILURE;
    }
    if (helpRequested_) {
        return AsccStatus::SUCCESS;
    }
    if (inputFile_.empty()) {
        Ascc::HandleError("Have no input file!");
        return AsccStatus::FAILURE;
    }
    if ((!parserQueue_["-arch"]->IsValid()) && (!parserQueue_["--npu-architecture"]->IsValid())) {
        Ascc::HandleError("Have not set npu architecture!");
        return AsccStatus::FAILURE;
    }
    if (GetArgument() == AsccStatus::FAILURE) {
        Ascc::HandleError("Argument get failed!");
        return AsccStatus::FAILURE;
    }
    auto &dumpFlags = Ascc::AsccDumpFlags::GetInstance();
    for (auto it : inputArgs_.definitions) {
        if (it == "ASCENDC_DUMP=0") {
            dumpFlags.SetIsDumpCloseManual();
            break;
        }
    }
    return AsccStatus::SUCCESS;
}

void AsccArgumentManager::PrintManual() const
{
    std::cout << "Usage: compiler [options] <input files>\n";
    std::cout << "Options:\n";
    for (const auto &pair : parserQueue_) {
        const auto &parser = pair.second;
        auto name = pair.first;
        auto description = parser->GetDescription();
        std::cout << "  " << name << std::setw(HELP_INFO_SHIFT - name.length()) << "\t" << description << std::endl;
    }
}

const bool& AsccArgumentManager::ManualRequested() const
{
    return helpRequested_;
}
const bool& AsccArgumentManager::DebugRequested() const
{
    return debugRequested_;
}
const bool& AsccArgumentManager::SanitizerRequested() const
{
    return sanitizerRequested_;
}
const bool& AsccArgumentManager::MFMTRequested() const
{
    return mfmtRequested_;
}

const PreTaskType& AsccArgumentManager::GetPreTaskType() const
{
    return preTaskType_;
}

const std::string& AsccArgumentManager::GetModulePath() const
{
    return modulePath_;
}

const std::string& AsccArgumentManager::GetInputFile() const
{
    return inputFile_;
}
const std::string& AsccArgumentManager::GetOutputFile() const
{
    return outputFile_;
}

const std::string& AsccArgumentManager::GetOptimizeLevel() const
{
    return optimizeLevel_;
}

const std::string& AsccArgumentManager::GetNpuArchStr() const
{
    return npuArchStr_;
}

const std::string& AsccArgumentManager::GetDependencyOption() const
{
    return dependencyOptions_;
}

const std::string& AsccArgumentManager::GetMFFileName() const
{
    return mfFile_;
}

const std::string& AsccArgumentManager::GetMTFileName() const
{
    return mtFile_;
}

const Ascc::ShortSoCVersion& AsccArgumentManager::GetNpuArch() const
{
    return npuArch_;
}
const Ascc::OutputFileType& AsccArgumentManager::GetCompileMode() const
{
    return compileMode_;
}
const CompileArgs& AsccArgumentManager::GetCompileOptions() const
{
    return inputArgs_;
}
}  // namespace Ascc