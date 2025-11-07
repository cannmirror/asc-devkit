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
 * \file ascc_argument_manager.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_ARGUMENT_MANAGER_H__
#define __INCLUDE_ASCC_ARGUMENT_MANAGER_H__
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>

#include "ascc_types.h"
#include "ascc_parser_value.h"
#include "ascc_parser_bool.h"

namespace Ascc {
class AsccArgumentManager {
public:
    inline static AsccArgumentManager& GetInstance()
    {
        static AsccArgumentManager instance;
        return instance;
    }
    AsccStatus ArgumentParse(std::vector<std::string>& args);
    const std::string& GetInputFile() const;
    const std::string& GetOutputFile() const;
    const std::string& GetNpuArchStr() const;
    const std::string& GetOptimizeLevel() const;
    const std::string& GetDependencyOption() const;
    const std::string& GetModulePath() const;
    const std::string& GetMFFileName() const;
    const std::string& GetMTFileName() const;
    const Ascc::ShortSoCVersion& GetNpuArch() const;
    const Ascc::OutputFileType& GetCompileMode() const;
    const CompileArgs& GetCompileOptions() const;
    const PreTaskType& GetPreTaskType() const;
    const bool& ManualRequested() const;
    const bool& DebugRequested() const;
    const bool& SanitizerRequested() const;
    const bool& MFMTRequested() const;
    void PrintManual() const;

private:
    AsccArgumentManager();
    ~AsccArgumentManager() = default;
    AsccArgumentManager(const AsccArgumentManager&) = delete;
    AsccArgumentManager& operator=(const AsccArgumentManager&) = delete;
    AsccArgumentManager(AsccArgumentManager&&) = delete;
    AsccArgumentManager& operator=(AsccArgumentManager&&) = delete;

    template <typename T>
    void OptionRegister(const std::string &name, const std::string &description,
        const bool &isContinue, const bool &isSupportEqual)
    {
        auto parser = std::make_unique<T>();
        parser->InitParser(name, description, isContinue, isSupportEqual);
        parserQueue_[name] = std::move(parser);
    }
    bool IsArgValid(const std::string& arg) const;
    void SharedHandle(const std::string& name);
    void DependencyHandle(const std::string& argName, const std::string& argValue);
    AsccStatus PreprocessTaskHandle(const std::string& argValue);
    AsccStatus ModulePathHandle(const std::string& argValue);
    AsccStatus ArgsCheck(const std::vector<std::string> &args);
    AsccStatus InfoCollection(const std::vector<std::string> &args);
    AsccStatus SaveTempsHandle(const std::string& tmpPath) const;
    AsccStatus OptimizeLevelHandle(const std::string& opLevel);
    AsccStatus ArchHandle(const std::string& archName);
    AsccStatus OutputHandle(const std::string& outputPath);
    AsccStatus GetArgument();

    PreTaskType preTaskType_ = PreTaskType::NONE;
    std::string inputFile_ = std::string();
    std::string outputFile_ = std::string();
    std::string npuArchStr_ = "Ascend910B1";
    std::string optimizeLevel_ = "O3";
    std::string dependencyOptions_;
    std::string modulePath_;                     // for storing sub_module stub files
    std::string mfFile_;                         // for option -MF
    std::string mtFile_;                         // for option -MT
    bool mfmtRequested_ = false;                 // whether need -MF + -MT for generate dependency file
    bool helpRequested_ = false;
    bool debugRequested_ = false;
    bool sanitizerRequested_ = false;

    CompileArgs inputArgs_;
    Ascc::ShortSoCVersion npuArch_ = Ascc::ShortSoCVersion::ASCEND910B;
    Ascc::OutputFileType compileMode_ = Ascc::OutputFileType::FILE_EXECUTABLE;

    std::unordered_map<std::string, std::unique_ptr<AsccParserBase>> parserQueue_;

    const std::unordered_set<std::string> supportSoc_ = {
        "Ascend910B1",
        "Ascend910B2",
        "Ascend910B2C",
        "Ascend910B3",
        "Ascend910B4",
        "Ascend910B4-1",
        "Ascend910_9391",
        "Ascend910_9381",
        "Ascend910_9372",
        "Ascend910_9392",
        "Ascend910_9382",
        "Ascend910_9362"
    };
    const std::unordered_map<std::string, Ascc::ShortSoCVersion> socToCore_ = {
        {"Ascend910B1", Ascc::ShortSoCVersion::ASCEND910B}, // ascend910b_list
        {"Ascend910B2", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910B2C", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910B3", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910B4", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910B4-1", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910_9391", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910_9381", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910_9372", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910_9392", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910_9382", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910_9362", Ascc::ShortSoCVersion::ASCEND910B},
        {"Ascend910A", Ascc::ShortSoCVersion::ASCEND910}, // ascend910_list
        {"Ascend910ProA", Ascc::ShortSoCVersion::ASCEND910},
        {"Ascend910B", Ascc::ShortSoCVersion::ASCEND910},
        {"Ascend910ProB", Ascc::ShortSoCVersion::ASCEND910},
        {"Ascend910PremiumA", Ascc::ShortSoCVersion::ASCEND910},
        {"Ascend310P1", Ascc::ShortSoCVersion::ASCEND310P}, // ascend310p_list
        {"Ascend310P3", Ascc::ShortSoCVersion::ASCEND310P},
        {"Ascend310B1", Ascc::ShortSoCVersion::ASCEND310B}, // ascend310b_list
        {"Ascend310B2", Ascc::ShortSoCVersion::ASCEND310B},
        {"Ascend310B3", Ascc::ShortSoCVersion::ASCEND310B},
        {"Ascend310B4", Ascc::ShortSoCVersion::ASCEND310B}
    };
};
} // Ascc
#endif // __INCLUDE_ASCC_ARGUMENT_MANAGER_H__