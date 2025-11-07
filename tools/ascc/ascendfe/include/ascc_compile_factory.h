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
 * \file ascc_compile_factory.h
 * \brief
 */
#ifndef __INCLUDE_ASCC_COMPILE_FACTORY_H__
#define __INCLUDE_ASCC_COMPILE_FACTORY_H__
#include <unordered_map>
#include <map>
#include <string>
#include <vector>

#include "ascc_log.h"
#include "ascc_types.h"
#include "ascc_compile_v220.h"
#include "ascc_compile_host.h"

// path // lib // option // define
namespace Ascc {

class AsccCompileFactory {
public:
    inline static AsccCompileFactory& GetInstance()
    {
        static AsccCompileFactory instance;
        return instance;
    }
    template <typename T>
    void Register(const std::string& file, const CompileArgs& args)
    {
        compileTaskList_.emplace(file, std::make_shared<T>(args));
    }
    AsccStatus Produce()
    {
        for (const auto& compileTask : compileTaskList_) {
            const auto& file = compileTask.first;
            const auto& task = compileTask.second;
            ASC_LOG_ASC_INFO(COMPILE, "Produce Compile Task File name [%s].", file.c_str());
            ASCC_CHECK((task->Compile() == AsccStatus::SUCCESS), {ASC_LOG_ASC_ERROR(COMPILE, "Compile Fail!");});
        }
        return AsccStatus::SUCCESS;
    }
    template <typename T>
    std::shared_ptr<T> GetCompileTask(const std::string& file)
    {
        if (compileTaskList_.find(file) == compileTaskList_.end()) {
            ASC_LOG_ASC_ERROR(COMPILE, "GetCompileTask fail!");
            return nullptr;
        }
        return std::dynamic_pointer_cast<T>(compileTaskList_[file]);
    }
    template <typename T>
    AsccStatus CompileTask(const CompileArgs& args) const
    {
        std::shared_ptr<T> task = std::make_shared<T>(args);
        return task->Compile();
    }
private:
    AsccCompileFactory() = default;
    ~AsccCompileFactory() = default;
    AsccCompileFactory(const AsccCompileFactory&) = delete;
    AsccCompileFactory& operator=(const AsccCompileFactory&) = delete;

private:
    std::unordered_map<std::string, std::shared_ptr<AsccCompileBase>> compileTaskList_;
};

/**
 * @brief Registers compilation task in scheduler queue
 * @tparam T compilation type (e.g., AsccCompileV220, AsccCompileHost)
 * @param args Configured compilation parameters
 * @return bool true/false indicates whether the execution is successful
 */
template <typename T>
AsccStatus CompileRegister(const Ascc::CompileArgs& args)
{
    auto& factory = AsccCompileFactory::GetInstance();
    factory.Register<T>(args.file, args);
    return AsccStatus::SUCCESS;
}


/**
 * @brief Get compilation task task from filepath and compilation type
 * @tparam T Expected task type (must derive from AsccCompileBase)
 * @param file Source filepath associated with the task
 * @return shared_ptr to task object if completed, get nullptr if failed
 */
template <typename T>
std::shared_ptr<T> GetCompileTask(const std::string& file)
{
    auto& factory = AsccCompileFactory::GetInstance();
    return factory.GetCompileTask<T>(file);
}

/**
 * @brief Compile immediately after passing parameters
 * @tparam T compilation type (e.g., AsccCompileV220, AsccCompileHost)
 * @param args Configured compilation parameters
 * @return true if compilation succeeded, false otherwise
 */
template <typename T>
AsccStatus CompileTask(const Ascc::CompileArgs& args)
{
    auto& factory = AsccCompileFactory::GetInstance();
    return factory.CompileTask<T>(args);
}

} // Ascc

#endif // __INCLUDE_ASCC_COMPILE_FACTORY_H__