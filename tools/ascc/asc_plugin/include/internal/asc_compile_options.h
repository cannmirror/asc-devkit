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
 * \file asc_compile_options.h
 * \brief
 */

#ifndef __INCLUDE_INTERNAL_ASC_COMPILE_OPTIONS_H__
#define __INCLUDE_INTERNAL_ASC_COMPILE_OPTIONS_H__

#include <string>
#include <vector>
#include <unordered_map>
#include <functional>

#include "asc_struct.h"
#include "asc_utils.h"

namespace AscPlugin {
KernelTypeResult CheckHasMixKernelFunc();
class CompileOptionManager {
public:
    CompileOptionManager();
    std::vector<std::string> GetHostCompileOptions() const;
    std::vector<std::string> GetDeviceCompileOptions(CoreType type) const;

private:
    template<ShortSocVersion soc>
    std::vector<std::string> GetDeviceCompileOptionsWithSoc(CoreType coreType) const;
    void InitDispatchTable();
    template<ShortSocVersion soc>
    void RegisterOptHandler();

private:
    using SocOptHandler = std::function<std::vector<std::string>(CoreType)>;

    ShortSocVersion socVersion_ = ShortSocVersion::ASCEND910B;
    bool isAutoSyncOn_ = true;
    bool l2CacheOn_ = true;

    std::string optiLevel_;
    std::vector<std::string> deviceCompileOpt_;
    std::unordered_map<ShortSocVersion, SocOptHandler> dispatchTable_;
};
} // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_COMPILE_OPTIONS_H__