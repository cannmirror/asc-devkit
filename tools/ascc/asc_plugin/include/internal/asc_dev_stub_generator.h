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
 * \file asc_dev_stub_generator.h
 * \brief
 */

#ifndef __INCLUDE_INTERNAL_ASC_DEV_STUB_GENERATOR_H__
#define __INCLUDE_INTERNAL_ASC_DEV_STUB_GENERATOR_H__

#include <sstream>
#include <iostream>
#include <utility>
#include <string>
#include <vector>

#include "asc_utils.h"
#include "asc_struct.h"

namespace AscPlugin {
class AscDevStubGenerator {
public:
    AscDevStubGenerator(const KernelInfo& kernelInfo, const std::vector<KernelMetaType>& kernelType,
        const KfcScene& kfcScene);
    std::string GenCode();

private:
    void GenStubFuncDecl(const std::string& globalSymbol, const std::vector<Param>& args,
        const KernelMetaType& kernelType);
    void GenStubFuncImpl(const bool& isMix, const bool& isHardSync, const std::string& templateArgs);
    void StubFuncDumpAndHardSyncImpl(const bool& isMix, const bool& isHardSync);
    void StubFuncWorkSpaceImpl(const bool& isMix);
    void StubFuncCallImpl(const std::string& templateArgs);
    void GenStubKernelFunc(const bool& isMix, const bool& isHardSync);
    void GenStubKernelFunc(const bool& isMix, const bool& isHardSync, const TemplateInstance& tempInst);

    std::pair<bool, bool> GetArchInfo(const ShortSocVersion& socVersion) const;
    std::string GetWorkspaceArgName() const;
    std::string GetTempArgsList(const TemplateInstance &tempInst);
    void UpdateParams();
    void GenCodeForL2Cache();

    KernelInfo kernelInfo_;
    std::vector<KernelMetaType> kernelType_ = {KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2};
    ShortSocVersion socVersion_ = ShortSocVersion::ASCEND910B;
    bool dumpTypeIsNotNone_ = false;
    bool dumpAscendCStamp_ = false;
    bool dumpTypeIsPrintf_ = false;
    KfcScene kfcScene_ = KfcScene::Close;
    std::stringstream codeStream_;
    std::string originParamsCallList_;
    std::string workspaceArgName_;
};


} // namespace AscPlugin
#endif // __INCLUDE_INTERNAL_ASC_DEV_STUB_GENERATOR_H__