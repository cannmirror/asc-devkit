/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#include <string>
#include <iostream>
#include <gtest/gtest.h>
#include <unistd.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "asc_dev_section_generate.h"
#include "asc_dev_stub_generator.h"
#include "asc_dev_meta_generator.h"
#include "asc_info_manager.h"

class TEST_ASC_DEV_SECTION_GENERATE : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_DEV_SECTION_GENERATE, asc_get_device_code_failure)
{
    AscPlugin::KernelInfo info;
    info.kernelMangledNameConsiderPrefix = "";
    const auto [deviceResult, deviceStub, metaInfo] =
        GetDeviceCode(info, {AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY}, AscPlugin::KfcScene::Close);
    EXPECT_EQ(deviceResult, 1);
}

TEST_F(TEST_ASC_DEV_SECTION_GENERATE, asc_get_device_code_success_310P)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND310P);
    std::string goldenAiv = R"(namespace Foo1::Foo2 {
extern "C" __attribute__((aiv)) __global__ __aicore__ void __device_stub__mangling_add(__attribute__((cce_global)) uint8_t * __ascendc_overflow_status)
{
    __origin__add();
}
}
)";
    std::string goldenAic = R"(namespace Foo1::Foo2 {
extern "C" __attribute__((aic)) __global__ __aicore__ void __device_stub__mangling_add(__attribute__((cce_global)) uint8_t * __ascendc_overflow_status)
{
    __origin__add();
}
}
)";
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.kernelMangledName = "mangling_add";
    info.kernelMangledNameConsiderPrefix = "prefix_mangling_add";
    info.namespaces = {"Foo1", "Foo2"};
    auto [deviceResult, deviceStub, metaInfo] =
        GetDeviceCode(info, {AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY}, AscPlugin::KfcScene::Close);
    EXPECT_EQ(deviceResult, 0);
    EXPECT_EQ(deviceStub, goldenAiv);
    std::tie(deviceResult, deviceStub, metaInfo) =
        GetDeviceCode(info, {AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY}, AscPlugin::KfcScene::Close);
    EXPECT_EQ(deviceResult, 0);
    EXPECT_EQ(deviceStub, goldenAic);

    manager.SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND910B);
}

TEST_F(TEST_ASC_DEV_SECTION_GENERATE, asc_get_device_code_success)
{
    std::string goldenAiv = R"(namespace Foo1::Foo2 {
extern "C" __attribute__((aiv)) __global__ __aicore__ void __device_stub__mangling_add(__attribute__((cce_global)) uint8_t * __ascendc_overflow_status)
{
    __origin__add();
}
}
)";
    std::string goldenAic = R"(namespace Foo1::Foo2 {
extern "C" __attribute__((aic)) __global__ __aicore__ void __device_stub__mangling_add(__attribute__((cce_global)) uint8_t * __ascendc_overflow_status)
{
    __origin__add();
}
}
)";
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.kernelMangledName = "mangling_add";
    info.kernelMangledNameConsiderPrefix = "prefix_mangling_add";
    info.namespaces = {"Foo1", "Foo2"};
    auto [deviceResult, deviceStub, metaInfo] =
        GetDeviceCode(info, {AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY}, AscPlugin::KfcScene::Close);
    EXPECT_EQ(deviceResult, 0);
    EXPECT_EQ(deviceStub, goldenAiv);
    std::tie(deviceResult, deviceStub, metaInfo) =
        GetDeviceCode(info, {AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY}, AscPlugin::KfcScene::Close);
    EXPECT_EQ(deviceResult, 0);
    EXPECT_EQ(deviceStub, goldenAic);
}

TEST_F(TEST_ASC_DEV_SECTION_GENERATE, asc_get_device_code_template_success)
{
    std::string goldenAiv = R"(extern "C" __attribute__((aiv)) __global__ __aicore__ void __device_stub__add_mangling_int_float(__attribute__((cce_global)) uint8_t * __ascendc_overflow_status)
{
    __origin__add<int, float>();
}
)";
    std::string goldenAic = R"(extern "C" __attribute__((aic)) __global__ __aicore__ void __device_stub__add_mangling_int_float(__attribute__((cce_global)) uint8_t * __ascendc_overflow_status)
{
    __origin__add<int, float>();
}
)";
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.kernelMangledName = "mangling_add";
    info.kernelMangledNameConsiderPrefix = "prefix_mangling_add";
    info.kernelParameters = {};
    info.templateParameters = {
        {"typename", "T", false, "", ""},
        {"typename", "U", false, "", ""}
    };
    info.templateInstances = {
        {
            {"int", "float"},
            {},
            "add_mangling_int_float",
            "prefix_add_mangling_int_float"
        }
    };
    info.isTemplate = true;

    auto [deviceResult, deviceStub, metaInfo] =
        GetDeviceCode(info, {AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY}, AscPlugin::KfcScene::Close);
    EXPECT_EQ(deviceResult, 0);
    EXPECT_EQ(deviceStub, goldenAiv);
    std::tie(deviceResult, deviceStub, metaInfo) =
        GetDeviceCode(info, {AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY}, AscPlugin::KfcScene::Close);
    EXPECT_EQ(deviceResult, 0);
    EXPECT_EQ(deviceStub, goldenAic);
}

TEST_F(TEST_ASC_DEV_SECTION_GENERATE, asc_dev_stub_generator)
{
    std::string golden = R"(namespace Foo1::Foo2 {
extern "C" __global__ __aicore__ void __device_stub__add_mangling_int_float(__attribute__((cce_global)) uint8_t * __ascendc_dump_addr, int i, __attribute__((annotate("kfc_workspace"))) uint8_t* workspace, __attribute__((cce_global)) uint8_t * __ascendc_overflow_status)
{
    AscendC::InitDump(true, __ascendc_dump_addr, ONE_CORE_DUMP_SIZE);
    icache_preload(1);
    if (g_sysFftsAddr != nullptr) {
        set_ffts_base_addr((uint64_t)g_sysFftsAddr);
    }
    uint64_t __ascendc_timestamp = 0;
    uint64_t __ascendc_version = 0;
     __gm__ char* __ascendc_version_str = nullptr;
    GetCannVersion(__ascendc_version_str, __ascendc_version, __ascendc_timestamp);
    if (__ascendc_timestamp == 0) {
        AscendC::printf("[WARNING]: CANN TimeStamp is invalid, CANN TimeStamp is %u\n", __ascendc_timestamp);
    } else {
        AscendC::printf("CANN Version: %s, TimeStamp: %u\n", (__gm__ const char*)(__ascendc_version_str), __ascendc_timestamp);
    }
    GM_ADDR ascendc_workspace_param;
    GM_ADDR ascendc_workspace_usr;
    ascendc_workspace_param = workspace;
    if (ascendc_workspace_param == nullptr) {
        return;
    }
    AscendC::SetSysWorkspaceForce(ascendc_workspace_param);
    ascendc_workspace_usr = AscendC::GetUserWorkspace(ascendc_workspace_param);
    if constexpr (g_coreType == AscendC::AIC) {
        matmul::clearWorkspace(ascendc_workspace_param);
    }
    workspace = ascendc_workspace_usr;
    __origin__add<int, float>(i, workspace);
}
}
)";
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.kernelMangledName = "add_mangling";
    info.kernelParameters = {
        {"int", "i", false, "", ""},
        {"uint8_t*", "workspace", false, "", "__attribute__((annotate(\"kfc_workspace\")))"}
    };
    info.ratio = {false, 0, 0};
    info.templateParameters = {
        {"typename", "T", false, "", ""},
        {"typename", "U", false, "", ""}
    };
    info.namespaces = {"Foo1", "Foo2"};
    info.templateInstances = {
        {
            {"int", "float"},
            {
                {"int", "i", false, "", ""},
                {"uint8_t*", "workspace", false, "", "__attribute__((annotate(\"kfc_workspace\")))"}
            },
            "add_mangling_int_float",
            "prefix_add_mangling_int_float",
            {true, 1, 1}
        }
    };
    info.isTemplate = true;
    AscPlugin::AscDevStubGenerator devStubGen = AscPlugin::AscDevStubGenerator(
        info, {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1}, AscPlugin::KfcScene::Open);
    devStubGen.dumpIsNeedInit_ = true;
    devStubGen.dumpIsNeedPrintVersion_ = true;
    devStubGen.socVersion_ = AscPlugin::ShortSocVersion::ASCEND910B;
    auto deviceStub = devStubGen.GenCode();
    EXPECT_EQ(deviceStub, golden);
}

TEST_F(TEST_ASC_DEV_SECTION_GENERATE, asc_dev_stub_workspace_arg)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.kernelMangledName = "add_mangling";
    info.kernelParameters = {
        {"uint8_t*", "workspace", false, "", "__attribute__((annotate(\"kfc_workspace\")))"},
        {"uint8_t*", "workspace_bak", false, "", "__attribute__((annotate(\"kfc_workspace\")))"}
    };
    AscPlugin::AscDevStubGenerator devStubGen = AscPlugin::AscDevStubGenerator(
        info, {AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY}, AscPlugin::KfcScene::Close);
    devStubGen.socVersion_ = AscPlugin::ShortSocVersion::ASCEND910B;
    manager.hasWorkspace_ = true;
    manager.hasTiling_ = true;
    EXPECT_EQ(devStubGen.GetWorkspaceArgName(), std::string("workspace"));
    manager.hasTiling_ = false;
    EXPECT_EQ(devStubGen.GetWorkspaceArgName(), std::string("workspace_bak"));
    manager.hasWorkspace_ = false;
}

TEST_F(TEST_ASC_DEV_SECTION_GENERATE, asc_dev_stub_workspace_arg_failure)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.kernelMangledName = "add_mangling";
    AscPlugin::AscDevStubGenerator devStubGen = AscPlugin::AscDevStubGenerator(
        info, {AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY}, AscPlugin::KfcScene::Close);
    devStubGen.socVersion_ = AscPlugin::ShortSocVersion::ASCEND910B;
    manager.hasWorkspace_ = true;
    manager.hasTiling_ = true;
    EXPECT_TRUE(devStubGen.GetWorkspaceArgName().empty());
    manager.hasTiling_ = false;
    EXPECT_TRUE(devStubGen.GetWorkspaceArgName().empty());
    manager.hasWorkspace_ = false;
}

#define TEST_DEV_META_GEN(goldenContent, kernelType, isTemplateFunc)                                     \
    TEST_F(TEST_ASC_DEV_SECTION_GENERATE, asc_dev_meta_generator_##kernelType##_##isTemplateFunc)        \
    {                                                                                                    \
        std::string golden = goldenContent;                                                              \
        AscPlugin::KernelInfo info;                                                                      \
        info.kernelName = "add";                                                                         \
        info.kernelMangledName = "add_mangling";                                                         \
        info.kernelMangledNameConsiderPrefix = "prefix_add_mangling";                                    \
        info.kernelParameters = {{"int", "i", false, "", ""}, {"uint8_t*", "workspace", false, "", "__attribute__((annotate(\"kfc_workspace\")))"}}; \
        info.templateParameters = {{"typename", "T", false, "", ""}, {"typename", "U", false, "", ""}};  \
        info.templateInstances = {{{"int", "float"},                                                     \
            {{"int", "i", false, "", ""}, {"uint8_t*", "workspace", false, "", "__attribute__((annotate(\"kfc_workspace\")))"}},                             \
            "add_mangling_int_float",                                                                    \
            "prefix_add_mangling_int_float"}};                                                           \
        info.isTemplate = isTemplateFunc;                                                                \
        AscPlugin::AscDevMetaGenerator devMetaGen =                                                      \
            AscPlugin::AscDevMetaGenerator(info, {AscPlugin::KernelMetaType::kernelType});               \
        auto metaSection = devMetaGen.GenCode();                                                         \
        EXPECT_EQ(metaSection, golden);                                                                  \
    }

TEST_DEV_META_GEN(
    "\n#if defined(__DAV_C220_CUBE__)\nstatic const struct FunLevelMixCoreType "
    "add_mangling_int_float_meta_section __attribute__((used, "
    "section(\".ascend.meta.add_mangling_int_float_mix_aic\"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, "
    "K_TYPE_MIX_AIC_MAIN}, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 1} };\n#endif\n\n#if "
    "defined(__DAV_C220_VEC__)\nstatic const struct FunLevelMixCoreType add_mangling_int_float_meta_section "
    "__attribute__((used, section(\".ascend.meta.add_mangling_int_float_mix_aiv\"))) = { {{F_TYPE_KTYPE, "
    "sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN}, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 1} };\n#endif\n"
    "static const struct AscendCFeatureFlag __ascendc_feature_ffts__0 __attribute__ ((used, section (\".ascend."
    "meta\"))) = {4, 4, 2};\n",
    KERNEL_TYPE_MIX_AIC_1_1, true)

TEST_DEV_META_GEN(
    "\n#if defined(__DAV_C220_CUBE__)\nstatic const struct FunLevelMixCoreType "
    "add_mangling_int_float_meta_section __attribute__((used, "
    "section(\".ascend.meta.add_mangling_int_float_mix_aic\"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, "
    "K_TYPE_MIX_AIC_MAIN}, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };\n#endif\n\n#if "
    "defined(__DAV_C220_VEC__)\nstatic const struct FunLevelMixCoreType add_mangling_int_float_meta_section "
    "__attribute__((used, section(\".ascend.meta.add_mangling_int_float_mix_aiv\"))) = { {{F_TYPE_KTYPE, "
    "sizeof(unsigned int)}, K_TYPE_MIX_AIC_MAIN}, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 2} };\n#endif\n"
    "static const struct AscendCFeatureFlag __ascendc_feature_ffts__1 __attribute__ ((used, section (\".ascend."
    "meta\"))) = {4, 4, 2};\n",
    KERNEL_TYPE_MIX_AIC_1_2, true)

TEST_DEV_META_GEN(
    "\n#if defined(__DAV_C220_CUBE__)\nstatic const struct FunLevelMixCoreType "
    "add_mangling_int_float_meta_section __attribute__((used, "
    "section(\".ascend.meta.add_mangling_int_float_mix_aic\"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, "
    "K_TYPE_MIX_AIC_MAIN}, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 1, 0} };\n#endif\n"
    "static const struct AscendCFeatureFlag __ascendc_feature_ffts__2 __attribute__ ((used, section (\".ascend."
    "meta\"))) = {4, 4, 2};\n",
    KERNEL_TYPE_MIX_AIC_1_0, true)

TEST_DEV_META_GEN(
    "\n#if defined(__DAV_C220_VEC__)\nstatic const struct FunLevelMixCoreType "
    "add_mangling_int_float_meta_section __attribute__((used, "
    "section(\".ascend.meta.add_mangling_int_float_mix_aiv\"))) = { {{F_TYPE_KTYPE, sizeof(unsigned int)}, "
    "K_TYPE_MIX_AIV_MAIN}, {{F_TYPE_MIX_TASK_RATION, sizeof(unsigned int)}, 0, 1} };\n#endif\n"
    "static const struct AscendCFeatureFlag __ascendc_feature_ffts__3 __attribute__ ((used, section (\".ascend."
    "meta\"))) = {4, 4, 2};\n",
    KERNEL_TYPE_MIX_AIV_1_0, true)

TEST_DEV_META_GEN(
    "\n#if defined(__DAV_C220_CUBE__)\nstatic const struct FunLevelKType add_mangling_int_float_meta_section "
    "__attribute__((used, section(\".ascend.meta.add_mangling_int_float\"))) = { {{F_TYPE_KTYPE, "
    "sizeof(unsigned int)}, K_TYPE_AIC},  };\n#endif\n",
    KERNEL_TYPE_AIC_ONLY, true)

TEST_DEV_META_GEN(
    "\n#if defined(__DAV_C220_VEC__)\nstatic const struct FunLevelKType add_mangling_int_float_meta_section "
    "__attribute__((used, section(\".ascend.meta.add_mangling_int_float\"))) = { {{F_TYPE_KTYPE, "
    "sizeof(unsigned int)}, K_TYPE_AIV},  };\n#endif\n",
    KERNEL_TYPE_AIV_ONLY, true)

TEST_DEV_META_GEN("\n#if defined(__DAV_C220_VEC__)\nstatic const struct FunLevelKType add_mangling_meta_section "
                  "__attribute__((used, section(\".ascend.meta.add_mangling\"))) = { {{F_TYPE_KTYPE, "
                  "sizeof(unsigned int)}, K_TYPE_AIV},  };\n#endif\n",
    KERNEL_TYPE_AIV_ONLY, false)