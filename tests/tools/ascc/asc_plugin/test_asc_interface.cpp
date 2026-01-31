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
#include <nlohmann/json.hpp>
#include <sys/utsname.h>
#define private public
#include "asc_utils.h"
#include "asc_info_manager.h"
#include "asc_interface.h"
#include "asc_json_string.h"
#include "asc_struct.h"
#include "asc_ast_device_analyzer.h"
#include "asc_dev_func_registry_generate.h"
#include "asc_ast_device_consumer.h"
#include "asc_dev_section_generate.h"

static std::string registerBinary(R"(#include <stdio.h>
#include <stdint.h>
extern "C" {
int32_t AscendDevBinaryLazyRegister(const char* binBuf, size_t binSize, void** handle);
int32_t AscendGetFuncFromBinary(void* const binHandle, const char* kernelName, void** funcHandle);
int32_t AscendLaunchKernelWithHostArgs(void* funcHandle,
    uint32_t blockDim, void* stream, void* hostArgs, size_t argsSize, uint32_t ubufDynamicSize);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
void AscendProfRegister();
using rtFuncHandle = void*;
uint32_t AscendCGetProfkTypeImpl(const rtFuncHandle funcHandle);
}

namespace {
class AscRegister {
public:
static AscRegister& GetInstance() {
    static AscRegister instance;
    return instance;
}
void* binHandle = nullptr;
private:
AscRegister() {
    uint32_t ret = AscendDevBinaryLazyRegister(fatbinDataPtr, fatbinDataLength, &binHandle);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] Kernel binary register failure! ret %d \n", ret);
    }
    AscendProfRegister();
}
~AscRegister() = default;
AscRegister(const AscRegister&) = delete;
AscRegister& operator=(const AscRegister&) = delete;
};

} // namespace

namespace AscPluginGenerator {
__attribute__ ((visibility("hidden"))) int32_t BindKernelRegisterFunc(void (*)(void*)) { return 0; }
__attribute__ ((visibility("hidden"))) uint32_t LaunchAndProfiling(const char *kernelName, uint32_t blockDim,
    void *stream, void **args, uint32_t size, uint32_t ktype, const uint32_t ubufDynamicSize)
{
    static auto& reg = AscRegister::GetInstance();
    void* funcHandle = nullptr;
    uint32_t ret = AscendGetFuncFromBinary(reg.binHandle, kernelName, &funcHandle);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] Get kernel function failure! ret %d \n", ret);
        return 1;
    }
    uint64_t startTime;
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        ktype = AscendCGetProfkTypeImpl(funcHandle);
        StartAscendProf(kernelName, &startTime);
    }
    ret = AscendLaunchKernelWithHostArgs(funcHandle, blockDim, stream, (void*)args, size, ubufDynamicSize);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] Launch kernel failure! ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(kernelName, blockDim, ktype, startTime);
    }
    return ret;
}
} // namespace AscPluginGenerator
)");

class TestAscInterFace : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

namespace AscPlugin {
    void to_json(nlohmann::json& jsonObj, const Param& param) {
        jsonObj = nlohmann::json{
            {"Type", param.type},
            {"Name", param.name},
            {"HasDefaultValue", param.hasDefaultValue},
            {"Attribute", param.attribute},
            {"DefaultValue", param.defaultValue},
            {"TypeClass", param.typeClass}
        };
    }
    void to_json(nlohmann::json& jsonObj, const CoreRatio& ratio) {
        jsonObj = nlohmann::json{
            {"IsCoreRatio", ratio.isCoreRatio},
            {"CubeNum", ratio.cubeNum},
            {"VecNum", ratio.vecNum}
        };
    }
    void to_json(nlohmann::json& jsonObj, const TemplateInstance& instance) {
        jsonObj = nlohmann::json{
            {"TemplateInstantiationArguments", instance.templateInstantiationArguments},
            {"InstanceKernelParameters", instance.instanceKernelParameters},
            {"InstanceMangledName", instance.instanceMangledName},
            {"InstanceMangledNameConsiderPrefix", instance.instanceMangledNameConsiderPrefix},
            {"Ratio", instance.ratio},
        };
    }
    void to_json(nlohmann::json& jsonObj, const KernelInfo& info) {
        jsonObj = nlohmann::json{
            {"KernelName", info.kernelName},
            {"KernelMangledName", info.kernelMangledName},
            {"KernelMangledNameConsiderPrefix", info.kernelMangledNameConsiderPrefix},
            {"FileName", info.fileName},
            {"LineNum", info.lineNum},
            {"ColNum", info.colNum},
            {"Namespaces", info.namespaces},
            {"KernelParameters", info.kernelParameters},
            {"KernelAttributes", info.kernelAttributes},
            {"Ratio", info.ratio},
            {"IsTemplate", info.isTemplate},
            {"TemplateParameters", info.templateParameters},
            {"TemplateInstances", info.templateInstances}
        };
    }

    void from_json(const nlohmann::json& jsonObj, GenKernelResult& result)
    {
        jsonObj.at("HostStub").get_to(result.hostStub);
        jsonObj.at("DeviceStub").get_to(result.deviceStub);
        jsonObj.at("MetaInfo").get_to(result.metaInfo);
        jsonObj.at("Type").get_to(result.type);
    }

    void to_json(nlohmann::json& jsonObj, const PrologueConfig& config) {
        jsonObj = nlohmann::json{
            {"SaveTemp", config.saveTemp},
            {"Verbose", config.verbose},
            {"GenMode", config.genMode},
            {"NpuSoc", config.npuSoc},
            {"NpuArch", config.npuArch},
            {"LogPath", config.logPath},
            {"TmpPath", config.tmpPath},
            {"Source", config.source},
            {"BinaryPtrName", config.binaryPtrName},
            {"BinaryLenName", config.binaryLenName},
            {"CompileArgs", config.compileArgs}
        };
    }

    void from_json(const nlohmann::json& jsonObj, PreCompileOptsResult& result)
    {
        jsonObj.at("CompileOptions").get_to(result.compileOptions);
    }
}

void PrologueConfigUpdate(AscPlugin::PrologueConfig& config)
{
    config.saveTemp = false;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB"};
}

// Note: should be the first testcase in this .cpp
TEST_F(TestAscInterFace, asc_cannpath_init)
{
    AscPlugin::InfoManager::GetInstance().saveTempRequested_ = false;
    MOCKER(AscPlugin::EndsWith).stubs().will(returnValue(true));
    const char* a;
    auto res = AscPlugin::PluginGetPreCompileOpts(&a);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    // expect cann path to be set up by InitCannPath
    // but in llt, the path is different with cann package
    EXPECT_EQ(manager.GetCannPath().empty(), false);
    EXPECT_EQ(res, AscPlugin::ASC_SUCCESS);
    free(const_cast<char*>(a));
    manager.SetCannPath("/usr/local/Ascend/cann");
}

TEST_F(TestAscInterFace, asc_plugin_gen_kernel_failure_with_no_kernel)
{
    AscPlugin::KernelInfo info;
    info.ratio = {false, 0, 0};
    nlohmann::json infoObj= info;
    const char* infoObjPtr = strdup(infoObj.dump().c_str());
    const char* a;
    auto res = AscPlugin::PluginGenKernel(&a, infoObjPtr);
    EXPECT_EQ(res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(infoObjPtr));
}

TEST_F(TestAscInterFace, asc_plugin_gen_kernel_failure_with_no_mangling)
{
    AscPlugin::KernelInfo info;
    info.ratio = {false, 0, 0};
    info.kernelName = "add";
    nlohmann::json infoObj= info;
    const char* infoObjPtr = strdup(infoObj.dump().c_str());
    const char* a;
    auto res = AscPlugin::PluginGenKernel(&a, infoObjPtr);
    EXPECT_EQ(res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(infoObjPtr));
}

TEST_F(TestAscInterFace, asc_plugin_gen_kernel_failure_with_nullptr)
{
    const char* a;
    auto res = AscPlugin::PluginGenKernel(&a, nullptr);
    EXPECT_EQ(res, AscPlugin::ASC_NULLPTR);
}

TEST_F(TestAscInterFace, asc_plugin_gen_kernel_gen_device_code_failure)
{
    const char* a;
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.lineNum = 32U;
    info.colNum = 24U;
    info.isTemplate = true;
    info.fileName = "test.cpp";
    info.kernelMangledName = "mangling_add";
    info.kernelMangledNameConsiderPrefix = "prefix_mangling_add";
    nlohmann::json infoObj= info;
    const char* infoObjPtr = strdup(infoObj.dump().c_str());
    std::unordered_set<AscPlugin::KernelMetaType> test_set{AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2};
    std::pair<std::unordered_set<AscPlugin::KernelMetaType>, AscPlugin::KfcScene> test_pair{test_set, AscPlugin::KfcScene::Open};
    MOCKER(AscPlugin::GetKernelFuncScene).stubs().will(returnValue(test_pair));
    MOCKER(AscPlugin::GetDeviceCode).stubs().will(returnValue(std::tuple{1, std::string("a"), std::string("a")}));
    AscPlugin::InfoManager::GetInstance().SetSocVersion("Ascend950DT_9591");
    AscPlugin::InfoManager::GetInstance().SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND950);
    auto res = AscPlugin::PluginGenKernel(&a, infoObjPtr);
    EXPECT_EQ(res, AscPlugin::ASC_FAILURE);
    AscPlugin::InfoManager::GetInstance().SetSocVersion("Ascend910B");
    AscPlugin::InfoManager::GetInstance().SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND910B);
}

TEST_F(TestAscInterFace, asc_plugin_gen_kernel_gen_device_code_success_without_910B)
{
    const char* a;
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.lineNum = 32U;
    info.colNum = 24U;
    info.isTemplate = true;
    info.fileName = "test.cpp";
    info.kernelMangledName = "mangling_add";
    info.kernelMangledNameConsiderPrefix = "prefix_mangling_add";
    nlohmann::json infoObj= info;
    const char* infoObjPtr = strdup(infoObj.dump().c_str());
    std::unordered_set<AscPlugin::KernelMetaType> test_set{AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2};
    std::pair<std::unordered_set<AscPlugin::KernelMetaType>, AscPlugin::KfcScene> test_pair{test_set, AscPlugin::KfcScene::Open};
    MOCKER(AscPlugin::GetKernelFuncScene).stubs().will(returnValue(test_pair));
    MOCKER(AscPlugin::GetDeviceCode).stubs().will(returnValue(std::tuple{0, std::string("a"), std::string("a")}));
    AscPlugin::InfoManager::GetInstance().SetSocVersion("Ascend950DT_9591");
    AscPlugin::InfoManager::GetInstance().SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND950);
    auto res = AscPlugin::PluginGenKernel(&a, infoObjPtr);
    EXPECT_EQ(res, AscPlugin::ASC_SUCCESS);
    AscPlugin::InfoManager::GetInstance().SetSocVersion("Ascend910B");
    AscPlugin::InfoManager::GetInstance().SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND910B);
}

TEST_F(TestAscInterFace, asc_plugin_gen_kernel_910B_template)
{
    const char* a;
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.lineNum = 32U;
    info.colNum = 24U;
    info.isTemplate = true;
    info.fileName = "test.cpp";
    info.kernelMangledName = "mangling_add";
    info.kernelMangledNameConsiderPrefix = "prefix_mangling_add";
    AscPlugin::TemplateInstance instanA, instanB;
    instanA.instanceMangledName = "tetsMangledName_a";
    instanB.instanceMangledName = "tetsMangledName_b";
    info.templateInstances = {instanA, instanB};
    nlohmann::json infoObj= info;
    const char* infoObjPtr = strdup(infoObj.dump().c_str());
    std::unordered_set<AscPlugin::KernelMetaType> test_set{AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2};
    std::pair<std::unordered_set<AscPlugin::KernelMetaType>, AscPlugin::KfcScene> test_pair{test_set, AscPlugin::KfcScene::Open};
    MOCKER(AscPlugin::GetKernelFuncScene).stubs().will(returnValue(test_pair));
    MOCKER(AscPlugin::GetDeviceCode).stubs().will(returnValue(std::tuple{1, std::string("a"), std::string("a")}));
    auto res = AscPlugin::PluginGenKernel(&a, infoObjPtr);
    EXPECT_EQ(res, AscPlugin::ASC_SUCCESS);
}

TEST_F(TestAscInterFace, asc_plugin_gen_kernel_success)
{
    std::string golden = R"()";
    AscPlugin::KernelInfo info;
    info.kernelName = "add";
    info.kernelMangledName = "mangling_add";
    info.kernelMangledNameConsiderPrefix = "prefix_mangling_add";
    info.namespaces = {"Foo1", "Foo2"};
    info.ratio = {true, 1, 1};
    nlohmann::json infoObj= info;
    const char* infoObjPtr = strdup(infoObj.dump().c_str());

    const char* a;
    auto res = AscPlugin::PluginGenKernel(&a, infoObjPtr);
    AscPlugin::GenKernelResult kernelResult = nlohmann::json::parse(a).get<AscPlugin::GenKernelResult>();
    EXPECT_EQ(res, 0);
    EXPECT_EQ(kernelResult.deviceStub, golden);
    free(const_cast<char*>(a));
    free(const_cast<char*>(infoObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(manager.GetSocVersion(), "Ascend910B1");
    free(const_cast<char*>(a));
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_saveTemp)
{
    AscPlugin::PrologueConfig config;
    config.saveTemp = true;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB","-sanitizer"};

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(manager.GetSocVersion(), "Ascend910B1");
    EXPECT_EQ(manager.IsL2CacheEnabled(), false);
    manager.enableL2Cache_ = true;
    free(const_cast<char*>(a));
    free(const_cast<char*>(configObjPtr));
    system("rm -rf /tmp/asc_plugin");
}

TEST_F(TestAscInterFace, asc_PluginPrologue_saveTemp_logFolderPath_fail)
{
    AscPlugin::PrologueConfig config;
    config.saveTemp = true;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB"};
    MOCKER(AscPlugin::CreateDirectory).stubs().will(returnValue(AscPlugin::ASC_FAILURE));

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);

    EXPECT_EQ(exec_res, 1);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_saveTemp_tmpFolderPath_fail)
{
    AscPlugin::PrologueConfig config;
    config.saveTemp = true;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB"};
    MOCKER(AscPlugin::CreateDirectory)
        .expects(exactly(2))
        .will(returnValue(AscPlugin::ASC_SUCCESS))
        .then(returnValue(AscPlugin::ASC_FAILURE));

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);

    EXPECT_EQ(exec_res, 1);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_saveTemp_logPathFolderNotExist)
{
    AscPlugin::PrologueConfig config;
    config.saveTemp = true;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.logPath = "/tmp/ascc_plugin/tests/log";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB"};

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    system("rm -rf /tmp/ascc_plugin/tests/log");
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_saveTemp_tmpPathFolderNotExist)
{
    AscPlugin::PrologueConfig config;
    config.saveTemp = true;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.tmpPath = "/tmp/ascc_plugin/tests/temp";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB"};

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    system("rm -rf /tmp/ascc_plugin/tests/temp");
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_saveTemp_unDefaultFolderPath)
{
    system("mkdir -p /tmp/ut_test_ascc_plugin/ut_test/temp");
    system("mkdir -p /tmp/ut_test_ascc_plugin/ut_test/log");
    AscPlugin::PrologueConfig config;
    config.saveTemp = true;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.logPath = "/tmp/ut_test_ascc_plugin/ut_test/log";
    config.tmpPath = "/tmp/ut_test_ascc_plugin/ut_test/temp";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB"};

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(manager.GetSocVersion(), "Ascend910B1");
    free(const_cast<char*>(a));
    free(const_cast<char*>(configObjPtr));
    system("rm -rf /tmp/ut_test_ascc_plugin");
    system("rm -rf /tmp/ut_test_ascc_plugin");
}

TEST_F(TestAscInterFace, asc_PluginPrologue_saveTemp_GenerateTimestamp_fail)
{
    AscPlugin::PrologueConfig config;
    config.saveTemp = true;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB"};
    MOCKER(AscPlugin::GenerateTimestamp).stubs().will(returnValue(std::string("")));

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);

    EXPECT_EQ(exec_res, 1);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_error_npuArch)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.npuArch = "dav-c220-mix";

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_empty_compileArgs)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.compileArgs = {};

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_error_soc)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.npuSoc = "Ascend910BGG";
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;

    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    EXPECT_EQ(exec_res, AscPlugin::ASC_SOC_NOT_SUPPORT);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_null_soc)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.npuSoc = "";
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;

    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    EXPECT_EQ(exec_res, AscPlugin::ASC_SUCCESS);
    free(const_cast<char*>(configObjPtr));
    free(const_cast<char*>(a));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_ast_failed)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.npuSoc = "Ascend950DT_9591";
    config.npuArch = "dav-c310";
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    MOCKER(&AscPlugin::AscAstDeviceAnalyzer::Process).stubs().will(returnValue(AscPlugin::ASC_FAILURE));
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(configObjPtr));
    AscPlugin::InfoManager::GetInstance().SetSocVersion("Ascend910B");
    AscPlugin::InfoManager::GetInstance().SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND910B);
}

TEST_F(TestAscInterFace, asc_PluginPrologue_binary_register_code_failed)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.binaryLenName = "";
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TestAscInterFace, asc_PluginPrologue_nullptr)
{
    AscPlugin::PrologueResult res;
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    int32_t exec_res = AscPlugin::PluginPrologue(nullptr, configObjPtr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_NULLPTR);
    free(const_cast<char*>(configObjPtr));
}

// Support MIX_1_1 and MIX_1_2 at same time in 71
TEST_F(TestAscInterFace, asc_PluginEpilogue_WithMix11_Mix12)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    manager.kernelFuncSymbolToFuncInfo_ = {};
    manager.AddGlobalSymbolInfo("__device_stub__add_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1, "a.cpp", 100, 110, AscPlugin::KfcScene::Open);
    manager.AddGlobalSymbolInfo("__device_stub__sub_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, "b.cpp", 120, 110, AscPlugin::KfcScene::Open);
    const char* a;
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    EXPECT_EQ(exec_res, AscPlugin::ASC_SUCCESS);
}

// MIX_1_2 with MIX_AIV_only is ok. Because has MIX, has -DA=A_mix_aiv/aic
TEST_F(TestAscInterFace, asc_PluginEpilogue_WithMixAIV_Mix12)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    manager.kernelFuncSymbolToFuncInfo_ = {};
    manager.AddGlobalSymbolInfo("__device_stub__add_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0, "a.cpp", 100, 110, AscPlugin::KfcScene::Close);
    manager.AddGlobalSymbolInfo("__device_stub__sub_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, "b.cpp", 120, 110, AscPlugin::KfcScene::Close);
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube", "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec", "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

// MIX_1_2 with MIX_AIC_only is ok. Because has MIX.
// Because has MIX_1_1
TEST_F(TestAscInterFace, asc_PluginEpilogue_WithMixAIC_Mix11)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    manager.kernelFuncSymbolToFuncInfo_ = {};
    manager.AddGlobalSymbolInfo("__device_stub__add_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0, "a.cpp", 100, 110, AscPlugin::KfcScene::Close);
    manager.AddGlobalSymbolInfo("__device_stub__sub_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1, "b.cpp", 120, 110, AscPlugin::KfcScene::Close);
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

// no function call, still pass device compile options
TEST_F(TestAscInterFace, asc_PluginEpilogue_no_device_call)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    manager.kernelFuncSymbolToFuncInfo_ = {};
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

// when not has mix, do not need extra compile options
// AIC_ONLY do not need -DA=A_mix_aic/aiv
TEST_F(TestAscInterFace, asc_PluginEpilogue)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    manager.AddGlobalSymbolInfo("__device_stub__add_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_AIC_ONLY, "a.cpp", 100, 110, AscPlugin::KfcScene::Close);
    manager.AddGlobalSymbolInfo("__device_stub__sub_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_AIV_ONLY, "a.cpp", 100, 130, AscPlugin::KfcScene::Close);
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TestAscInterFace, asc_PluginEpilogue_nullptr)
{
    int32_t exec_res = AscPlugin::PluginEpilogue(nullptr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_NULLPTR);
}

TEST_F(TestAscInterFace, asc_PluginEpilogue_no_dump)
{
    const char* a;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        // include paths
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        // include paths
        "-DL2_CACHE_HINT",
    };
    EXPECT_EQ(exec_res, 0);
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TestAscInterFace, asc_PluginEpilogue_310P)
{
    AscPlugin::PrologueResult res2;
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.compileArgs = {"-DASCENDC_DUMP=0", "-O1"};
    nlohmann::json configObj = config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* prologueRes;
    int32_t exec_res1 = AscPlugin::PluginPrologue(&prologueRes, configObjPtr);
    free(const_cast<char*>(prologueRes));
    free(const_cast<char*>(configObjPtr));

    AscPlugin::EpilogueResult res;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND310P);
    manager.SetCannPath("A");
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O2",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O2", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-m200",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-fp-ceiling=2",
        "-mllvm", "-cce-aicore-record-overflow=false", "-mllvm", "-cce-aicore-mask-opt=false",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O2", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-m200-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-fp-ceiling=2",
        "-mllvm", "-cce-aicore-record-overflow=false", "-mllvm", "-cce-aicore-mask-opt=false",
        "-D__ENABLE_VECTOR_CORE__",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom",
    };
    EXPECT_EQ(exec_res, 0);
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));

    manager.SetShortSocVersion(AscPlugin::ShortSocVersion::ASCEND910B);
}

TEST_F(TestAscInterFace, asc_PluginEpilogue_optimize_lv1)
{
    AscPlugin::PrologueResult res2;
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.compileArgs = {"-DASCENDC_DUMP=0", "-O1"};
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* prologueRes;
    int32_t exec_res1 = AscPlugin::PluginPrologue(&prologueRes, configObjPtr);
    free(const_cast<char*>(prologueRes));
    free(const_cast<char*>(configObjPtr));

    AscPlugin::EpilogueResult res;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O2",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O2", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O2", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TestAscInterFace, asc_PluginEpilogue_user_no_dump)
{
    AscPlugin::PrologueResult res2;
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.compileArgs = {"-DASCENDC_DUMP=0", "--cce-aicore-input-parameter-size=50"};
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* prologueRes;
    int32_t exec_res1 = AscPlugin::PluginPrologue(&prologueRes, configObjPtr);
    free(const_cast<char*>(prologueRes));
    free(const_cast<char*>(configObjPtr));

    AscPlugin::EpilogueResult res;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TestAscInterFace, asc_PluginEpilogue_dump_on)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TestAscInterFace, asc_PluginEpilogue_dump_on_assert)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TestAscInterFace, asc_PluginEpilogue_user_dump_on)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.compileArgs = {"-DASCENDC_DUMP=1"};
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* prologueRes;
    int32_t exec_res1 = AscPlugin::PluginPrologue(&prologueRes, configObjPtr);
    free(const_cast<char*>(prologueRes));
    free(const_cast<char*>(configObjPtr));

    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    const char* a;
    MOCKER(AscPlugin::FunctionRegistryImpl).stubs().will(returnValue(std::string("")));
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    std::vector<std::string> expHostExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3",
        // define macros
        "-D__NPU_HOST__", "-DTILING_KEY_VAR=0",
    };
    std::vector<std::string> expDeviceCubeExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-cube",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-D__ENABLE_ASCENDC_PRINTF__",
        "--cce-aicore-arch=dav-c220-vec",
        "--cce-auto-sync",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TestAscInterFace, asc_PluginFatbinLink)
{
    AscPlugin::FatbinLinkResult res;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    const char* a;
    int32_t exec_res = AscPlugin::PluginFatbinLink(&a);
    std::vector<std::string> expLinkOptions = {
        "-lascendc_runtime", "-lascendcl", "-lruntime", "-lerror_manager", "-lprofapi", "-lunified_dlog", "-lmmpa", "-lascend_dump", "-lc_sec", "-lstdc++",
        "-LA/lib64",
        "-LA/tools/simulator/Ascend910B1/lib"
    };

    nlohmann::json jsonObj;
    jsonObj["BinaryRegisterCode"] = registerBinary;
    jsonObj["ExtraFatbinHostLinkOptions"] = expLinkOptions;
    const std::string jsonStr = jsonObj.dump();
    const char* expectedRes = jsonStr.c_str();
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
}

TEST_F(TestAscInterFace, asc_PluginFatbinLink_nullptr)
{
    int32_t exec_res = AscPlugin::PluginFatbinLink(nullptr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_NULLPTR);
}

TEST_F(TestAscInterFace, asc_PluginFatbinLink_null_soc)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.socVersion_ = "";

    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.npuSoc = "";
    config.compileArgs = {"-O2"};
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* prologueRes;
    int32_t exec_res = AscPlugin::PluginPrologue(&prologueRes, configObjPtr);
    free(const_cast<char*>(prologueRes));
    free(const_cast<char*>(configObjPtr));

    AscPlugin::FatbinLinkResult res2;
    const char* a;
    int32_t exec_res2 = AscPlugin::PluginFatbinLink(&a);
    std::vector<std::string> expLinkOptions = {
        "-lascendc_runtime", "-lascendcl", "-lruntime", "-lerror_manager", "-lprofapi", "-lunified_dlog", "-lmmpa", "-lascend_dump", "-lc_sec", "-lstdc++",
        "-LA/lib64"
    };
    EXPECT_EQ(exec_res, AscPlugin::ASC_SUCCESS);
    EXPECT_EQ(exec_res2, AscPlugin::ASC_SUCCESS);
    nlohmann::json jsonObj;
    jsonObj["BinaryRegisterCode"] = registerBinary;
    jsonObj["ExtraFatbinHostLinkOptions"] = expLinkOptions;
    const std::string jsonStr = jsonObj.dump();
    const char* expectedRes = jsonStr.c_str();
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
}

TEST_F(TestAscInterFace, asc_PluginFatbinLink_910B4_1)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    config.npuSoc = "Ascend910B4-1";
    config.compileArgs = {"-O2"};
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* prologueRes;
    int32_t exec_res = AscPlugin::PluginPrologue(&prologueRes, configObjPtr);
    free(const_cast<char*>(prologueRes));
    free(const_cast<char*>(configObjPtr));

    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    AscPlugin::FatbinLinkResult res2;
    const char* a;
    int32_t exec_res2 = AscPlugin::PluginFatbinLink(&a);
    std::vector<std::string> expLinkOptions = {
        "-lascendc_runtime", "-lascendcl", "-lruntime", "-lerror_manager", "-lprofapi", "-lunified_dlog", "-lmmpa", "-lascend_dump", "-lc_sec", "-lstdc++",
        "-LA/lib64",
        "-LA/tools/simulator/Ascend910B4/lib"
    };
    EXPECT_EQ(exec_res, AscPlugin::ASC_SUCCESS);
    EXPECT_EQ(exec_res2, AscPlugin::ASC_SUCCESS);
    nlohmann::json jsonObj;
    jsonObj["BinaryRegisterCode"] = registerBinary;
    jsonObj["ExtraFatbinHostLinkOptions"] = expLinkOptions;
    const std::string jsonStr = jsonObj.dump();
    const char* expectedRes = jsonStr.c_str();
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(std::string(a), std::string(expectedRes));
    free(const_cast<char*>(a));
    manager.saveTempRequested_ = false;
    system("rm -rf /tmp/asc_plugin");
}

TEST_F(TestAscInterFace, asc_PluginPreCompile)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";

    const char* a;
    int32_t exec_res = AscPlugin::PluginGetPreCompileOpts(&a);

    struct utsname info;
    std::string prefix;
    if (uname(&info) < 0) {
        prefix = "/compiler";
    } else {
        std::string machine = info.machine;
        if (machine == "x86_64") {
            prefix = "/x86_64-linux";
        } else if (machine == "aarch64" || machine == "arm64" || machine == "arm") {
            prefix = "/aarch64-linux";
        } else {
            prefix = "/compiler";
        }
    }

    std::vector<std::string> hostOptions = {
        "-std=c++17",
        "-IA" + prefix + "/include", "-IA" + prefix + "/include/ascendc/host_api",
        "-IA" + prefix + "/ascendc/include/highlevel_api", "-IA" + prefix + "/tikcpp/tikcfw",
        "-IA" + prefix + "/tikcpp/tikcfw/lib", "-IA" + prefix + "/tikcpp/tikcfw/lib/matmul",
        "-IA" + prefix + "/tikcpp/tikcfw/impl", "-IA" + prefix + "/tikcpp/tikcfw/interface",

        "-IA" + prefix + "/asc/impl/adv_api",
        "-IA" + prefix + "/asc/impl/basic_api",
        "-IA" + prefix + "/asc/impl/c_api",
        "-IA" + prefix + "/asc/impl/micro_api",
        "-IA" + prefix + "/asc/impl/simt_api",
        "-IA" + prefix + "/asc/impl/utils",
        "-IA" + prefix + "/asc",

        "-IA" + prefix + "/asc/include",
        "-IA" + prefix + "/asc/include/adv_api",
        "-IA" + prefix + "/asc/include/adv_api/matmul",
        "-IA" + prefix + "/asc/include/aicpu_api",
        "-IA" + prefix + "/asc/include/basic_api",
        "-IA" + prefix + "/asc/include/c_api",
        "-IA" + prefix + "/asc/include/interface",
        "-IA" + prefix + "/asc/include/micro_api",
        "-IA" + prefix + "/asc/include/simt_api",
        "-IA" + prefix + "/asc/include/tiling",
        "-IA" + prefix + "/asc/include/utils"
    };
    AscPlugin::PreCompileOptsResult configInfo = nlohmann::json::parse(a).get<AscPlugin::PreCompileOptsResult>();
    EXPECT_EQ(configInfo.compileOptions, hostOptions);
    free(const_cast<char*>(a));
}

TEST_F(TestAscInterFace, asc_PluginPreCompile_nullptr)
{
    auto res = AscPlugin::PluginGetPreCompileOpts(nullptr);
    EXPECT_EQ(res, AscPlugin::ASC_NULLPTR);
}
