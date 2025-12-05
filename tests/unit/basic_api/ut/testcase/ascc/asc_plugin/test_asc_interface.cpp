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
#include "asc_dev_funcRegistry_generate.h"

static std::string registerBinary(R"(#include <stdio.h>
#include <stdint.h>
#include <vector>

namespace AscPluginGenerator {
constexpr unsigned int ascendcExceptionDumpHead = 2U;
typedef struct rtArgsSizeInfoAsc {
    void *infoAddr;
    uint32_t atomicIndex;
} rtArgsSizeInfoAsc_t;
} // namespace AscPluginGenerator

extern "C" {
int32_t rtSetExceptionExtInfo(const AscPluginGenerator::rtArgsSizeInfoAsc_t * const sizeInfo);
namespace Adx {
void *AdumpGetSizeInfoAddr(uint32_t space, uint32_t &atomicIndex);
} // namespace Adx
}

namespace AscPluginGenerator {
__attribute__ ((visibility("hidden"))) uint32_t ascendc_set_exception_dump_info(uint32_t dumpSize)
{
    uint32_t atomicIndex = 0U;
    constexpr uint32_t addrNum = 1U;
    void *exceptionDumpAddr = Adx::AdumpGetSizeInfoAddr(addrNum + ascendcExceptionDumpHead, atomicIndex);
    if (exceptionDumpAddr == nullptr) {
        ::printf("[ERROR] [AscPlugin] Get exceptionDumpAddr is nullptr.\n");
        return 1;
    }
    uint64_t *sizeInfoAddr = reinterpret_cast<uint64_t *>(exceptionDumpAddr);
    *sizeInfoAddr = static_cast<uint64_t>(atomicIndex);
    sizeInfoAddr++;
    *sizeInfoAddr = static_cast<uint64_t>(1);
    sizeInfoAddr++;
    *sizeInfoAddr = dumpSize * 75;
    constexpr uint64_t workspaceOffset = (4ULL << 56ULL);
    *sizeInfoAddr |= workspaceOffset;
    const rtArgsSizeInfoAsc sizeInfo = {exceptionDumpAddr, atomicIndex};
    int32_t ret = rtSetExceptionExtInfo(&sizeInfo);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] rtSetExceptionExtInfo failed, ret = %d.\n", ret);
        return 1;
    }
    return 0;
}
} // namespace AscPluginGenerator

extern "C" {
int32_t AscendDevBinaryRegister(const void *fileBuf, size_t fileSize, void **handle);
int32_t AscendKernelLaunchWithFlagV2(const char *stubFunc, const uint32_t blockDim, void **args,
    uint32_t size, const void *stream, const uint32_t ubufDynamicSize);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
void AscendProfRegister();
}

namespace {
char ascendcErrMsg[4096] = {0};
void *g_kernel_handle = nullptr;

typedef void (*KernelFuncRegister)(void*);

class AscPluginRegFuncRegister {
public:
    inline static AscPluginRegFuncRegister& GetInstance()
    {
        static AscPluginRegFuncRegister instance;
        return instance;
    }

public:
    std::vector<KernelFuncRegister> regFuncCallbackList;
private:
    AscPluginRegFuncRegister() = default;
    ~AscPluginRegFuncRegister() = default;
    AscPluginRegFuncRegister(const AscPluginRegFuncRegister&) = delete;
    AscPluginRegFuncRegister& operator=(const AscPluginRegFuncRegister&) = delete;
    AscPluginRegFuncRegister(AscPluginRegFuncRegister&&) = delete;
    AscPluginRegFuncRegister& operator=(AscPluginRegFuncRegister&&) = delete;
};

void RegisterKernels(void)
{
    int32_t ret;
    ret = AscendDevBinaryRegister(fatbinDataPtr, fatbinDataLength, &g_kernel_handle);
    if (ret != 0) {
        ::printf("[ERROR] [AscPlugin] Kernel binary register failure! ret %d \n", ret);
    }
    AscendProfRegister();
}

class KernelHandleGradUnregister {
private:
    KernelHandleGradUnregister() = default;
    ~KernelHandleGradUnregister() {
        if (g_kernel_handle) {
            UnregisterAscendBinary(g_kernel_handle);
            g_kernel_handle = nullptr;
        }
    }
    KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;
    KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;
public:
    static KernelHandleGradUnregister& GetInstance() {
        static KernelHandleGradUnregister instance;
        return instance;
    }
};

class AscendCOperatorRegister {
public:
static AscendCOperatorRegister& GetInstance() {
    static AscendCOperatorRegister instance;
    return instance;
}
private:
AscendCOperatorRegister() {
    RegisterKernels();
    const auto& inst = AscPluginRegFuncRegister::GetInstance();
    for (auto func : inst.regFuncCallbackList) {
        func(g_kernel_handle);
    }
}
~AscendCOperatorRegister() = default;
AscendCOperatorRegister(const AscendCOperatorRegister&) = delete;
AscendCOperatorRegister& operator=(const AscendCOperatorRegister&) = delete;
};

} // namespace

namespace AscPluginGenerator {
__attribute__ ((visibility("hidden"))) int32_t BindKernelRegisterFunc(KernelFuncRegister func)
{
    auto& inst = AscPluginRegFuncRegister::GetInstance();
    inst.regFuncCallbackList.emplace_back(func);
    return 0;
}

__attribute__ ((visibility("hidden"))) void GetHandleUnregisterInst() {
    auto& regMng = KernelHandleGradUnregister::GetInstance();
}

__attribute__ ((visibility("hidden"))) uint32_t LaunchAndProfiling(const char *stubFunc, uint32_t blockDim,
    void *stream, void **args, uint32_t size, uint32_t ktype, const uint32_t ubufDynamicSize)
{
    const auto& reg = AscendCOperatorRegister::GetInstance();
    uint64_t startTime;
    const char *name = stubFunc;
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle == nullptr) {
        ::printf("[ERROR] [AscPlugin] %s\n", ascendcErrMsg);
        return 1;
    }
    int32_t retLaunch = AscendKernelLaunchWithFlagV2(stubFunc, blockDim, args, size, stream, ubufDynamicSize);
    if (retLaunch != 0) {
        ::printf("[ERROR] [AscPlugin] AscendKernelLaunchWithFlagV2 ret %u\n", retLaunch);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, ktype, startTime);
    }
    return retLaunch;
}

} // namespace AscPluginGenerator
)");

class TEST_ASC_INTERFACE : public testing::Test {
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
TEST_F(TEST_ASC_INTERFACE, asc_cannpath_init)
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
    manager.SetCannPath("/usr/local/Ascend/latest");
}

TEST_F(TEST_ASC_INTERFACE, asc_plugin_gen_kernel_failure_with_no_kernel)
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

TEST_F(TEST_ASC_INTERFACE, asc_plugin_gen_kernel_failure_with_no_mangling)
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

TEST_F(TEST_ASC_INTERFACE, asc_plugin_gen_kernel_failure_with_nullptr)
{
    const char* a;
    auto res = AscPlugin::PluginGenKernel(&a, nullptr);
    EXPECT_EQ(res, AscPlugin::ASC_NULLPTR);
}

TEST_F(TEST_ASC_INTERFACE, asc_plugin_gen_kernel_success)
{
    std::string golden = R"(namespace Foo1::Foo2 {
extern "C" __global__ __aicore__ void __device_stub__mangling_add(__attribute__((cce_global)) uint8_t * __ascendc_overflow_status)
{
    icache_preload(1);
    if (g_sysFftsAddr != nullptr) {
        set_ffts_base_addr((uint64_t)g_sysFftsAddr);
    }
    __origin__add();
}
}
)";
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_saveTemp)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_saveTemp_logFolderPath_fail)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_saveTemp_tmpFolderPath_fail)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_saveTemp_logPathFolderNotExist)
{
    AscPlugin::PrologueConfig config;
    config.saveTemp = true;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.logPath = "/tmp/ascc_plugin/test/log";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB"};

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    system("rm -rf /tmp/ascc_plugin/test/log");
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_saveTemp_tmpPathFolderNotExist)
{
    AscPlugin::PrologueConfig config;
    config.saveTemp = true;
    config.verbose = false;
    config.npuSoc = "Ascend910B1";
    config.npuArch = "dav-c220";
    config.source = "a.cpp";
    config.tmpPath = "/tmp/ascc_plugin/test/temp";
    config.binaryPtrName = "aiv_buf";
    config.binaryLenName = "aiv_file_len";
    config.genMode = AscPlugin::GenMode::AICORE_ONLY;
    config.compileArgs = {"-DABB"};

    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    system("rm -rf /tmp/ascc_plugin/test/temp");
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_saveTemp_unDefaultFolderPath)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_saveTemp_GenerateTimestamp_fail)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_error_npuArch)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_empty_compileArgs)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_error_soc)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_null_soc)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_ast_failed)
{
    AscPlugin::PrologueConfig config;
    PrologueConfigUpdate(config);
    nlohmann::json configObj= config;
    const char* configObjPtr = strdup(configObj.dump().c_str());
    const char* a;
    MOCKER(&AscPlugin::AscAstDeviceAnalyzer::Process).stubs().will(returnValue(AscPlugin::ASC_FAILURE));
    int32_t exec_res = AscPlugin::PluginPrologue(&a, configObjPtr);
    auto& manager = AscPlugin::InfoManager::GetInstance();
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
    free(const_cast<char*>(configObjPtr));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_binary_register_code_failed)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPrologue_nullptr)
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

// Not support MIX_1_1 and MIX_1_2 at same time, will return erorr
TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_WithMix11_Mix12)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    manager.kernelFuncSymbolToFuncInfo_ = {};
    manager.AddGlobalSymbolInfo("__device_stub__add_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_1, "a.cpp", 100, 110, AscPlugin::KfcScene::Open);
    manager.AddGlobalSymbolInfo("__device_stub__sub_custom", AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2, "b.cpp", 120, 110, AscPlugin::KfcScene::Open);
    const char* a;
    int32_t exec_res = AscPlugin::PluginEpilogue(&a);
    EXPECT_EQ(exec_res, AscPlugin::ASC_FAILURE);
}

// MIX_1_2 with MIX_AIV_only is ok. Because has MIX, has -D__MIX_CORE_MACRO__=1, has -DA=A_mix_aiv/aic
TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_WithMixAIV_Mix12)
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "-D__MIX_CORE_MACRO__=1",
        "--cce-auto-sync",
        "--cce-aicore-arch=dav-c220-cube",
        "-D__device_stub__sub_custom=sub_custom_mix_aic",
        "-D__device_stub__add_custom=add_custom_mix_aic",
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "-D__MIX_CORE_MACRO__=1",
        "--cce-auto-sync",
        "--cce-aicore-arch=dav-c220-vec",
        "-D__device_stub__sub_custom=sub_custom_mix_aiv",
        "-D__device_stub__add_custom=add_custom_mix_aiv"
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

// MIX_1_2 with MIX_AIC_only is ok. Because has MIX, has -D__MIX_CORE_MACRO__=1.
// Because has MIX_1_1, has -D__MIX_CORE_AIC_RATION__=1
TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_WithMixAIC_Mix11)
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "-D__MIX_CORE_MACRO__=1", "-D__MIX_CORE_AIC_RATION__=1",
        "--cce-auto-sync",
        "--cce-aicore-arch=dav-c220-cube",
        "-D__device_stub__sub_custom=sub_custom_mix_aic",
        "-D__device_stub__add_custom=add_custom_mix_aic"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "-D__MIX_CORE_MACRO__=1", "-D__MIX_CORE_AIC_RATION__=1",
        "--cce-auto-sync",
        "--cce-aicore-arch=dav-c220-vec",
        "-D__device_stub__sub_custom=sub_custom_mix_aiv",
        "-D__device_stub__add_custom=add_custom_mix_aiv"
    };

    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

// no function call, still pass device compile options
TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_no_device_call)
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "--cce-aicore-arch=dav-c220-cube"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "--cce-aicore-arch=dav-c220-vec"
    };

    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

// when not has mix, do not need extra compile options
// AIC_ONLY do not need -DA=A_mix_aic/aiv
TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue)
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "--cce-aicore-arch=dav-c220-cube",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "--cce-aicore-arch=dav-c220-vec",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_nullptr)
{
    int32_t exec_res = AscPlugin::PluginEpilogue(nullptr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_NULLPTR);
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_no_dump)
{
    const char* a;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    manager.userDumpStatus_ = false;
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        // include paths
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=0",
        "--cce-aicore-arch=dav-c220-cube",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000", "-mllvm", "-cce-aicore-function-stack-size=0x8000",

        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        // include paths
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=0",
        "--cce-aicore-arch=dav-c220-vec",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_310P)
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
    manager.userDumpStatus_ = false;
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
        "-mllvm", "-cce-aicore-fp-ceiling=2",
        "-mllvm", "-cce-aicore-record-overflow=false", "-mllvm", "-cce-aicore-mask-opt=false",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=0",
        "--cce-aicore-arch=dav-m200",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O2", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-fp-ceiling=2",
        "-mllvm", "-cce-aicore-record-overflow=false", "-mllvm", "-cce-aicore-mask-opt=false",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=0",
        "--cce-aicore-arch=dav-m200-vec",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom",
        "-D__ENABLE_VECTOR_CORE__"
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_optimize_lv1)
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
    manager.userDumpStatus_ = false;
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=0",
        "--cce-aicore-arch=dav-c220-cube",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O2", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=0",
        "--cce-aicore-arch=dav-c220-vec",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_user_no_dump)
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
    manager.userDumpStatus_ = false;
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=0",
        "--cce-aicore-arch=dav-c220-cube",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=0",
        "--cce-aicore-arch=dav-c220-vec",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_dump_on)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    manager.userDumpStatus_ = true;
    manager.hasPrintf_ = true;
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=1",
        "-DONE_CORE_DUMP_SIZE=1048576",
        "--cce-aicore-arch=dav-c220-cube",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=1",
        "-DONE_CORE_DUMP_SIZE=1048576",
        "--cce-aicore-arch=dav-c220-vec",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_dump_on_assert)
{
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    manager.userDumpStatus_ = true;
    manager.hasPrintf_ = false;
    manager.hasAssert_ = true;
    manager.UpdateOneCoreDumpSize();
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=1",
        "-DONE_CORE_DUMP_SIZE=1024",
        "--cce-aicore-arch=dav-c220-cube",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=1",
        "-DONE_CORE_DUMP_SIZE=1024",
        "--cce-aicore-arch=dav-c220-vec",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginEpilogue_user_dump_on)
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
    manager.userDumpStatus_ = true;
    manager.hasPrintf_ = true;
    manager.UpdateOneCoreDumpSize();
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
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=1",
        "-DONE_CORE_DUMP_SIZE=1048576",
        "--cce-aicore-arch=dav-c220-cube",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };

    std::vector<std::string> expDeviceVecExtraCompileOptions = {
        // compile options
        "-std=c++17", "-O3", "-D__NPU_DEVICE__", "-DTILING_KEY_VAR=0",
        "-mllvm", "-cce-aicore-stack-size=0x8000",
        "-mllvm", "-cce-aicore-function-stack-size=0x8000",
        "-mllvm", "-cce-aicore-dcci-insert-for-scalar=false",
        "-DL2_CACHE_HINT",
        "--cce-auto-sync",
        "-DASCENDC_DUMP=1",
        "-DONE_CORE_DUMP_SIZE=1048576",
        "--cce-aicore-arch=dav-c220-vec",
        "-D__device_stub__sub_custom=sub_custom",
        "-D__device_stub__add_custom=add_custom"
    };
    AscPlugin::EpilogueResult expectRes = {"", expHostExtraCompileOptions, expDeviceCubeExtraCompileOptions,
        expDeviceVecExtraCompileOptions};
    nlohmann::json jsonObj = expectRes;
    const char* expectedRes = strdup(jsonObj.dump().c_str());
    EXPECT_EQ(exec_res, 0);
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
    free(const_cast<char*>(expectedRes));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginFatbinLink)
{
    AscPlugin::FatbinLinkResult res;
    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetCannPath("A");
    manager.optimizeLevel_ = "-O3";
    const char* a;
    int32_t exec_res = AscPlugin::PluginFatbinLink(&a);
    std::vector<std::string> expLinkOptions = {
        "-lascendc_runtime", "-lascendcl", "-lruntime", "-lerror_manager", "-lprofapi", "-lascendalog", "-lmmpa", "-lascend_dump", "-lc_sec", "-lstdc++",
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginFatbinLink_nullptr)
{
    int32_t exec_res = AscPlugin::PluginFatbinLink(nullptr);
    EXPECT_EQ(exec_res, AscPlugin::ASC_NULLPTR);
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginFatbinLink_null_soc)
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
        "-lascendc_runtime", "-lascendcl", "-lruntime", "-lerror_manager", "-lprofapi", "-lascendalog", "-lmmpa", "-lascend_dump", "-lc_sec", "-lstdc++",
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
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginFatbinLink_910B4_1)
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
        "-lascendc_runtime", "-lascendcl", "-lruntime", "-lerror_manager", "-lprofapi", "-lascendalog", "-lmmpa", "-lascend_dump", "-lc_sec", "-lstdc++",
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
    EXPECT_EQ(strcmp(expectedRes, a), 0);
    free(const_cast<char*>(a));
    manager.saveTempRequested_ = false;
    system("rm -rf /tmp/asc_plugin");
}

TEST_F(TEST_ASC_INTERFACE, asc_PluginPreCompile)
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

TEST_F(TEST_ASC_INTERFACE, asc_PluginPreCompile_nullptr)
{
    auto res = AscPlugin::PluginGetPreCompileOpts(nullptr);
    EXPECT_EQ(res, AscPlugin::ASC_NULLPTR);
}
