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
#include <string>
#include <iostream>
#include <gtest/gtest.h>
#include <unistd.h>
#include <mockcpp/mockcpp.hpp>
#include <unordered_set>
#define private public
#include "asc_info_manager.h"
#include "asc_host_code_generate.h"
#include "asc_host_stub_generator.h"

class TEST_ASC_HOST_STUB_GENERATOR : public testing::Test {
protected:
    void SetUp() {}
    void TearDown()
    {
        GlobalMockObject::verify();
    }
};

TEST_F(TEST_ASC_HOST_STUB_GENERATOR, asc_get_host_stub_namespace)
{
    // template/add_custom_namespace
    std::string golden = R"(template<typename T, typename U> void AscendC::AOT::Kernel::add_custom(uint32_t __ascendc_blockDim, void* __ascendc_hold, void* __ascendc_stream, uint8_t * x, uint8_t * y, uint8_t * z)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* z;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args {x, y, z, };
    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    const char* __ascendc_name = "add_custom";
    const char* __ascendc_manglingName = nullptr;
    uint32_t __ascendc_kType = 10;
    if constexpr (AscendC::Std::is_same<T, short>::value && AscendC::Std::is_same<U, short>::value) {
        __ascendc_manglingName = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
        __ascendc_kType = 10;
    }
    if (__ascendc_manglingName == nullptr) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "call kernel function failure!");
        return;
    }
    __ascendc_ret = AscPluginGenerator::LaunchAndProfiling(__ascendc_manglingName, __ascendc_blockDim, __ascendc_stream, (void **)&__ascendc_args, sizeof(__ascendc_args), __ascendc_kType);
    if(__ascendc_ret != 0) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "kernel launch failure!");
        return;
    }
    AscPluginGenerator::GetHandleUnregisterInst();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
}
)";
    AscPlugin::KernelInfo kernelInfo;
    kernelInfo.kernelName = "add_custom";
    kernelInfo.kernelMangledName = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
    kernelInfo.kernelMangledNameConsiderPrefix = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
    kernelInfo.namespaces = {"AscendC", "AOT", "Kernel"};
    kernelInfo.kernelParameters = {
        {"uint8_t *", "x", false, "", ""},
        {"uint8_t *", "y", false, "", ""},
        {"uint8_t *", "z", false, "", ""}
    };
    kernelInfo.kernelAttributes = {};
    kernelInfo.isTemplate = true;
    kernelInfo.templateParameters = {
        {"typename", "T", false, "", "", AscPlugin::ParamType::TEMPLATE_TYPE},
        {"typename", "U", false, "", "", AscPlugin::ParamType::TEMPLATE_TYPE}
    };
    kernelInfo.templateInstances = {
        {
            {"short", "short"},
            {
                {"uint8_t *", "x", false, "", ""},
                {"uint8_t *", "y", false, "", ""},
                {"uint8_t *", "z", false, "", ""}
            },
            "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_",
            "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_",
            {true, 1, 2}
        }
    };

    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetHasPrintf(false);
    manager.SetHasAssert(false);
    manager.SetUserDumpStatus(false);
    std::unordered_set<AscPlugin::KernelMetaType> kernelType = {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2};
    auto hostStubCode = GetHostStubCode(kernelInfo, kernelType);
    EXPECT_EQ(hostStubCode, golden);
}


TEST_F(TEST_ASC_HOST_STUB_GENERATOR, asc_get_host_stub_namespace_mix_aiv_1_0)
{
    // template/add_custom_namespace
    std::string golden = R"(template<typename T, typename U> void AscendC::AOT::Kernel::add_custom(uint32_t __ascendc_blockDim, void* __ascendc_hold, void* __ascendc_stream, uint8_t * x, uint8_t * y, uint8_t * z)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* z;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args {x, y, z, };
    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    const char* __ascendc_name = "add_custom";
    const char* __ascendc_manglingName = nullptr;
    uint32_t __ascendc_kType = 7;
    if constexpr (AscendC::Std::is_same<T, short>::value && AscendC::Std::is_same<U, short>::value) {
        __ascendc_manglingName = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
        __ascendc_kType = 7;
    }
    if (__ascendc_manglingName == nullptr) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "call kernel function failure!");
        return;
    }
    __ascendc_ret = AscPluginGenerator::LaunchAndProfiling(__ascendc_manglingName, __ascendc_blockDim, __ascendc_stream, (void **)&__ascendc_args, sizeof(__ascendc_args), __ascendc_kType);
    if(__ascendc_ret != 0) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "kernel launch failure!");
        return;
    }
    AscPluginGenerator::GetHandleUnregisterInst();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
}
)";
    AscPlugin::KernelInfo kernelInfo;
    kernelInfo.kernelName = "add_custom";
    kernelInfo.kernelMangledName = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
    kernelInfo.kernelMangledNameConsiderPrefix = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
    kernelInfo.namespaces = {"AscendC", "AOT", "Kernel"};
    kernelInfo.kernelParameters = {
        {"uint8_t *", "x", false, "", ""},
        {"uint8_t *", "y", false, "", ""},
        {"uint8_t *", "z", false, "", ""}
    };
    kernelInfo.kernelAttributes = {};
    kernelInfo.isTemplate = true;
    kernelInfo.templateParameters = {
        {"typename", "T", false, "", "", AscPlugin::ParamType::TEMPLATE_TYPE},
        {"typename", "U", false, "", "", AscPlugin::ParamType::TEMPLATE_TYPE}
    };
    kernelInfo.templateInstances = {
        {
            {"short", "short"},
            {
                {"uint8_t *", "x", false, "", ""},
                {"uint8_t *", "y", false, "", ""},
                {"uint8_t *", "z", false, "", ""}
            },
            "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_",
            "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_",
            {true, 0, 1}
        }
    };

    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetHasPrintf(false);
    manager.SetHasAssert(false);
    manager.SetUserDumpStatus(false);
    std::unordered_set<AscPlugin::KernelMetaType> kernelType = {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIV_1_0};
    auto hostStubCode = GetHostStubCode(kernelInfo, kernelType);
    EXPECT_EQ(hostStubCode, golden);
}


TEST_F(TEST_ASC_HOST_STUB_GENERATOR, asc_get_host_stub_namespace_mix_aic_1_0)
{
    // template/add_custom_namespace
    std::string golden = R"(template<typename T, typename U> void AscendC::AOT::Kernel::add_custom(uint32_t __ascendc_blockDim, void* __ascendc_hold, void* __ascendc_stream, uint8_t * x, uint8_t * y, uint8_t * z)
{
    struct {
        alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* z;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args {x, y, z, };
    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    const char* __ascendc_name = "add_custom";
    const char* __ascendc_manglingName = nullptr;
    uint32_t __ascendc_kType = 8;
    if constexpr (AscendC::Std::is_same<T, short>::value && AscendC::Std::is_same<U, short>::value) {
        __ascendc_manglingName = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
        __ascendc_kType = 8;
    }
    if (__ascendc_manglingName == nullptr) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "call kernel function failure!");
        return;
    }
    __ascendc_ret = AscPluginGenerator::LaunchAndProfiling(__ascendc_manglingName, __ascendc_blockDim, __ascendc_stream, (void **)&__ascendc_args, sizeof(__ascendc_args), __ascendc_kType);
    if(__ascendc_ret != 0) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "kernel launch failure!");
        return;
    }
    AscPluginGenerator::GetHandleUnregisterInst();
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
}
)";
    AscPlugin::KernelInfo kernelInfo;
    kernelInfo.kernelName = "add_custom";
    kernelInfo.kernelMangledName = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
    kernelInfo.kernelMangledNameConsiderPrefix = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
    kernelInfo.namespaces = {"AscendC", "AOT", "Kernel"};
    kernelInfo.kernelParameters = {
        {"uint8_t *", "x", false, "", ""},
        {"uint8_t *", "y", false, "", ""},
        {"uint8_t *", "z", false, "", ""}
    };
    kernelInfo.kernelAttributes = {};
    kernelInfo.isTemplate = true;
    kernelInfo.templateParameters = {
        {"typename", "T", false, "", "", AscPlugin::ParamType::TEMPLATE_TYPE},
        {"typename", "U", false, "", "", AscPlugin::ParamType::TEMPLATE_TYPE}
    };
    kernelInfo.templateInstances = {
        {
            {"short", "short"},
            {
                {"uint8_t *", "x", false, "", ""},
                {"uint8_t *", "y", false, "", ""},
                {"uint8_t *", "z", false, "", ""}
            },
            "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_",
            "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_",
            {true, 1, 0}
        }
    };

    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetHasPrintf(false);
    manager.SetHasAssert(false);
    manager.SetUserDumpStatus(false);
    std::unordered_set<AscPlugin::KernelMetaType> kernelType = {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0};
    auto hostStubCode = GetHostStubCode(kernelInfo, kernelType);
    EXPECT_EQ(hostStubCode, golden);
}

// TEST_F(TEST_ASC_HOST_STUB_GENERATOR, asc_get_host_stub_bisheng_core_ratio)
// {
//     // template/add_custom_namespace
//     std::string golden = R"(template<int32_t cube, int32_t vec> void AscendC::AOT::Kernel::add_custom(uint32_t __ascendc_blockDim, void* __ascendc_hold, void* __ascendc_stream, uint8_t * x, uint8_t * y, uint8_t * z)
// {
//     struct {
//         alignas(((alignof(void*) + 3) >> 2) << 2) void* x;
//         alignas(((alignof(void*) + 3) >> 2) << 2) void* y;
//         alignas(((alignof(void*) + 3) >> 2) << 2) void* z;
//         alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
//     } __ascendc_args {x, y, z, };
//     uint32_t __ascendc_ret;
//     constexpr uint32_t __ascendc_overflow_status_size = 8;
//     AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
//     const char* __ascendc_name = "add_custom";
//     const char* __ascendc_manglingName = nullptr;
//     uint32_t __ascendc_kType = 10;
//     if constexpr (AscendC::Std::is_same<cube, 1>::value && AscendC::Std::is_same<vec, 2>::value) {
//         __ascendc_manglingName = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
//         __ascendc_kType = 10;
//     }
//     if constexpr (AscendC::Std::is_same<cube, 1>::value && AscendC::Std::is_same<vec, 0>::value) {
//         __ascendc_manglingName = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S5_";
//         __ascendc_kType = 8;
//     }
//     if (__ascendc_manglingName == nullptr) {
//         ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "call kernel function failure!");
//         return;
//     }
//     __ascendc_ret = AscPluginGenerator::LaunchAndProfiling(__ascendc_manglingName, __ascendc_blockDim, __ascendc_stream, (void **)&__ascendc_args, sizeof(__ascendc_args), __ascendc_kType);
//     if(__ascendc_ret != 0) {
//         ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "kernel launch failure!");
//         return;
//     }
//     AscPluginGenerator::GetHandleUnregisterInst();
//     FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
// }
// )";
//     AscPlugin::KernelInfo kernelInfo;
//     kernelInfo.kernelName = "add_custom";
//     kernelInfo.kernelMangledName = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
//     kernelInfo.kernelMangledNameConsiderPrefix = "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_";
//     kernelInfo.namespaces = {"AscendC", "AOT", "Kernel"};
//     kernelInfo.kernelParameters = {
//         {"uint8_t *", "x", false, "", ""},
//         {"uint8_t *", "y", false, "", ""},
//         {"uint8_t *", "z", false, "", ""}
//     };
//     kernelInfo.kernelAttributes = {};
//     kernelInfo.isTemplate = true;
//     kernelInfo.templateParameters = {
//         {"int32_t", "cube", false, "", "", AscPlugin::ParamType::TEMPLATE_TYPE},
//         {"int32_t", "vec", false, "", "", AscPlugin::ParamType::TEMPLATE_TYPE}
//     };
//     kernelInfo.templateInstances = {
//         {
//             {"1", "2"},
//             {
//                 {"uint8_t *", "x", false, "", ""},
//                 {"uint8_t *", "y", false, "", ""},
//                 {"uint8_t *", "z", false, "", ""}
//             },
//             "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_",
//             "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S3_",
//             {true, 1, 2}
//         },
//         {
//             {"1", "0"},
//             {
//                 {"uint8_t *", "x", false, "", ""},
//                 {"uint8_t *", "y", false, "", ""},
//                 {"uint8_t *", "z", false, "", ""}
//             },
//             "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S5_",
//             "_ZN7AscendC3AOT6Kernel25__device_stub__add_customIssEEvPhS3_S5_",
//             {true, 1, 0}
//         }
//     };

//     auto& manager = AscPlugin::InfoManager::GetInstance();
//     manager.SetHasPrintf(false);
//     manager.SetHasAssert(false);
//     manager.SetUserDumpStatus(false);
//     std::unordered_set<AscPlugin::KernelMetaType> kernelType = {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2,
//         AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_0};
//     auto hostStubCode = GetHostStubCode(kernelInfo, kernelType);
//     EXPECT_EQ(hostStubCode, golden);
// }

TEST_F(TEST_ASC_HOST_STUB_GENERATOR, asc_get_host_stub_printf)
{
    // single_src/hello_world
    std::string golden = R"(void hello_world(uint32_t __ascendc_blockDim, void* __ascendc_hold, void* __ascendc_stream)
{
    struct {
        void* __ascendc_dump;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args {nullptr, };
    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_one_core_dump_size = 1048576;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * 75);
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    const char* __ascendc_name = "hello_world";
    const char* __ascendc_manglingName = nullptr;
    uint32_t __ascendc_kType = 10;
    __ascendc_manglingName = "_Z26__device_stub__hello_worldv";
    if (__ascendc_manglingName == nullptr) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "call kernel function failure!");
        return;
    }
    __ascendc_ret = AscPluginGenerator::LaunchAndProfiling(__ascendc_manglingName, __ascendc_blockDim, __ascendc_stream, (void **)&__ascendc_args, sizeof(__ascendc_args), __ascendc_kType);
    if(__ascendc_ret != 0) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "kernel launch failure!");
        return;
    }
    AscPluginGenerator::GetHandleUnregisterInst();
    Adx::AdumpPrintWorkSpace(__ascendc_args.__ascendc_dump, __ascendc_one_core_dump_size * 75, __ascendc_stream, __ascendc_name);
    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
}
)";
    AscPlugin::KernelInfo kernelInfo;
    kernelInfo.kernelName = "hello_world";
    kernelInfo.kernelMangledName = "_Z26__device_stub__hello_worldv";
    kernelInfo.kernelMangledNameConsiderPrefix = "_Z26__device_stub__hello_worldv";
    kernelInfo.namespaces = {};
    kernelInfo.kernelParameters = {};
    kernelInfo.kernelAttributes = {};
    kernelInfo.isTemplate = false;
    kernelInfo.templateParameters = {};
    kernelInfo.templateInstances = {};

    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetHasPrintf(true);
    manager.SetHasAssert(false);
    manager.SetUserDumpStatus(true);
    std::unordered_set<AscPlugin::KernelMetaType> kernelType = {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2};
    auto hostStubCode = GetHostStubCode(kernelInfo, kernelType);
    EXPECT_EQ(hostStubCode, golden);
}

TEST_F(TEST_ASC_HOST_STUB_GENERATOR, asc_get_host_stub_template)
{
    // single_src/hello_world
    std::string golden = R"(template <template<typename , typename > class I>
struct __AsccIsMyType0 : AscendC::Std::false_type {};
template <>
struct __AsccIsMyType0<MyTempClassA> : AscendC::Std::true_type {};
template <template<typename , typename > class O>
struct __AsccIsMyType1 : AscendC::Std::false_type {};
template <>
struct __AsccIsMyType1<MyTempClassB> : AscendC::Std::true_type {};
template<typename T, int32_t Y, const auto& U, template<typename , typename > class I, template<typename , typename > class O> void Foo1::Foo2::hello_world(uint32_t __ascendc_blockDim, void* __ascendc_hold, void* __ascendc_stream, int i, uint8_t* workspace)
{
    struct {
        void* __ascendc_dump;
        alignas(((alignof(int) + 3) >> 2) << 2) int i;
        alignas(((alignof(uint8_t*) + 3) >> 2) << 2) uint8_t* workspace;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args {nullptr, i, workspace, };
    uint32_t __ascendc_ret;
    constexpr uint32_t __ascendc_one_core_dump_size = 1048576;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * 75);
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    const char* __ascendc_name = "hello_world";
    const char* __ascendc_manglingName = nullptr;
    uint32_t __ascendc_kType = 10;
    if constexpr (AscendC::Std::is_same<T, float>::value && Y == static_cast<int32_t>(7) && &U == &MyPocStruct && __AsccIsMyType0<I>::value && __AsccIsMyType1<O>::value) {
        __ascendc_manglingName = "hello_world_mangling_int_float";
        __ascendc_kType = 10;
    }
    if (__ascendc_manglingName == nullptr) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "call kernel function failure!");
        return;
    }
    __ascendc_ret = AscPluginGenerator::ascendc_set_exception_dump_info(__ascendc_one_core_dump_size);
    if(__ascendc_ret != 0) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "init assert dump failure!");
        return;
    }
    __ascendc_ret = AscPluginGenerator::LaunchAndProfiling(__ascendc_manglingName, __ascendc_blockDim, __ascendc_stream, (void **)&__ascendc_args, sizeof(__ascendc_args), __ascendc_kType);
    if(__ascendc_ret != 0) {
        ASC_PLUGIN_LAUNCH_LOGE(__ascendc_name, __ascendc_stream, __ascendc_blockDim, "kernel launch failure!");
        return;
    }
    AscPluginGenerator::GetHandleUnregisterInst();
    Adx::AdumpPrintWorkSpace(__ascendc_args.__ascendc_dump, __ascendc_one_core_dump_size * 75, __ascendc_stream, __ascendc_name);
    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
}
)";
    AscPlugin::KernelInfo kernelInfo;
    kernelInfo.kernelName = "hello_world";
    kernelInfo.kernelMangledName = "hello_world_mangling";
    kernelInfo.kernelMangledNameConsiderPrefix = "__device_stub__hello_world_mangling";
    kernelInfo.namespaces = {"Foo1", "Foo2"};
    kernelInfo.kernelParameters = {
        {"int", "i", false, "", ""},
        {"uint8_t*", "workspace", false, "", ""}
    };
    kernelInfo.kernelAttributes = {};
    kernelInfo.isTemplate = true;
    kernelInfo.templateParameters = {
        {"typename", "T", false, "", "", AscPlugin::ParamType::TEMPLATE_TYPE},
        {"int32_t", "Y", false, "", "", AscPlugin::ParamType::TEMPLATE_INT},
        {"const auto&", "U", false, "", "", AscPlugin::ParamType::TEMPLATE_DECL},
        {"template<typename , typename > class", "I", false, "", "", AscPlugin::ParamType::TEMPLATE_TEMPLATE},
        {"template<typename , typename > class", "O", false, "", "", AscPlugin::ParamType::TEMPLATE_TEMPLATE}
    };
    kernelInfo.templateInstances = {
        {
            {"float", "7", "MyPocStruct", "MyTempClassA", "MyTempClassB"},
            {
                {"int", "i", false, "", ""},
                {"uint8_t*", "workspace", false, "", ""}
            },
            "hello_world_mangling_int_float",
            "__device_stub__hello_world_mangling_int_float"
        }
    };

    auto& manager = AscPlugin::InfoManager::GetInstance();
    manager.SetHasPrintf(true);
    manager.SetHasAssert(true);
    manager.SetUserDumpStatus(true);
    std::unordered_set<AscPlugin::KernelMetaType> kernelType = {AscPlugin::KernelMetaType::KERNEL_TYPE_MIX_AIC_1_2};
    auto hostStubCode = GetHostStubCode(kernelInfo, kernelType);
    EXPECT_EQ(hostStubCode, golden);
    manager.SetHasAssert(false);
}