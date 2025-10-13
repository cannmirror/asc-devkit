/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file kfc_register_obj.h
 * \brief
 */
#ifndef LIB_KFC_REGISTER_OBJ_H
#define LIB_KFC_REGISTER_OBJ_H
#include "kernel_operator.h"

namespace AscendC {
__aicore__ inline void ClearWorkspace(__gm__  uint8_t *workspace)
{
#if __CCE_AICORE__ == 220
#ifdef __ASCENDC_ENABLE_SUPER_KERNEL__
    SetAtomicNone();
    if ASCEND_IS_AIC {
        SetMaskNorm();
        SetLoadDataBoundary((uint64_t)0);
        SetLoadDataPaddingValue((uint64_t)0);
        ClearWorkspaceImpl(workspace);
        NotifyEvent<PIPE_MTE3>(KFC_SYNC_ID);
    } else {
        SetVectorMask<uint64_t, MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
        SetMaskNorm();
    }
#endif
#endif
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
    SetAtomicNone();
    if ASCEND_IS_AIC {
        SetMaskNorm();
        SetLoadDataPaddingValue((uint64_t)0);
        ClearSSbufImpl();
#if defined(__DAV_310R6__)
        CrossCoreSetFlag<KFC_INTRA_MODE, PIPE_S>(KFC_SYNC_ID);
#else
        CrossCoreSetFlag<KFC_INTRA_MODE, PIPE_S>(KFC_SYNC_ID);
        CrossCoreSetFlag<KFC_INTRA_MODE, PIPE_S>(KFC_SYNC_ID + 16);
#endif
    } else {
        SetVectorMask<uint64_t, MaskMode::NORMAL>((uint64_t)-1, (uint64_t)-1);
        SetMaskNorm();
    }
#endif
}

#if defined(__DAV_310R6__)
    constexpr bool ENABLE_HARD_POOL = true;
#else
    constexpr bool ENABLE_HARD_POOL = false;
#endif

constexpr int8_t CUBEOBJ_MIX_MODE = 2; // 0 no, 1 all, 2 mix
template <class... Args> struct CubeObjs {};

template <class... Args> __aicore__ inline CubeObjs<Args...> GetObjType(Args...); // noexception

template <class... Args> struct GetCubeObjConfig;
template <class T, class... Args> struct GetCubeObjConfig<CubeObjs<T, Args...>> {
    static constexpr int8_t headMixDualMasterValue = GetCubeObjConfig<CubeObjs<T>>::enableMixDualMasterValue;
    static constexpr int8_t headABShareValue = GetCubeObjConfig<CubeObjs<T>>::enableABShareValue;
    static constexpr int8_t tailMixDualMasterValue = GetCubeObjConfig<CubeObjs<Args...>>::enableMixDualMasterValue;
    static constexpr int8_t tailABShareValue = GetCubeObjConfig<CubeObjs<Args...>>::enableABShareValue;

    static constexpr int8_t enableMixDualMasterValue = (headMixDualMasterValue == -1) ?
        tailMixDualMasterValue :
        (tailMixDualMasterValue == -1) ?
        headMixDualMasterValue :
        (headMixDualMasterValue == tailMixDualMasterValue) ? headMixDualMasterValue : CUBEOBJ_MIX_MODE;
    static constexpr int8_t enableABShareValue = (headABShareValue == -1) ?
        tailABShareValue :
        (tailABShareValue == -1) ? headABShareValue : (headABShareValue == tailABShareValue) ? headABShareValue : CUBEOBJ_MIX_MODE;
};
template <class T> struct GetCubeObjConfig<CubeObjs<T>> {
    static constexpr int8_t enableMixDualMasterValue = T::enableMixDualMaster;
    static constexpr int8_t enableABShareValue = T::enableABShare;
};

template <class T> struct GetCubeObjConfig<CubeObjs<T *>> {
    static constexpr int8_t enableMixDualMasterValue = -1;
    static constexpr int8_t enableABShareValue = -1;
};

template <> struct GetCubeObjConfig<CubeObjs<>> {
    static constexpr int8_t enableMixDualMasterValue = -1;
    static constexpr int8_t enableABShareValue = -1;
};

template <class T, class... Args> __aicore__ static T* GetCurTiling(T* t, Args&&... b)
{
    return t;
}

template <class T, class... Args>
__aicore__ inline void InitCurObjSkip(AscendC::TPipe* tpipe, T* a,
    Args&&... b)
{
    InitCurObj(tpipe, b...);
}

template <class T, class... Args>
__aicore__ inline void InitCurObj(AscendC::TPipe* tpipe, T& a, Args&&... b)
{
    ASSERT(tpipe != nullptr && "tpipe cannot be nullptr");
    if constexpr (sizeof...(b) == 0) {
        SetTPipe(a, tpipe);
    } else {
        auto tiling = GetCurTiling(b...);
        a.SetSubBlockIdx(0);
        a.Init(tiling, tpipe);
        if constexpr (sizeof...(b) > 1) {
            InitCurObjSkip(tpipe, b...);
        }
    }
}
}

#ifdef ASCENDC_CPU_DEBUG
#if __CCE_AICORE__ == 220 || defined(__DAV_C310__) || defined(__DAV_310R6__)
#ifdef ASCENDC_CUBE_ONLY
#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    AscendC::InitCurObj(tpipe, __VA_ARGS__)

#else
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
template <class T, class... Args> __aicore__ inline void CountMatmulObj(AscendC::TPipe* tpipe, T &a, Args &&... b);

template <class T, class... Args> __aicore__ inline void CountMatmulObjSkip(AscendC::TPipe* tpipe, T *a, Args &&... b);

template <class T, class... Args> __aicore__ static inline constexpr bool IsTilingObj()
{
    return sizeof(T) == sizeof(void *);
}

template <class T, class... Args> __aicore__ inline void CountMatmulObjSkip(AscendC::TPipe* tpipe, T *a, Args &&... b)
{
    CountMatmulObj(tpipe, b...);
}

template <class T, class... Args> __aicore__ inline void CountMatmulObj(AscendC::TPipe* tpipe, T &a, Args &&... b)
{
    ASSERT(tpipe != nullptr && "tpipe cannot be nullptr");

    if constexpr (sizeof...(b) == 0) {
        g_matmulCount++;
    } else if constexpr (IsTilingObj<Args...>) {
        g_matmulCount++;
        CountMatmulObj(tpipe, b...);
    } else {
        g_matmulCount++;
        if constexpr (sizeof...(b) > 1) {
            CountMatmulObjSkip(tpipe, b...);
        }
    }
    ASSERT(g_matmulCount <= 4 && "cube objs count exceed 4");
}

#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                                            \
    CountMatmulObj(tpipe, __VA_ARGS__);                                                                                   \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>;                        \
    constexpr int8_t enableHardPollKfc = AscendC::ENABLE_HARD_POOL ? AscendC::ENABLE_HARD_POOL                            \
                                        : ASCubeObjConfig::enableABShareValue;                                            \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;                                   \
    AscendC::TQue<AscendC::QuePosition::CO1, 1, &AscendC::gCO1Config> qCO1;                                               \
    if ASCEND_IS_AIC {                                                                                                    \
        if constexpr (!asEnableMixDualMaster) {                                                                           \
            AscendC::ClearWorkspace(reinterpret_cast<__gm__ uint8_t *>(workspace));                                       \
        }                                                                                                                 \
        AscendC::gCO1Que = &qCO1;                                                                                         \
        AscendC::KfcServer<enableHardPollKfc> server;                                                                     \
        server.Init(workspace);                                                                                           \
        server.InitObj(tpipe, __VA_ARGS__);                                                                               \
        if constexpr (!asEnableMixDualMaster) {                                                                           \
            while (server.isRun()) {                                                                                      \
                server.Run(__VA_ARGS__);                                                                                  \
            };                                                                                                            \
            server.Quit();                                                                                                \
            return;                                                                                                       \
        }                                                                                                                 \
    }                                                                                                                     \
    AscendC::KfcCommClient __kfcClient__(workspace, AscendC::GetSubBlockIdx(), enableHardPollKfc, asEnableMixDualMaster); \
    if ASCEND_IS_AIV {                                                                                                    \
        if constexpr (!asEnableMixDualMaster) {                                                                           \
            AscendC::g_kfcClient = &__kfcClient__;                                                                        \
        } else {                                                                                                          \
            AscendC::g_kfcClient = nullptr;                                                                               \
        }                                                                                                                 \
        AscendC::SetMatrixKfc(tpipe, &__kfcClient__, 0, workspace, __VA_ARGS__);                                          \
        if constexpr (!asEnableMixDualMaster) {                                                                           \
            AscendC::CrossCoreWaitFlag<matmul::INTRA_MODE, PIPE_S>(AscendC::KFC_SYNC_ID);                                 \
        }                                                                                                                 \
    }

#else
#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                         \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>;     \
    static_assert(ASCubeObjConfig::enableABShareValue != AscendC::CUBEOBJ_MIX_MODE,                    \
        "If both aType ibshare and bType ibshare are set to true, the values must "                    \
        "be the same for all cube objects.");                                                          \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;                \
    static_assert(asEnableMixDualMaster != AscendC::CUBEOBJ_MIX_MODE,                                  \
        "enableMixDualMaster must be consistent for all cube objects.");                               \
    if constexpr (!asEnableMixDualMaster) {                                                            \
        AscendC::ClearWorkspace(reinterpret_cast<__gm__ uint8_t *>(workspace));                        \
    }                                                                                                  \
    if ASCEND_IS_AIC {                                                                                 \
        AscendC::KfcServer server;                                                                     \
        server.Init(workspace);                                                                        \
        server.InitObj(tpipe, __VA_ARGS__);                                                            \
        if constexpr (!asEnableMixDualMaster) {                                                        \
            while (server.isRun()) {                                                                   \
                server.Run(__VA_ARGS__);                                                               \
            };                                                                                         \
            server.Quit();                                                                             \
            return;                                                                                    \
        }                                                                                              \
    }                                                                                                  \
    AscendC::KfcCommClient __kfcClient__(workspace, AscendC::GetSubBlockIdx(), asEnableMixDualMaster); \
    if ASCEND_IS_AIV {                                                                                 \
        if constexpr (!asEnableMixDualMaster) {                                                        \
            AscendC::g_kfcClient = &__kfcClient__;                                                     \
        } else {                                                                                       \
            AscendC::g_kfcClient = nullptr;                                                            \
        }                                                                                              \
        AscendC::SetMatrixKfc(tpipe, &__kfcClient__, 0, workspace, __VA_ARGS__);                       \
    }
#endif
#endif

#else

#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    AscendC::InitCurObj(tpipe, __VA_ARGS__)
#endif

#else

#ifdef SPLIT_CORE_CUBE
#ifdef ASCENDC_CUBE_ONLY
#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    AscendC::InitCurObj(tpipe, __VA_ARGS__);   \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER_OBJ))

#define REGIST_CUBE_OBJ_REMOTE(tpipe, workspace, ...)
#else
#if defined(__DAV_C310__) || defined(__DAV_310R6__)
#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                             \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>;         \
    constexpr int8_t enableHardPollKfc = AscendC::ENABLE_HARD_POOL ? AscendC::ENABLE_HARD_POOL             \
                                        : ASCubeObjConfig::enableABShareValue;                             \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;                    \
    if constexpr (!asEnableMixDualMaster) {                                                                \
        AscendC::ClearWorkspace(reinterpret_cast<__gm__ uint8_t *>(workspace));                            \
    }                                                                                                      \
    AscendC::TQue<AscendC::QuePosition::CO1, 1, &AscendC::gCO1Config> qCO1;                                \
    AscendC::gCO1Que = &qCO1;                                                                              \
    AscendC::KfcServer<enableHardPollKfc> server;                                                          \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER));        \
    server.Init(workspace);                                                                                \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER_INIT));   \
    server.InitObj(tpipe, __VA_ARGS__);                                                                    \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER_OBJ));    \
    if constexpr (!asEnableMixDualMaster) {                                                                \
        while (server.isRun()) {                                                                           \
            server.Run(__VA_ARGS__);                                                                       \
        };                                                                                                 \
        server.Quit();                                                                                     \
        return;                                                                                            \
    }
#else
#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                             \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>;         \
    static_assert(ASCubeObjConfig::enableABShareValue != AscendC::CUBEOBJ_MIX_MODE,                        \
        "If both aType ibshare and bType ibshare are set to true, the values must "                        \
        "be the same for all cube objects.");                                                              \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;                    \
    static_assert(asEnableMixDualMaster != AscendC::CUBEOBJ_MIX_MODE,                                      \
        "enableMixDualMaster must be consistent for all cube objects.");                                   \
    if constexpr (!asEnableMixDualMaster) {                                                                \
        AscendC::ClearWorkspace(reinterpret_cast<__gm__ uint8_t *>(workspace));                            \
    }                                                                                                      \
    AscendC::KfcServer server;                                                                             \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER));        \
    server.Init(workspace);                                                                                \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER_INIT));   \
    server.InitObj(tpipe, __VA_ARGS__);                                                                    \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_SERVER_OBJ));    \
    if constexpr (!asEnableMixDualMaster) {                                                                \
        while (server.isRun()) {                                                                           \
            server.Run(__VA_ARGS__);                                                                       \
        };                                                                                                 \
        server.Quit();                                                                                     \
        return;                                                                                            \
    }
#endif
#endif

#elif defined(SPLIT_CORE_VEC)
#ifdef ASCENDC_CUBE_ONLY
#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    return
#elif defined(__DAV_C310__) || defined(__DAV_310R6__)
#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                     \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>; \
    constexpr int8_t enableHardPollKfc = AscendC::ENABLE_HARD_POOL ? AscendC::ENABLE_HARD_POOL     \
                                        : ASCubeObjConfig::enableABShareValue;                     \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;            \
    AscendC::KfcCommClient __kfcClient__(workspace, AscendC::GetSubBlockIdx(), enableHardPollKfc,  \
        asEnableMixDualMaster);                                                                    \
    if constexpr (!asEnableMixDualMaster) {                                                        \
        AscendC::g_kfcClient = &__kfcClient__;                                                     \
    } else {                                                                                       \
        AscendC::g_kfcClient = nullptr;                                                            \
    }                                                                                              \
    AscendC::SetMatrixKfc(tpipe, &__kfcClient__, 0, workspace, __VA_ARGS__);                       \
    if constexpr (!asEnableMixDualMaster) {                                                        \
        AscendC::CrossCoreWaitFlag<matmul::INTRA_MODE, PIPE_S>(AscendC::KFC_SYNC_ID);              \
    }
#else
#define REGIST_CUBE_OBJ(tpipe, workspace, ...)                                                            \
    using ASCubeObjConfig = AscendC::GetCubeObjConfig<decltype(AscendC::GetObjType(__VA_ARGS__))>;        \
    static_assert(ASCubeObjConfig::enableABShareValue != AscendC::CUBEOBJ_MIX_MODE,                       \
        "If both aType ibshare and bType ibshare are set to true, the values must "                       \
        "be the same for all cube objects.");                                                             \
    constexpr int8_t asEnableMixDualMaster = ASCubeObjConfig::enableMixDualMasterValue;                   \
    static_assert(asEnableMixDualMaster != AscendC::CUBEOBJ_MIX_MODE,                                     \
        "enableMixDualMaster must be consistent for all cube objects.");                                  \
    AscendC::KfcCommClient __kfcClient__(workspace, AscendC::GetSubBlockIdx(), asEnableMixDualMaster);    \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_CLIENT_KFC));   \
    if constexpr (!asEnableMixDualMaster) {                                                               \
        AscendC::g_kfcClient = &__kfcClient__;                                                            \
    }                                                                                                     \
    AscendC::SetMatrixKfc(tpipe, &__kfcClient__, 0, workspace, __VA_ARGS__);                              \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_MATRIX_KFC));   \
    if constexpr (!asEnableMixDualMaster) {                                                               \
        AscendC::WaitEvent(AscendC::WORKSPACE_SYNC_ID);                                                   \
    }                                                                                                     \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_WAIT_EVE))
#endif
#else

#define REGIST_CUBE_OBJ(tpipe, workspace, ...) \
    AscendC::InitCurObj(tpipe, __VA_ARGS__);   \
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_MATMUL_OBJ))
#endif
#endif

#endif