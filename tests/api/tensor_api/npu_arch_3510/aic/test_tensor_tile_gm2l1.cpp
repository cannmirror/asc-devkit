/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <cstring>
#include <numeric>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <cxxabi.h>
#include "mockcpp/mockcpp.hpp"
#include "tensor_api/stub/cce_stub.h"
#include "include/experimental/tensor_api/tensor.h"

using namespace AscendC::Te;
using namespace AscendC;

// Mock implementations for data copy about gm2l1 functions
extern void set_loop1_stride_outtol1(uint64_t config);
extern void set_loop2_stride_outtol1(uint64_t config);
extern void set_loop_size_outtol1(uint64_t config);
extern void set_pad_val_outtol1(uint64_t config);
extern void copy_gm_to_cbuf_align_v2(half* dst, half* src, uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                                     uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit,
                                     uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride);
extern void copy_gm_to_cbuf_multi_nd2nz(half* dst, half* src, uint8_t sid, uint64_t loop1_src_stride,
                                        uint8_t l2_cache_ctl, uint16_t n_value, uint32_t d_value,
                                        uint64_t loop4_src_stride, bool smallc0_en);
extern void copy_gm_to_cbuf_multi_dn2nz(half* dst, half* src, uint8_t sid, uint64_t loop1_src_stride,
                                        uint8_t l2_cache_ctl, uint16_t n_value, uint32_t d_value,
                                        uint64_t loop4_src_stride, bool smallc0_en);
extern void set_mte2_nz_para(uint64_t para);

struct CopyGm2L1AlignV2Capture {
    void* dst = nullptr;
    void* src = nullptr;
    uint32_t blockCount = 0;
    uint32_t blockLen = 0;
    uint8_t leftPaddingCnt = 0;
    uint8_t rightPaddingCnt = 0;
    bool dataSelectBit = false;
    uint8_t l2CacheCtl = 0;
    uint64_t srcStride = 0;
    uint32_t dstStride = 0;
};

struct CopyGm2L1ND2NzCapture {
    void* dst = nullptr;
    void* src = nullptr;
    uint64_t loop1SrcStride = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t loop4SrcStride = 0;
    bool enableSmallC0 = false;
};

struct CopyGm2L1DN2NzCapture {
    void* dst = nullptr;
    void* src = nullptr;
    uint64_t loop1SrcStride = 0;
    uint16_t nValue = 0;
    uint32_t dValue = 0;
    uint64_t loop4SrcStride = 0;
    bool enableSmallC0 = false;
};

struct CopyGm2L1NzParaCapture {
    union {
        struct {
            uint16_t ndNum;          // MTE2_NZ_PARA[15:0]
            uint16_t loop2DstStride; // MTE2_NZ_PARA[31:16]
            uint16_t loop3DstStride; // MTE2_NZ_PARA[47:32]
            uint16_t loop4DstStride; // MTE2_NZ_PARA[63:48]
        };
        uint64_t mte2NzPara;
    };
};

// Global capture object
std::vector<CopyGm2L1AlignV2Capture> gGm2L1AlignV2Captures;
std::vector<CopyGm2L1ND2NzCapture> gGm2L1ND2NzCaptures;
std::vector<CopyGm2L1DN2NzCapture> gGm2L1DN2NzCaptures;
std::vector<CopyGm2L1NzParaCapture> gGm2L1NzParaCaptures;

// Reset capture data
void ResetCapture()
{
    gGm2L1AlignV2Captures.clear();
    gGm2L1ND2NzCaptures.clear();
    gGm2L1DN2NzCaptures.clear();
    gGm2L1NzParaCaptures.clear();
}

void PrintCaptureData()
{
    for (const auto& capture : gGm2L1AlignV2Captures) {
        std::cout << "CopyGmToCbufAlignV2 Capture - dst: " << capture.dst << ", src: " << capture.src
                  << ", blockCount: " << capture.blockCount << ", blockLen: " << capture.blockLen
                  << ", leftPaddingCnt: " << static_cast<int>(capture.leftPaddingCnt)
                  << ", rightPaddingCnt: " << static_cast<int>(capture.rightPaddingCnt)
                  << ", l2CacheCtl: " << static_cast<int>(capture.l2CacheCtl) << ", srcStride: " << capture.srcStride
                  << ", dstStride: " << capture.dstStride << std::endl;
    }

    for (const auto& capture : gGm2L1ND2NzCaptures) {
        std::cout << "CopyGmToCbufMultiND2nz Capture - dst: " << capture.dst << ", src: " << capture.src
                  << ", loop1SrcStride: " << capture.loop1SrcStride << ", nValue: " << capture.nValue
                  << ", dValue: " << capture.dValue << ", loop4SrcStride: " << capture.loop4SrcStride
                  << ", enableSmallC0: " << std::boolalpha << capture.enableSmallC0 << std::endl;
    }

    for (const auto& capture : gGm2L1DN2NzCaptures) {
        std::cout << "CopyGmToCbufMultiDN2nz Capture - dst: " << capture.dst << ", src: " << capture.src
                  << ", loop1SrcStride: " << capture.loop1SrcStride << ", nValue: " << capture.nValue
                  << ", dValue: " << capture.dValue << ", loop4SrcStride: " << capture.loop4SrcStride
                  << ", enableSmallC0: " << std::boolalpha << capture.enableSmallC0 << std::endl;
    }

    for (const auto& capture : gGm2L1NzParaCaptures) {
        std::cout << "SetMTE2NzPara Capture - mte2NzPara: " << capture.mte2NzPara << ", ndNum: " << capture.ndNum
                  << ", loop2DstStride: " << capture.loop2DstStride << ", loop3DstStride: " << capture.loop3DstStride
                  << ", loop4DstStride: " << capture.loop4DstStride << std::endl;
    }
}

template <typename T>
void PrintTensor(const T& src)
{
    using srcType = typename T::elementType;
    auto srcLayout = src.Layout();
    uint32_t M0 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(srcLayout);
    uint32_t N0 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
    uint32_t M1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    uint32_t N1 = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    if constexpr (IsScaleANDFormat<T>::value) {
        std::cout << "ScaleAND";
    } else if constexpr (IsNDFormat<T>::value) {
        std::cout << "ND";
    } else if constexpr (IsDNFormat<T>::value) {
        std::cout << "DN";
    } else if constexpr (IsNZFormat<T>::value) {
        std::cout << "NZ";
    } else if constexpr (IsZNFormat<T>::value) {
        std::cout << "ZN";
    } else if constexpr (IsZZFormat<T>::value) {
        std::cout << "ZZ";
    } else {
        std::cout << "UnknownLayout";
    }
    if (M0 == 1 && N0 == 1) { // for 2D layout, print in 2D format
        std::cout << " Layout Result (2D) (" << M1 << ", " << N1 << "): " << std::endl;
        for (int i = 0; i < M1; i++) {
            std::cout << i << ":\t";
            for (int j = 0; j < N1; j++) {
                auto dataAddr = &(src[MakeCoord(i, j)]);
                if constexpr (sizeof(srcType) == 1) {
                    std::cout << static_cast<uint32_t>(*(reinterpret_cast<uint8_t*>(dataAddr))) << "\t";
                } else {
                    std::cout << *dataAddr << "\t";
                }
            }
            std::cout << std::endl;
        }
    } else { // for NZ, ZN, ZZ, print in 4D format
        std::cout << " Layout Result (4D) (" << M1 << ", " << N1 << ", " << M0 << ", " << N0 << "): " << std::endl;
        for (int i0 = 0; i0 < M1; i0++) {
            for (int i1 = 0; i1 < M0; i1++) {
                for (int j0 = 0; j0 < N1; j0++) {
                    uint32_t block_id = j0 * M1 + i0;
                    for (int j1 = 0; j1 < N0; j1++) {
                        auto dataAddr = &(src[MakeCoord(MakeCoord(i1, i0), MakeCoord(j1, j0))]);
                        if constexpr (sizeof(srcType) == 1) {
                            std::cout << static_cast<uint32_t>(*(reinterpret_cast<uint8_t*>(dataAddr))) << "\t";
                        } else {
                            std::cout << *dataAddr << "\t";
                        }
                    }
                    std::cout << "|";
                }
                std::cout << std::endl;
            }
            std::cout << "-----------------------------------------" << std::endl;
        }
    }
}

inline void __print_type_hierarchy(const std::string& type_str)
{
    int indent_level = 1;
    const int indent_spaces = 4; // 每层缩进的空格数
    for (int s = 0; s < indent_level * indent_spaces; ++s)
        std::cout << " ";
    for (size_t i = 0; i < type_str.size(); ++i) {
        char c = type_str[i];
        if (c == '<') {
            // 遇到 <，换行并增加缩进
            std::cout << c << "\n";
            indent_level++;
            // 打印缩进
            for (int s = 0; s < indent_level * indent_spaces; ++s)
                std::cout << " ";
        } else if (c == ',' && indent_level > 0) {
            // 遇到逗号，换行并保持当前缩进
            std::cout << c << "\n";
            for (int s = 0; s < indent_level * indent_spaces - 1; ++s)
                std::cout << " ";
        } else if (c == '>') {
            // 遇到 >，先换行，减少缩进，再打印 >
            std::cout << "\n";
            indent_level--;
            for (int s = 0; s < indent_level * indent_spaces; ++s)
                std::cout << " ";
            std::cout << c;
        } else {
            // 普通字符直接打印
            std::cout << c;
        }
    }
    std::cout << std::endl;
}

template <typename T, typename... Args>
inline void PrintTypeHierarchy(const Args&... args)
{
    if constexpr (!std::is_same_v<T, void>) {
        std::cout << "Type Hierarchy for: ";
    }
    ((std::cout << args << " "), ...);
    std::cout << std::endl;
    if constexpr (std::is_same_v<T, void>) {
        return;
    }
    std::string raw_name = typeid(T).name();
    int status = -4;
    char* res = abi::__cxa_demangle(raw_name.c_str(), NULL, NULL, &status);
    std::string ret = (status == 0) ? res : raw_name;
    if (status == 0)
        std::free(res);
    __print_type_hierarchy(ret);
}

template <typename T, typename U>
void SimND2ND(const T& dst, const U& src)
{
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    uint32_t M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    uint32_t N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcRowStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);
    auto srcColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);
    auto dstRowStride = GetEleFromLayout<decltype(dstLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(dstLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    if (M == M1 && N == N1 && srcRowStride == N && dstRowStride == N) {
        N1 = M * N;
        N = M * N;
        M = 1;
        M1 = 1;
    }
    N1 = (N1 + c0Elements - 1) / c0Elements * c0Elements; // align N1 to C0 boundary

    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * M0 + m0) * N0 + n0;
                    uint32_t srcColNAlignC0 = ((N + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[srcRow * srcRowStride + srcCol * srcColStride];
                    } else if (srcRow < M && srcCol >= N && srcCol < srcColNAlignC0) {
                        // padding with 0 if out of bound
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U, typename Coord>
void SimND2ND(const T& dst, const U& src, const Coord& coord)
{
    static_assert(IsNDFormat<U>::value && IsNDFormat<T>::value);
    // get slice src, and then call SimND2ND with the slice
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto sliceTensor = src(coord, dst);
    SimND2ND(dst, sliceTensor);
}

template <typename T, typename U>
void SimND2Nz(const T& dst, const U& src)
{
    static_assert(IsNDFormat<U>::value && IsNZFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcRowStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t m1 = 0; m1 < M1; m1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((n1 * M1 + m1) * M0 + m0) * N0 + n0;
                    uint32_t srcColNAlignC0 = ((N + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[srcRow * srcRowStride + srcCol];
                    } else if (srcRow < M && srcCol >= N && srcCol < srcColNAlignC0) {
                        // right padding and bottom not padding, right pad to the next C0 boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U, typename Coord>
void SimND2Nz(const T& dst, const U& src, const Coord& coord)
{
    static_assert(IsNDFormat<U>::value && IsNZFormat<T>::value);
    // get slice src, and then call SimND2Nz with the slice
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto sliceTensor = src(coord, dst);
    SimND2Nz(dst, sliceTensor);
}

template <typename T, typename U>
void SimDN2Zn(const T& dst, const U& src)
{
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);

    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t n0 = 0; n0 < N0; n0++) {
                for (uint32_t m0 = 0; m0 < M0; m0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * N0 + n0) * M0 + m0;
                    uint32_t srcRowNAlignC0 = ((M + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcCol < N && srcRow < M) {
                        dst.Data()[dstIndex] = src.Data()[srcCol * srcColStride + srcRow];
                    } else if (srcCol < N && srcRow >= M && srcRow < srcRowNAlignC0) {
                        // bottom padding and right not padding, bottom pad to the next C0 boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimDN2Nz(const T& dst, const U& src)
{
    using srcType = typename U::elementType;
    static_assert(!is_b4_type<srcType>, "DN2NZ does not support b4 type");
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
    for (uint32_t n1 = 0; n1 < N1; n1++) {
        for (uint32_t m1 = 0; m1 < M1; m1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((n1 * M1 + m1) * M0 + m0) * N0 + n0;
                    uint32_t srcColNAlignC0 = ((N + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[srcRow + srcCol * srcColStride];
                    } else if (srcRow < M && srcCol >= N && srcCol < srcColNAlignC0) {
                        // right padding and bottom not padding, right pad to the next C0 boundary
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimScaleAND2Zz(const T& dst, const U& src)
{
    // static_assert(IsScaleANDFormat<U>::value && IsZZFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto N = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);

    auto srcRowStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::ROW, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * M0 + m0) * N0 + n0;
                    uint32_t srcRowNAlignC0 = ((M + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[srcRow * srcRowStride + srcCol];
                    } else if (srcCol < N && srcRow >= M && srcRow < srcRowNAlignC0) {
                        // bottom padding and right not padding, bottom pad to the next C0 boundary
                        // use dn2nz way to pad, which means padding in the raw row direction, actual col direction
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

template <typename T, typename U>
void SimScaleADN2Zz(const T& dst, const U& src)
{
    static_assert(IsScaleADNFormat<U>::value && IsZZFormat<T>::value);
    using srcType = typename U::elementType;
    static_assert(std::is_same_v<srcType, typename T::elementType>, "src and dst element types must be the same");
    auto dstLayout = dst.Layout();
    auto srcLayout = src.Layout();
    auto M = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(srcLayout);
    auto SN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(srcLayout);
    auto BN = GetEleFromLayout<decltype(srcLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(srcLayout);
    auto N = SN * BN;

    auto srcBColStride = GetEleFromLayout<decltype(srcLayout), AttrInfo::STRIDE, AttrInfo::COLUMN, 1>(srcLayout);

    uint32_t c0Elements = C0_SIZE<srcType> / sizeof(srcType);
    uint32_t M0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 0>(dstLayout);
    uint32_t N0 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 0>(dstLayout);
    uint32_t M1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::ROW, 1>(dstLayout);
    uint32_t N1 = GetEleFromLayout<decltype(dstLayout), AttrInfo::SHAPE, AttrInfo::COLUMN, 1>(dstLayout);
    for (uint32_t m1 = 0; m1 < M1; m1++) {
        for (uint32_t n1 = 0; n1 < N1; n1++) {
            for (uint32_t m0 = 0; m0 < M0; m0++) {
                for (uint32_t n0 = 0; n0 < N0; n0++) {
                    uint32_t srcRow = m1 * M0 + m0;
                    uint32_t srcCol = n1 * N0 + n0;
                    uint32_t dstIndex = ((m1 * N1 + n1) * M0 + m0) * N0 + n0;
                    uint32_t srcRowNAlignC0 = ((M + c0Elements - 1) / c0Elements) * c0Elements;
                    if (srcRow < M && srcCol < N) {
                        dst.Data()[dstIndex] = src.Data()[n1 * srcBColStride + srcRow * 2 + n0];
                    } else if (srcCol < N && srcRow >= M && srcRow < srcRowNAlignC0) {
                        // bottom padding and right not padding, bottom pad to the next C0 boundary
                        // use dn2nz way to pad, which means padding in the raw row direction, actual col direction
                        dst.Data()[dstIndex] = static_cast<srcType>(0);
                    }
                }
            }
        }
    }
}

extern void copy_gm_to_cbuf_multi_nd2nz(half* dst, half* src, uint8_t sid, uint64_t loop1_src_stride,
                                        uint8_t l2_cache_ctl, uint16_t n_value, uint32_t d_value,
                                        uint64_t loop4_src_stride, bool smallc0_en);
template <typename T>
void SimulateND2nzDataCopy(T* dst, T* src, uint64_t loop1SrcStride, uint16_t nValue, uint32_t dValue,
                           uint64_t loop4SrcStride, bool enableSmallC0)
{
    if (gGm2L1NzParaCaptures.empty()) {
        return;
    }
    uint16_t ndNum = gGm2L1NzParaCaptures.back().ndNum;
    uint16_t loop2DstStride = gGm2L1NzParaCaptures.back().loop2DstStride;
    uint16_t loop3DstStride = gGm2L1NzParaCaptures.back().loop3DstStride;
    uint16_t loop4DstStride = gGm2L1NzParaCaptures.back().loop4DstStride;
    constexpr uint32_t typeSize = sizeof(T);
    uint32_t c0Elements = C0_SIZE<T> / typeSize; // Number of elements in one C0 block
    if (enableSmallC0) {
        for (int h = 0; h < ndNum; h++) {
            const uint8_t* srcNDAddr = reinterpret_cast<const uint8_t*>(src) + h * loop4SrcStride;
            uint8_t* dstNDAddr = reinterpret_cast<uint8_t*>(dst) + h * loop4DstStride * C0_SIZE<T>;

            uint16_t nCeil = (nValue + 3) / 4;
            for (int j = 0; j < nCeil; j++) {
                const uint8_t* srcNAddr = (j < nValue) ? (srcNDAddr + j * loop1SrcStride) : nullptr;
                uint8_t* dstNAddr = dstNDAddr + j * 4 * typeSize;
                for (int k = 0; k < 4; k++) {
                    uint8_t* dstEleAddr = dstNAddr + k * typeSize;
                    if ((k < dValue) && (srcNAddr != nullptr)) {
                        const uint8_t* srcEleAddr = srcNAddr + k * typeSize;
                        std::copy(srcEleAddr, srcEleAddr + typeSize, dstEleAddr);
                    } else {
                        std::fill(dstEleAddr, dstEleAddr + typeSize, 0); // Padding with zeros
                    }
                }
            }
        }
    } else {
        uint32_t blockNum = (dValue + c0Elements - 1) / c0Elements;
        for (int h = 0; h < ndNum; h++) {
            const uint8_t* srcNDAddr = reinterpret_cast<const uint8_t*>(src) + h * loop4SrcStride;
            uint8_t* dstNDAddr = reinterpret_cast<uint8_t*>(dst) + h * loop4DstStride * C0_SIZE<T>;
            for (int i = 0; i < blockNum; i++) {
                const uint8_t* srcBlockAddr = srcNDAddr + i * C0_SIZE<T>;
                uint8_t* dstBlockAddr = dstNDAddr + i * loop3DstStride * C0_SIZE<T>;

                for (int j = 0; j < nValue; j++) {
                    const uint8_t* srcNAddr = srcBlockAddr + j * loop1SrcStride;
                    uint8_t* dstNAddr = dstBlockAddr + j * loop2DstStride * C0_SIZE<T>;
                    for (int k = 0; k < c0Elements; k++) {
                        uint32_t srcEleIndex = i * c0Elements + k;
                        uint8_t* dstEleAddr = dstNAddr + k * typeSize;
                        if (srcEleIndex < dValue) {
                            const uint8_t* srcEleAddr = srcNAddr + k * typeSize;
                            std::copy(srcEleAddr, srcEleAddr + typeSize, dstEleAddr);
                        } else {
                            std::fill(dstEleAddr, dstEleAddr + typeSize, 0); // Padding with zeros
                        }
                    }
                }
            }
        }
    }
}

extern void copy_gm_to_cbuf_multi_dn2nz(half* dst, half* src, uint8_t sid, uint64_t loop1_src_stride,
                                        uint8_t l2_cache_ctl, uint16_t n_value, uint32_t d_value,
                                        uint64_t loop4_src_stride, bool smallc0_en);
template <typename T>
void SimulateDN2nzDataCopy(T* dst, T* src, uint64_t loop1SrcStride, uint16_t nValue, uint32_t dValue,
                           uint64_t loop4SrcStride, bool enableSmallC0)
{
    if (gGm2L1NzParaCaptures.empty()) {
        return;
    }
    uint16_t dnNum = gGm2L1NzParaCaptures.back().ndNum;
    uint16_t loop2DstStride = gGm2L1NzParaCaptures.back().loop2DstStride;
    uint16_t loop3DstStride = gGm2L1NzParaCaptures.back().loop3DstStride;
    uint16_t loop4DstStride = gGm2L1NzParaCaptures.back().loop4DstStride;
    constexpr uint32_t typeSize = sizeof(T);
    uint32_t c0Elements = C0_SIZE<T> / typeSize; // Number of elements in one C0 block
    if (enableSmallC0) {
        for (int h = 0; h < dnNum; h++) {
            const uint8_t* srcDNAddr = reinterpret_cast<const uint8_t*>(src) + h * loop4SrcStride;
            uint8_t* dstDNAddr = reinterpret_cast<uint8_t*>(dst) + h * loop4DstStride * C0_SIZE<T>;

            uint16_t nCeil = (nValue + 3) / 4;
            for (int j = 0; j < nCeil; j++) {
                const uint8_t* srcNAddr = (j < nValue) ? (srcDNAddr + j * typeSize) : nullptr;
                uint8_t* dstNAddr = dstDNAddr + j * 4 * typeSize;
                for (int k = 0; k < 4; k++) {
                    uint8_t* dstEleAddr = dstNAddr + k * typeSize;
                    if ((k < dValue) && (srcNAddr != nullptr)) {
                        const uint8_t* srcEleAddr = srcNAddr + k * loop1SrcStride;
                        std::copy(srcEleAddr, srcEleAddr + typeSize, dstEleAddr);
                    } else {
                        std::fill(dstEleAddr, dstEleAddr + typeSize, 0); // Padding with zeros
                    }
                }
            }
        }
    } else {
        uint32_t blockNum = (dValue + c0Elements - 1) / c0Elements;
        for (int h = 0; h < dnNum; h++) {
            const uint8_t* srcDNAddr = reinterpret_cast<const uint8_t*>(src) + h * loop4SrcStride;
            uint8_t* dstDNAddr = reinterpret_cast<uint8_t*>(dst) + h * loop4DstStride * C0_SIZE<T>;
            for (int i = 0; i < blockNum; i++) {
                const uint8_t* srcBlockAddr = srcDNAddr + i * loop1SrcStride * c0Elements;
                uint8_t* dstBlockAddr = dstDNAddr + i * loop3DstStride * C0_SIZE<T>;

                for (int j = 0; j < nValue; j++) {
                    const uint8_t* srcNAddr = srcBlockAddr + j * typeSize;
                    uint8_t* dstNAddr = dstBlockAddr + j * loop2DstStride * C0_SIZE<T>;
                    for (int k = 0; k < c0Elements; k++) {
                        uint32_t srcEleIndex = i * c0Elements + k;
                        uint8_t* dstEleAddr = dstNAddr + k * typeSize;
                        if (srcEleIndex < dValue) {
                            const uint8_t* srcEleAddr = srcNAddr + k * loop1SrcStride;
                            std::copy(srcEleAddr, srcEleAddr + typeSize, dstEleAddr);
                        } else {
                            std::fill(dstEleAddr, dstEleAddr + typeSize, 0); // Padding with zeros
                        }
                    }
                }
            }
        }
    }
}

extern void copy_gm_to_cbuf_align_v2(half* dst, half* src, uint8_t sid, uint32_t burst_num, uint32_t burst_len,
                                     uint8_t left_padding_count, uint8_t right_padding_count, bool data_select_bit,
                                     uint8_t l2_cache_ctl, uint64_t burst_src_stride, uint32_t burst_dst_stride);
template <typename T>
void SimulateAlignV2DataCopy(T* dst, T* src, uint32_t blockCount, uint32_t blockLen, uint8_t leftPaddingCnt,
                             uint8_t rightPaddingCnt, bool dataSelectBit, uint64_t srcStride, uint32_t dstStride)
{
    bool isLPRPMode = (leftPaddingCnt > 0) || (rightPaddingCnt > 0);
    bool isCompactMode = (dstStride == blockLen);
    uint32_t totalBurstSize = blockLen + leftPaddingCnt * sizeof(T) + rightPaddingCnt * sizeof(T);
    uint32_t padSize = (totalBurstSize % C0_SIZE<T> == 0) ? 0 : (C0_SIZE<T> - (totalBurstSize % C0_SIZE<T>));
    uint32_t padElem = padSize / sizeof(T);
    // compact mode, left and right pad cnt is zero, dstStride equals blockLen, can directly copy without padding
    if (isLPRPMode) {
        // In LPRP mode, dstStride should be aligned to C0 size
        EXPECT_TRUE(dstStride % C0_SIZE<T> == 0);
        for (uint32_t blockId = 0; blockId < blockCount; blockId++) {
            uint8_t* srcBurst = reinterpret_cast<uint8_t*>(src) + blockId * srcStride;
            uint8_t* dstBurst = reinterpret_cast<uint8_t*>(dst) + blockId * dstStride;

            if (leftPaddingCnt > 0) {
                std::fill(dstBurst, dstBurst + leftPaddingCnt * sizeof(T), 0); // Padding with zeros
            }
            std::copy(srcBurst, srcBurst + blockLen, dstBurst + leftPaddingCnt * sizeof(T));

            uint32_t rightPadOffset = leftPaddingCnt * sizeof(T) + blockLen;
            if (rightPaddingCnt > 0) {
                std::fill(dstBurst + rightPadOffset, dstBurst + rightPadOffset + rightPaddingCnt * sizeof(T),
                          0); // Padding with zeros
            }

            uint32_t padOffset = leftPaddingCnt * sizeof(T) + blockLen + rightPaddingCnt * sizeof(T);
            if (padElem > 0) {
                std::fill(dstBurst + padOffset, dstBurst + padOffset + padElem * sizeof(T), 0); // Padding with zeros
            }
        }
        return;
    }
    if (isCompactMode) {
        uint8_t* srcBase = reinterpret_cast<uint8_t*>(src);
        uint8_t* dstBase = reinterpret_cast<uint8_t*>(dst);
        for (uint32_t blockId = 0; blockId < blockCount; blockId++) {
            const uint8_t* srcBurst = srcBase + blockId * srcStride;
            uint8_t* dstBurst = dstBase + blockId * dstStride;
            std::copy(srcBurst, srcBurst + blockLen, dstBurst);
        }
        // check tail padding
        uint32_t totalDataLen = blockCount * blockLen;
        uint64_t aligndSize = ((totalDataLen + C0_SIZE<T> - 1) / C0_SIZE<T>)*C0_SIZE<T>;
        if (aligndSize > totalDataLen) {
            uint8_t* padStart = dstBase + totalDataLen;
            std::fill(padStart, padStart + (aligndSize - totalDataLen), 0); // Padding with zeros
        }
    } else {
        // normal mode
        for (uint32_t blockId = 0; blockId < blockCount; blockId++) {
            uint8_t* srcBurst = reinterpret_cast<uint8_t*>(src) + blockId * srcStride;
            uint8_t* dstBurst = reinterpret_cast<uint8_t*>(dst) + blockId * dstStride;
            std::copy(srcBurst, srcBurst + blockLen, dstBurst);
            if (padElem > 0) {
                uint8_t* padStart = dstBurst + blockLen;
                std::fill(padStart, padStart + padElem * sizeof(T), 0); // Padding with zeros
            }
        }
    }
}

#define CAPTURE_GM_TO_L1(type)                                                                                         \
    void CaptureCopyGmToCbufAlignV2_##type(__cbuf__ type* dst, __gm__ type* src, uint8_t sid, uint32_t blockCount,     \
                                           uint32_t blockLen, uint8_t leftPaddingCnt, uint8_t rightPaddingCnt,         \
                                           bool dataSelectBit, uint8_t l2CacheCtl, uint64_t srcStride,                 \
                                           uint32_t dstStride)                                                         \
    {                                                                                                                  \
        CopyGm2L1AlignV2Capture capture;                                                                               \
        capture.dst = reinterpret_cast<void*>(dst);                                                                    \
        capture.src = reinterpret_cast<void*>(src);                                                                    \
        capture.blockCount = blockCount;                                                                               \
        capture.blockLen = blockLen;                                                                                   \
        capture.leftPaddingCnt = leftPaddingCnt;                                                                       \
        capture.rightPaddingCnt = rightPaddingCnt;                                                                     \
        capture.dataSelectBit = dataSelectBit;                                                                         \
        capture.l2CacheCtl = l2CacheCtl;                                                                               \
        capture.srcStride = srcStride;                                                                                 \
        capture.dstStride = dstStride;                                                                                 \
        gGm2L1AlignV2Captures.push_back(capture);                                                                      \
        SimulateAlignV2DataCopy(dst, src, blockCount, blockLen, leftPaddingCnt, rightPaddingCnt, dataSelectBit,        \
                                srcStride, dstStride);                                                                 \
    }                                                                                                                  \
    void CaptureCopyGmToCbufMultiND2nz_##type(__cbuf__ type* dst, __gm__ type* src, uint8_t sid,                       \
                                              uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,       \
                                              uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)            \
    {                                                                                                                  \
        CopyGm2L1ND2NzCapture capture;                                                                                 \
        capture.dst = reinterpret_cast<void*>(dst);                                                                    \
        capture.src = reinterpret_cast<void*>(src);                                                                    \
        capture.loop1SrcStride = loop1_src_stride;                                                                     \
        capture.nValue = n_value;                                                                                      \
        capture.dValue = d_value;                                                                                      \
        capture.loop4SrcStride = loop4_src_stride;                                                                     \
        capture.enableSmallC0 = smallc0_en;                                                                            \
        gGm2L1ND2NzCaptures.push_back(capture);                                                                        \
        SimulateND2nzDataCopy(dst, src, loop1_src_stride, n_value, d_value, loop4_src_stride, smallc0_en);             \
    }                                                                                                                  \
    void CaptureCopyGmToCbufMultiDN2nz_##type(__cbuf__ type* dst, __gm__ type* src, uint8_t sid,                       \
                                              uint64_t loop1_src_stride, uint8_t l2_cache_ctl, uint16_t n_value,       \
                                              uint32_t d_value, uint64_t loop4_src_stride, bool smallc0_en)            \
    {                                                                                                                  \
        CopyGm2L1DN2NzCapture capture;                                                                                 \
        capture.dst = reinterpret_cast<void*>(dst);                                                                    \
        capture.src = reinterpret_cast<void*>(src);                                                                    \
        capture.loop1SrcStride = loop1_src_stride;                                                                     \
        capture.nValue = n_value;                                                                                      \
        capture.dValue = d_value;                                                                                      \
        capture.loop4SrcStride = loop4_src_stride;                                                                     \
        capture.enableSmallC0 = smallc0_en;                                                                            \
        gGm2L1DN2NzCaptures.push_back(capture);                                                                        \
        SimulateDN2nzDataCopy(dst, src, loop1_src_stride, n_value, d_value, loop4_src_stride, smallc0_en);             \
    }

CAPTURE_GM_TO_L1(uint8_t);
CAPTURE_GM_TO_L1(half);
CAPTURE_GM_TO_L1(uint16_t);
CAPTURE_GM_TO_L1(float);
CAPTURE_GM_TO_L1(uint32_t);

void CaptureSetMTE2NzPara(uint64_t para)
{
    CopyGm2L1NzParaCapture capture;
    capture.mte2NzPara = para;
    gGm2L1NzParaCaptures.push_back(capture);
}

#define MOCKER_GM_TO_L1(type)                                                                                          \
    MOCKER(copy_gm_to_cbuf_align_v2, void (*)(__cbuf__ type*, __gm__ type*, uint8_t, uint32_t, uint32_t, uint8_t,      \
                                              uint8_t, bool, uint8_t, uint64_t, uint32_t))                             \
        .stubs()                                                                                                       \
        .will(invoke(CaptureCopyGmToCbufAlignV2_##type));                                                              \
    MOCKER(copy_gm_to_cbuf_multi_nd2nz,                                                                                \
           void (*)(__cbuf__ type*, __gm__ type*, uint8_t, uint64_t, uint8_t, uint16_t, uint32_t, uint64_t, bool))     \
        .stubs()                                                                                                       \
        .will(invoke(CaptureCopyGmToCbufMultiND2nz_##type));                                                           \
    MOCKER(copy_gm_to_cbuf_multi_dn2nz,                                                                                \
           void (*)(__cbuf__ type*, __gm__ type*, uint8_t, uint64_t, uint8_t, uint16_t, uint32_t, uint64_t, bool))     \
        .stubs()                                                                                                       \
        .will(invoke(CaptureCopyGmToCbufMultiDN2nz_##type))

class Tensor_Api_Gm2L1 : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    virtual void SetUp()
    {
        ResetCapture();
        MOCKER_GM_TO_L1(uint8_t);
        MOCKER_GM_TO_L1(half);
        MOCKER_GM_TO_L1(uint16_t);
        MOCKER_GM_TO_L1(float);
        MOCKER_GM_TO_L1(uint32_t);
        MOCKER(set_mte2_nz_para, void (*)(uint64_t)).stubs().will(invoke(CaptureSetMTE2NzPara));
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
    }

private:
    template <typename T>
    void InitializeData()
    {
        using CastT = Std::conditional_t<sizeof(T) == 1, uint8_t, T>;
        std::iota(reinterpret_cast<CastT*>(src0Gm), reinterpret_cast<CastT*>(src0Gm + GmSize), static_cast<CastT>(1));
        std::fill(reinterpret_cast<CastT*>(l1ABuf), reinterpret_cast<CastT*>(l1ABuf + L1Size), static_cast<CastT>(1));
        std::fill(reinterpret_cast<CastT*>(l1ABufGolden), reinterpret_cast<CastT*>(l1ABufGolden + L1Size),
                  static_cast<CastT>(1));
    }

private:
    constexpr static uint32_t GmSize = 2 * 1024 * 1024; // 2MB GM buffer size
    constexpr static uint32_t L1Size = 512 * 1024;      // 512KB L1 buffer size
    __gm__ uint8_t src0Gm[GmSize] = {0};
    __cbuf__ uint8_t l1ABuf[L1Size] = {0};
    __cbuf__ uint8_t l1ABufGolden[L1Size] = {0};
};

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationND2ND)
{
    using T = uint16_t;

    {
        constexpr uint32_t singleM = 17;
        constexpr uint32_t singleK = 18;

        constexpr uint32_t baseM = 19;
        constexpr uint32_t baseK = 32;

        auto gmALayout = MakeNDLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeNDLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
        SimND2ND(l1ATensorGolden, gmA);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
    {
        constexpr uint32_t singleM = 17;
        constexpr uint32_t singleK = 18;

        constexpr uint32_t baseM = 17;
        constexpr uint32_t baseK = 18;

        auto gmALayout = MakeNDLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeNDLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
        SimND2ND(l1ATensorGolden, gmA);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationND2NDWithCoord)
{
    using T = float;

    constexpr uint32_t singleM = 33;
    constexpr uint32_t singleK = 25;

    constexpr uint32_t baseM = 19;
    constexpr uint32_t baseK = 32;

    auto gmALayout = MakeNDLayout<T>(singleM, singleK);
    auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

    auto l1ALayout = MakeNDLayout<T>(baseM, baseK);
    auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
    auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

    auto coord = MakeCoord(10, 10);
    InitializeData<T>();
    atomCopy.Call(l1ATensor, gmA, coord);
    SimND2ND(l1ATensorGolden, gmA, coord);
    EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationND2ND1Dim)
{
    using T = float;

    constexpr uint32_t singleM = 1;
    constexpr uint32_t singleK = 17;

    constexpr uint32_t baseM = 1;
    constexpr uint32_t baseK = 19;

    auto gmALayout = MakeNDLayout<T>(singleM, singleK);
    auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

    auto l1ALayout = MakeNDLayout<T>(baseM, baseK);
    auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
    auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

    InitializeData<T>();
    atomCopy.Call(l1ATensor, gmA);
    SimND2ND(l1ATensorGolden, gmA);
    EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationND2ND1DimInt)
{
    using T = float;

    constexpr auto singleM = Std::Int<1>();
    constexpr uint32_t singleK = 17;

    constexpr auto baseM = Std::Int<1>();
    constexpr uint32_t baseK = 19;

    // auto gmALayout = MakeNDLayout<T>(singleM, singleK); // current have a bug
    auto gmALayout = LayoutConstructor(Std::Int<1>{}, singleM, Std::Int<1>{}, singleK, Std::Int<0>{}, singleK,
                                       Std::Int<0>{}, Std::Int<1>{});
    auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

    auto l1ALayout = MakeNDLayout<T>(baseM, baseK);
    auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
    auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

    InitializeData<T>();
    atomCopy.Call(l1ATensor, gmA);
    SimND2ND(l1ATensorGolden, gmA);
    EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationND2Nz)
{
    using T = float;

    constexpr uint32_t singleM = 18;
    constexpr uint32_t singleK = 17;

    constexpr uint32_t baseM = 19;
    constexpr uint32_t baseK = 18;

    auto gmALayout = MakeNDLayout<T>(singleM, singleK);
    auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

    auto l1ALayout = MakeNzLayout<T>(baseM, baseK);
    auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
    auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

    InitializeData<T>();
    atomCopy.Call(l1ATensor, gmA);
    SimND2Nz(l1ATensorGolden, gmA);
    EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationND2NzFp4)
{
    using T = fp4x2_e2m1_t;

    {
        constexpr uint32_t singleM = 18;
        constexpr uint32_t singleK = 18;

        constexpr uint32_t baseM = 19;
        constexpr uint32_t baseK = 20;

        auto gmALayout = MakeNDLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeNzLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
        SimND2Nz(l1ATensorGolden, gmA);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
    {
        constexpr uint32_t singleM = 68;
        constexpr uint32_t singleK = 68;

        constexpr uint32_t baseM = 69;
        constexpr uint32_t baseK = 70;

        auto gmALayout = MakeNDLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeNzLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
        SimND2Nz(l1ATensorGolden, gmA);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationND2NzB8)
{
    using T = uint8_t;

    constexpr uint32_t singleM = 18;
    constexpr uint32_t singleK = 9;

    constexpr uint32_t baseM = 19;
    constexpr uint32_t baseK = 10;

    auto gmALayout = MakeNDLayout<T>(singleM, singleK);
    auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

    auto l1ALayout = MakeNzLayout<T>(baseM, baseK);
    auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
    auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

    InitializeData<T>();
    atomCopy.Call(l1ATensor, gmA);
    SimND2Nz(l1ATensorGolden, gmA);
    EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationND2NzWithCoord)
{
    using T = float;

    constexpr uint32_t singleM = 33;
    constexpr uint32_t singleK = 25;

    constexpr uint32_t baseM = 19;
    constexpr uint32_t baseK = 18;

    auto gmALayout = MakeNDLayout<T>(singleM, singleK);
    auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

    auto l1ALayout = MakeNzLayout<T>(baseM, baseK);
    auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
    auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

    {
        InitializeData<T>();
        auto coord = MakeCoord(0, 0);
        atomCopy.Call(l1ATensor, gmA, coord);
        SimND2Nz(l1ATensorGolden, gmA, coord);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
    {
        InitializeData<T>();
        auto coord = MakeCoord(10, 10);
        atomCopy.Call(l1ATensor, gmA, coord);
        SimND2Nz(l1ATensorGolden, gmA, coord);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationND2Zn)
{
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 32;

    __gm__ float src0Gm[M * K] = {0};
    __cbuf__ float l1ABuf[M * K] = {0};

    auto gmALayout = MakeNDLayout<float>(M, K);
    auto gmA = MakeTensor(MakeGMmemPtr(src0Gm), gmALayout);

    auto l1ALayout = MakeZnLayout<float>(M, K);
    auto l1ATensor = MakeTensor(MakeL1memPtr(l1ABuf), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});
    atomCopy.Call(l1ATensor, gmA);
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationDN2Nz)
{
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 32;

    __gm__ float src0Gm[M * K] = {0};
    __cbuf__ float l1ABuf[M * K] = {0};

    auto gmALayout = MakeDNLayout<float>(M, K);
    auto gmA = MakeTensor(MakeGMmemPtr(src0Gm), gmALayout);

    auto l1ALayout = MakeNzLayout<float>(M, K);
    auto l1ATensor = MakeTensor(MakeL1memPtr(l1ABuf), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});
    atomCopy.Call(l1ATensor, gmA);
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationDN2NzB32)
{
    using T = float;

    constexpr uint32_t singleM = 17;
    constexpr uint32_t singleK = 18;

    constexpr uint32_t baseM = 18;
    constexpr uint32_t baseK = 19;

    auto gmALayout = MakeDNLayout<T>(singleM, singleK);
    auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

    auto l1ALayout = MakeNzLayout<T>(baseM, baseK);
    auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
    auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

    InitializeData<T>();
    atomCopy.Call(l1ATensor, gmA);
    SimDN2Nz(l1ATensorGolden, gmA);
    EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationDN2Zn)
{
    constexpr uint32_t M = 16;
    constexpr uint32_t K = 32;

    __gm__ float src0Gm[M * K] = {0};
    __cbuf__ float l1ABuf[M * K] = {0};

    auto gmALayout = MakeDNLayout<float>(M, K);
    auto gmA = MakeTensor(MakeGMmemPtr(src0Gm), gmALayout);

    auto l1ALayout = MakeZnLayout<float>(M, K);
    auto l1ATensor = MakeTensor(MakeL1memPtr(l1ABuf), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});
    atomCopy.Call(l1ATensor, gmA);
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationDN2ZnB32)
{
    using T = float;

    constexpr uint32_t singleM = 18;
    constexpr uint32_t singleK = 9;

    constexpr uint32_t baseM = 19;
    constexpr uint32_t baseK = 10;

    auto gmALayout = MakeDNLayout<T>(singleM, singleK);
    auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

    auto l1ALayout = MakeZnLayout<T>(baseM, baseK);
    auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
    auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

    auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

    InitializeData<T>();
    atomCopy.Call(l1ATensor, gmA);
    SimDN2Zn(l1ATensorGolden, gmA);
    EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationDN2ZnFp4)
{
    using T = fp4x2_e2m1_t;

    {
        constexpr uint32_t singleM = 18;
        constexpr uint32_t singleK = 18;

        constexpr uint32_t baseM = 19;
        constexpr uint32_t baseK = 20;

        auto gmALayout = MakeDNLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeZnLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
        SimDN2Zn(l1ATensorGolden, gmA);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
    {
        constexpr uint32_t singleM = 68;
        constexpr uint32_t singleK = 68;

        constexpr uint32_t baseM = 69;
        constexpr uint32_t baseK = 70;

        auto gmALayout = MakeDNLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeZnLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
        SimDN2Zn(l1ATensorGolden, gmA);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationNz2Nz)
{
    using T = uint16_t;
    {
        constexpr uint32_t singleM = 17;
        constexpr uint32_t singleK = 18;

        constexpr uint32_t baseM = 19 * 2;
        constexpr uint32_t baseK = 20 * 2;

        auto gmALayout = MakeNzLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeNzLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
    }
    {
        constexpr uint32_t singleM = 13;
        constexpr uint32_t singleK = 18;

        constexpr uint32_t baseM = 14;
        constexpr uint32_t baseK = 20;

        auto gmALayout = MakeNzLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeNzLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
    }
    {
        constexpr uint32_t singleM = 17;
        constexpr uint32_t singleK = 18;

        constexpr uint32_t baseM = 19;
        constexpr uint32_t baseK = 20;

        auto gmALayout = MakeNzLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeNzLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
    }
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationScaleAND2Zz)
{
    using T = fp8_e8m0_t;
    {
        constexpr uint32_t singleM = 18;
        constexpr uint32_t singleK = 34;

        constexpr uint32_t baseM = 19;
        constexpr uint32_t baseK = 36;

        auto gmALayout = MakeScaleANDLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeZzLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        // atomCopy.Call(l1ATensor, gmA);
        // SimScaleAND2Zz(l1ATensorGolden, gmA);
        // EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
    {
        constexpr uint32_t singleM = 32;
        constexpr uint32_t singleK = 32;

        constexpr uint32_t baseM = 32;
        constexpr uint32_t baseK = 32;

        auto gmALayout = MakeScaleANDLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeZzLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        // atomCopy.Call(l1ATensor, gmA);
        // SimScaleAND2Zz(l1ATensorGolden, gmA);
        // EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
}

TEST_F(Tensor_Api_Gm2L1, CopyGm2L1OperationScaleADN2Zz)
{
    using T = fp8_e8m0_t;
    {
        constexpr uint32_t singleM = 18;
        constexpr uint32_t singleK = 34;

        constexpr uint32_t baseM = 19;
        constexpr uint32_t baseK = 36;

        auto gmALayout = MakeScaleADNLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeZzLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
        SimScaleADN2Zz(l1ATensorGolden, gmA);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
    {
        constexpr uint32_t singleM = 32;
        constexpr uint32_t singleK = 32;

        constexpr uint32_t baseM = 32;
        constexpr uint32_t baseK = 32;

        auto gmALayout = MakeScaleADNLayout<T>(singleM, singleK);
        auto gmA = MakeTensor(MakeGMmemPtr(reinterpret_cast<T*>(src0Gm)), gmALayout);

        auto l1ALayout = MakeZzLayout<T>(baseM, baseK);
        auto l1ATensor = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABuf)), l1ALayout);
        auto l1ATensorGolden = MakeTensor(MakeL1memPtr(reinterpret_cast<T*>(l1ABufGolden)), l1ALayout);

        auto atomCopy = MakeCopy(CopyGM2L1{}, DataCopyTraitDefault{});

        InitializeData<T>();
        atomCopy.Call(l1ATensor, gmA);
        SimScaleADN2Zz(l1ATensorGolden, gmA);
        EXPECT_TRUE(std::equal(l1ABuf, l1ABuf + L1Size, l1ABufGolden));
    }
}
