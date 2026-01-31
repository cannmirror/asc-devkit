/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#ifndef SIMT_COMPILER_STUB_H
#define SIMT_COMPILER_STUB_H

#include "simt_api/dav_c310/kernel_simt_cpu.h"
#include "kernel_vectorized.h"
#include "simt_stub.h"
#include "stub_def.h"
#include "stub_fun.h"

#include <cstdint>

namespace bisheng {
namespace cce {
namespace simt {

int32_t get_block_idx();

int32_t get_block_num();

}
}
}

enum L1CacheType : uint32_t { NON_CACHEABLE = 0, CACHEABLE = 1 };
enum class LD_L2CacheType : uint32_t { L2_CACHE_HINT_NORMAL_FV = 0 };
enum class ST_L2CacheType : uint32_t { L2_CACHE_HINT_NORMAL_FV = 0 };

template <LD_L2CacheType L2Cache = LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
          L1CacheType L1CacheType = L1CacheType::NON_CACHEABLE, typename T>
T __ldg(__gm__ T* address)
{
    return *address;
}
template <ST_L2CacheType L2Cache = ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
          L1CacheType L1CacheType = L1CacheType::NON_CACHEABLE, typename T>
void __stg(__gm__ T* address, T val)
{
    *address = val;
}

int32_t asc_get_block_idx();

int32_t asc_get_block_num();

long int asc_ldcg(long int* address);

unsigned long int asc_ldcg(unsigned long int* address);

long long int asc_ldcg(long long int* address);

unsigned long long int asc_ldcg(unsigned long long int* address);

long2 asc_ldcg(long2* address);

ulong2 asc_ldcg(ulong2* address);

long4 asc_ldcg(long4* address);

inline ulong4 asc_ldcg(ulong4* address);

longlong2 asc_ldcg(longlong2* address);

ulonglong2 asc_ldcg(ulonglong2* address);

longlong4 asc_ldcg(longlong4* address);

ulonglong4 asc_ldcg(ulonglong4* address);

signed char asc_ldcg(signed char* address);

unsigned char asc_ldcg(unsigned char* address);

char2 asc_ldcg(char2* address);

uchar2 asc_ldcg(uchar2* address);

char4 asc_ldcg(char4* address);

uchar4 asc_ldcg(uchar4* address);

short asc_ldcg(short* address);

unsigned short asc_ldcg(unsigned short* address);

short2 asc_ldcg(short2* address);

ushort2 asc_ldcg(ushort2* address);

short4 asc_ldcg(short4* address);

ushort4 asc_ldcg(ushort4* address);

int asc_ldcg(int* address);

unsigned int asc_ldcg(unsigned int* address);

int2 asc_ldcg(int2* address);

uint2 asc_ldcg(uint2* address);

int4 asc_ldcg(int4* address);

uint4 asc_ldcg(uint4* address);

float asc_ldcg(float* address);

float2 asc_ldcg(float2* address);

float4 asc_ldcg(float4* address);

long int asc_ldca(long int* address);

unsigned long int asc_ldca(unsigned long int* address);

long long int asc_ldca(long long int* address);

unsigned long long int asc_ldca(unsigned long long int* address);

long2 asc_ldca(long2* address);

ulong2 asc_ldca(ulong2* address);

long4 asc_ldca(long4* address);

ulong4 asc_ldca(ulong4* address);

longlong2 asc_ldca(longlong2* address);

ulonglong2 asc_ldca(ulonglong2* address);

longlong4 asc_ldca(longlong4* address);

ulonglong4 asc_ldca(ulonglong4* address);

signed char asc_ldca(signed char* address);

unsigned char asc_ldca(unsigned char* address);

char2 asc_ldca(char2* address);

uchar2 asc_ldca(uchar2* address);

char4 asc_ldca(char4* address);

uchar4 asc_ldca(uchar4* address);

short asc_ldca(short* address);

unsigned short asc_ldca(unsigned short* address);

short2 asc_ldca(short2* address);

ushort2 asc_ldca(ushort2* address);

short4 asc_ldca(short4* address);

ushort4 asc_ldca(ushort4* address);

int asc_ldca(int* address);

unsigned int asc_ldca(unsigned int* address);

int2 asc_ldca(int2* address);

uint2 asc_ldca(uint2* address);

int4 asc_ldca(int4* address);

uint4 asc_ldca(uint4* address);

float asc_ldca(float* address);

float2 asc_ldca(float2* address);

float4 asc_ldca(float4* address);

void asc_stcg(long int* address, long int val);

void asc_stcg(unsigned long int* address, unsigned long int val);

void asc_stcg(long long int* address, long long int val);

void asc_stcg(unsigned long long int* address, unsigned long long int val);

void asc_stcg(long2* address, long2 val);

void asc_stcg(ulong2* address, ulong2 val);

void asc_stcg(long4* address, long4 val);

void asc_stcg(ulong4* address, ulong4 val);

void asc_stcg(longlong2* address, longlong2 val);

void asc_stcg(ulonglong2* address, ulonglong2 val);

void asc_stcg(longlong4* address, longlong4 val);

void asc_stcg(ulonglong4* address, ulonglong4 val);

void asc_stcg(signed char* address, signed char val);

void asc_stcg(unsigned char* address, unsigned char val);

void asc_stcg(char2* address, char2 val);

void asc_stcg(uchar2* address, uchar2 val);

void asc_stcg(char4* address, char4 val);

void asc_stcg(uchar4* address, uchar4 val);

void asc_stcg(short* address, short val);

void asc_stcg(unsigned short* address, unsigned short val);

void asc_stcg(short2* address, short2 val);

void asc_stcg(ushort2* address, ushort2 val);

void asc_stcg(short4* address, short4 val);

void asc_stcg(ushort4* address, ushort4 val);

void asc_stcg(int* address, int val);

void asc_stcg(unsigned int* address, unsigned int val);

void asc_stcg(int2* address, int2 val);

void asc_stcg(uint2* address, uint2 val);

void asc_stcg(int4* address, int4 val);

void asc_stcg(uint4* address, uint4 val);

void asc_stcg(float* address, float val);

void asc_stcg(float2* address, float2 val);

void asc_stcg(float4* address, float4 val);

void asc_stwt(long int* address, long int val);

void asc_stwt(unsigned long int* address, unsigned long int val);

void asc_stwt(long long int* address, long long int val);

void asc_stwt(unsigned long long int* address, unsigned long long int val);

void asc_stwt(long2* address, long2 val);

void asc_stwt(ulong2* address, ulong2 val);

void asc_stwt(long4* address, long4 val);

void asc_stwt(ulong4* address, ulong4 val);

void asc_stwt(longlong2* address, longlong2 val);

void asc_stwt(ulonglong2* address, ulonglong2 val);

void asc_stwt(longlong4* address, longlong4 val);

void asc_stwt(ulonglong4* address, ulonglong4 val);

void asc_stwt(signed char* address, signed char val);

void asc_stwt(unsigned char* address, unsigned char val);

void asc_stwt(char2* address, char2 val);

void asc_stwt(uchar2* address, uchar2 val);

void asc_stwt(char4* address, char4 val);

void asc_stwt(uchar4* address, uchar4 val);

void asc_stwt(short* address, short val);

void asc_stwt(unsigned short* address, unsigned short val);

void asc_stwt(short2* address, short2 val);

void asc_stwt(ushort2* address, ushort2 val);

void asc_stwt(short4* address, short4 val);

void asc_stwt(ushort4* address, ushort4 val);

void asc_stwt(int* address, int val);

void asc_stwt(unsigned int* address, unsigned int val);

void asc_stwt(int2* address, int2 val);

void asc_stwt(uint2* address, uint2 val);

void asc_stwt(int4* address, int4 val);

void asc_stwt(uint4* address, uint4 val);

void asc_stwt(float* address, float val);

void asc_stwt(float2* address, float2 val);

void asc_stwt(float4* address, float4 val);

half2 h2exp(half2 x);

half2 h2log(half2 x);

half2 h2sqrt(half2 x);

half2 h2rsqrt(half2 x);

#endif