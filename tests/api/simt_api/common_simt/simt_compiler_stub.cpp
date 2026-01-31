/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "simt_compiler_stub.h"
#include "kernel_process_lock.h"
#include "kernel_utils.h"
#include "dav_c310/kernel_simt_cpu.h"
#include "stub_def.h"
#include <cmath>

namespace bisheng {
namespace cce {
namespace simt {

int32_t get_block_idx()
{
    return block_idx;
}

int32_t get_block_num()
{
    return block_num;
}

}
}
}

int32_t asc_get_block_idx()
{
    return bisheng::cce::simt::get_block_idx();
}

int32_t asc_get_block_num()
{
    return bisheng::cce::simt::get_block_num();
}

long int asc_ldcg(long int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

unsigned long int asc_ldcg(unsigned long int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

long long int asc_ldcg(long long int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
}

unsigned long long int asc_ldcg(unsigned long long int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((uint64_t*)address);
}

long2 asc_ldcg(long2* address)
{
    longlong2 t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((longlong2*)address);
    return (long2&)t;
}

ulong2 asc_ldcg(ulong2* address)
{
    longlong2 t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((longlong2*)address);
    return (ulong2&)t;
}

long4 asc_ldcg(long4* address)
{
    longlong4 t;
    t.x = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
    t.y = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 1);
    t.z = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 2);
    t.w = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 3);
    return (long4&)t;
}

ulong4 asc_ldcg(ulong4* address)
{
    longlong4 t;
    t.x = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
    t.y = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 1);
    t.z = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 2);
    t.w = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 3);
    return (ulong4&)t;
}

longlong2 asc_ldcg(longlong2* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

ulonglong2 asc_ldcg(ulonglong2* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

longlong4 asc_ldcg(longlong4* address)
{
    longlong4 t;
    t.x = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
    t.y = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 1);
    t.z = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 2);
    t.w = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 3);
    return t;
}

ulonglong4 asc_ldcg(ulonglong4* address)
{
    longlong4 t;
    t.x = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
    t.y = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 1);
    t.z = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 2);
    t.w = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 3);
    return (ulonglong4&)t;
}

signed char asc_ldcg(signed char* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int8_t*)address);
}

unsigned char asc_ldcg(unsigned char* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int8_t*)address);
}

char2 asc_ldcg(char2* address)
{
    int16_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int16_t*)address);
    return (char2&)t;
}

uchar2 asc_ldcg(uchar2* address)
{
    int16_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int16_t*)address);
    return (uchar2&)t;
}

char4 asc_ldcg(char4* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address);
    return (char4&)t;
}

uchar4 asc_ldcg(uchar4* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address);
    return (uchar4&)t;
}

short asc_ldcg(short* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

unsigned short asc_ldcg(unsigned short* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

short2 asc_ldcg(short2* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address);
    return (short2&)t;
}

ushort2 asc_ldcg(ushort2* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address);
    return (ushort2&)t;
}
short4 asc_ldcg(short4* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
    return (short4&)t;
}

ushort4 asc_ldcg(ushort4* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
    return (ushort4&)t;
}

int asc_ldcg(int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

unsigned int asc_ldcg(unsigned int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

int2 asc_ldcg(int2* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
    return (int2&)t;
}

uint2 asc_ldcg(uint2* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
    return (uint2&)t;
}

int4 asc_ldcg(int4* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

uint4 asc_ldcg(uint4* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

float asc_ldcg(float* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

float2 asc_ldcg(float2* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address);
    return (float2&)t;
}

float4 asc_ldcg(float4* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address);
}

long int asc_ldca(long int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

unsigned long int asc_ldca(unsigned long int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

long long int asc_ldca(long long int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
}

unsigned long long int asc_ldca(unsigned long long int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((uint64_t*)address);
}

long2 asc_ldca(long2* address)
{
    longlong2 t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((longlong2*)address);
    return (long2&)t;
}

ulong2 asc_ldca(ulong2* address)
{
    longlong2 t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((longlong2*)address);
    return (ulong2&)t;
}

long4 asc_ldca(long4* address)
{
    long4 t;
    t.x = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
    t.y = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 1);
    t.z = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 2);
    t.w = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 3);
    return t;
}

ulong4 asc_ldca(ulong4* address)
{
    ulong4 t;
    t.x = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
    t.y = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 1);
    t.z = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 2);
    t.w = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 3);
    return t;
}

longlong2 asc_ldca(longlong2* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

ulonglong2 asc_ldca(ulonglong2* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

longlong4 asc_ldca(longlong4* address)
{
    longlong4 t;
    t.x = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
    t.y = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 1);
    t.z = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 2);
    t.w = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 3);
    return t;
}

ulonglong4 asc_ldca(ulonglong4* address)
{
    longlong4 t;
    t.x = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
    t.y = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 1);
    t.z = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 2);
    t.w = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 3);
    return (ulonglong4&)t;
}

signed char asc_ldca(signed char* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

unsigned char asc_ldca(unsigned char* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

char2 asc_ldca(char2* address)
{
    int16_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int16_t*)address);
    return (char2&)t;
}

uchar2 asc_ldca(uchar2* address)
{
    int16_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int16_t*)address);
    return (uchar2&)t;
}

char4 asc_ldca(char4* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address);
    return (char4&)t;
}

uchar4 asc_ldca(uchar4* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address);
    return (uchar4&)t;
}

short asc_ldca(short* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

unsigned short asc_ldca(unsigned short* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

short2 asc_ldca(short2* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address);
    return (short2&)t;
}

ushort2 asc_ldca(ushort2* address)
{
    int32_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address);
    return (ushort2&)t;
}
short4 asc_ldca(short4* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
    return (short4&)t;
}

ushort4 asc_ldca(ushort4* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
    return (ushort4&)t;
}

int asc_ldca(int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

unsigned int asc_ldca(unsigned int* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

int2 asc_ldca(int2* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
    return (int2&)t;
}

uint2 asc_ldca(uint2* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
    return (uint2&)t;
}

int4 asc_ldca(int4* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

uint4 asc_ldca(uint4* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

float asc_ldca(float* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

float2 asc_ldca(float2* address)
{
    int64_t t = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address);
    return (float2&)t;
}

float4 asc_ldca(float4* address)
{
    return __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address);
}

void asc_stcg(long int* address, long int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(unsigned long int* address, unsigned long int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(long long int* address, long long int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stcg(unsigned long long int* address, unsigned long long int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((uint64_t*)address, (uint64_t&)val);
}

void asc_stcg(long2* address, long2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((longlong2*)address, (longlong2&)val);
}

void asc_stcg(ulong2* address, ulong2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((ulonglong2*)address, (ulonglong2&)val);
}

void asc_stcg(long4* address, long4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t)val.x);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 1, (int64_t)val.y);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 2, (int64_t)val.z);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 3, (int64_t)val.w);
}

void asc_stcg(ulong4* address, ulong4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t)val.x);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 1, (int64_t)val.y);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 2, (int64_t)val.z);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 3, (int64_t)val.w);
}

void asc_stcg(longlong2* address, longlong2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(ulonglong2* address, ulonglong2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(longlong4* address, longlong4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t)val.x);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 1, (int64_t)val.y);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 2, (int64_t)val.z);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 3, (int64_t)val.w);
}

void asc_stcg(ulonglong4* address, ulonglong4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t)val.x);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 1, (int64_t)val.y);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 2, (int64_t)val.z);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address + 3, (int64_t)val.w);
}

void asc_stcg(signed char* address, signed char val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(unsigned char* address, unsigned char val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(char2* address, char2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int16_t*)address, (int16_t&)val);
}

void asc_stcg(uchar2* address, uchar2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int16_t*)address, (int16_t&)val);
}

void asc_stcg(char4* address, char4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address, (int32_t&)val);
}

void asc_stcg(uchar4* address, uchar4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address, (int32_t&)val);
}

void asc_stcg(short* address, short val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(unsigned short* address, unsigned short val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(short2* address, short2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address, (int32_t&)val);
}

void asc_stcg(ushort2* address, ushort2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int32_t*)address, (int32_t&)val);
}

void asc_stcg(short4* address, short4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stcg(ushort4* address, ushort4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stcg(int* address, int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(unsigned int* address, unsigned int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(int2* address, int2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stcg(uint2* address, uint2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stcg(int4* address, int4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(uint4* address, uint4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(float* address, float val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stcg(float2* address, float2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stcg(float4* address, float4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(address, val);
}

void asc_stwt(long int* address, long int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(unsigned long int* address, unsigned long int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(long long int* address, long long int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stwt(unsigned long long int* address, unsigned long long int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((uint64_t*)address, (uint64_t&)val);
}

void asc_stwt(long2* address, long2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((longlong2*)address, (longlong2&)val);
}

void asc_stwt(ulong2* address, ulong2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((ulonglong2*)address, (ulonglong2&)val);
}

void asc_stwt(long4* address, long4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address, (int64_t)val.x);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 1, (int64_t)val.y);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 2, (int64_t)val.z);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 3, (int64_t)val.w);
}

void asc_stwt(ulong4* address, ulong4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((uint64_t*)address, (uint64_t)val.x);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((uint64_t*)address + 1, (uint64_t)val.y);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((uint64_t*)address + 2, (uint64_t)val.z);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((uint64_t*)address + 3, (uint64_t)val.w);
}

void asc_stwt(longlong2* address, longlong2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(ulonglong2* address, ulonglong2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(longlong4* address, longlong4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address, (int64_t)val.x);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 1, (int64_t)val.y);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 2, (int64_t)val.z);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 3, (int64_t)val.w);
}

void asc_stwt(ulonglong4* address, ulonglong4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address, (int64_t)val.x);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 1, (int64_t)val.y);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 2, (int64_t)val.z);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address + 3, (int64_t)val.w);
}

void asc_stwt(signed char* address, signed char val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(unsigned char* address, unsigned char val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(char2* address, char2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int16_t*)address, (int16_t&)val);
}

void asc_stwt(uchar2* address, uchar2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int16_t*)address, (int16_t&)val);
}

void asc_stwt(char4* address, char4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address, (int32_t&)val);
}

void asc_stwt(uchar4* address, uchar4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address, (int32_t&)val);
}

void asc_stwt(short* address, short val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(unsigned short* address, unsigned short val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(short2* address, short2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address, (int32_t&)val);
}

void asc_stwt(ushort2* address, ushort2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int32_t*)address, (int32_t&)val);
}

void asc_stwt(short4* address, short4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stwt(ushort4* address, ushort4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stwt(int* address, int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(unsigned int* address, unsigned int val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(int2* address, int2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stwt(uint2* address, uint2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stwt(int4* address, int4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(uint4* address, uint4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(float* address, float val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

void asc_stwt(float2* address, float2 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>((int64_t*)address, (int64_t&)val);
}

void asc_stwt(float4* address, float4 val)
{
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::CACHEABLE>(address, val);
}

half2 h2exp(half2 x)
{
    half htmp1 = x.x;
    half htmp2 = x.y;
    float tmp1 = htmp1.ToFloat();
    float tmp2 = htmp2.ToFloat();
    tmp1 = exp(tmp1);
    tmp2 = exp(tmp2);
    htmp1 = half(tmp1);
    htmp2 = half(tmp2);
    return {htmp1, htmp2};
}

half2 h2log(half2 x)
{
    half htmp1 = x.x;
    half htmp2 = x.y;
    float tmp1 = htmp1.ToFloat();
    float tmp2 = htmp2.ToFloat();
    tmp1 = log(tmp1);
    tmp2 = log(tmp2);
    htmp1 = half(tmp1);
    htmp2 = half(tmp2);
    return {htmp1, htmp2};
}

half2 h2sqrt(half2 x)
{
    half htmp1 = x.x;
    half htmp2 = x.y;
    float tmp1 = htmp1.ToFloat();
    float tmp2 = htmp2.ToFloat();
    tmp1 = sqrt(tmp1);
    tmp2 = sqrt(tmp2);
    htmp1 = half(tmp1);
    htmp2 = half(tmp2);
    return {htmp1, htmp2};
}

half2 h2rsqrt(half2 x)
{
    half tmp1 = (half)1.0 / __sqrtf(x.x);
    half tmp2 = (half)1.0 / __sqrtf(x.y);
    return {tmp1, tmp2};
}