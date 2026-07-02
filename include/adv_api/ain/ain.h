/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ain.h
 * \brief Ain interface
 */
#if !defined(__ASCENDC_INCLUDE_INTERNAL_HEADERS__)
#define __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#define __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_AIN_H__
#endif

#ifndef INCLUDE_ADV_API_AIN_AIN_H
#define INCLUDE_ADV_API_AIN_AIN_H

#include "../hcomm/hcomm.h"
#include "ain_common.h"

namespace AscendC {

/*!
 * @class Ain
 * @brief This class provides device-side one-sided communication primitives (put/get) layered on top
 *        of the Hcomm point-to-point engine. It resolves the per-peer communication channel from the
 *        device communication table, translates symmetric-window handles into remote/local addresses,
 *        and either rings the doorbell immediately or leaves tasks to be completed by a later flush().
 */
class Ain {
public:
    /*!
     * @brief Construct an Ain instance bound to a device communication context.
     * @param [in] devComm: Device communication context.
     * @param [in] contextIndex: Index of the communication context to operate on.
     */
    template <unsigned commEngineMask = AIN_MASK_ALL>
    AIN_DEVICE Ain(AinDevComm devComm, int contextIndex);
    AIN_DEVICE ~Ain();

    /*!
     * @brief Drain every peer channel in the communication table.
     */
    AIN_DEVICE void flush();

    /*!
     * @brief Issue a one-sided write (put) from a local window to a remote peer's window.
     */
    template <
        typename RemoteAction = AinRemoteNone, typename LocalAction = AinLocalNone,
        typename DescriptorUbuf = AinDescriptorUbuf>
    AIN_DEVICE void put(
        AinTeam team, uint32_t peer, AinCommSymWindow dstWin, uint64_t dstOffset, AinCommSymWindow srcWin,
        uint64_t srcOffset, uint64_t bytes, RemoteAction remoteAction = RemoteAction{},
        LocalAction localAction = LocalAction{}, const DescriptorUbuf& ubuf = DescriptorUbuf{},
        AinCommitFlags commitFlags = AIN_COMMIT_IMMED);

    /*!
     * @brief Issue a one-sided read (get) from a remote peer's window into a local window.
     */
    template <typename DescriptorUbuf = AinDescriptorUbuf>
    AIN_DEVICE void get(
        AinTeam team, uint32_t peer, AinCommSymWindow dstWin, uint64_t dstOffset, AinCommSymWindow srcWin,
        uint64_t srcOffset, uint64_t bytes, const DescriptorUbuf& ubuf = DescriptorUbuf{},
        AinCommitFlags commitFlags = AIN_COMMIT_IMMED);

private:
    uint64_t devComm_ = 0;
    int contextIndex_ = 0;
    AscendC::Hcomm<AscendC::CommProtocol::COMM_PROTOCOL_UBC_CTP> hcomm_;
};

} // namespace AscendC

#include "../../../impl/adv_api/detail/ain/impl/ain_impl.h"

#endif // INCLUDE_ADV_API_AIN_AIN_H

#if defined(__UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_AIN_H__)
#undef __ASCENDC_INCLUDE_INTERNAL_HEADERS__
#undef __UNDEF_ASCENDC_INCLUDE_INTERNAL_HEADERS_AIN_H__
#endif
