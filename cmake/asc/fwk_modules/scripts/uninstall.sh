#!/bin/bash
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

vendor_name=customize

sourcedir=$PWD/packages
vendordir=vendors/$vendor_name

log() {
    cur_date=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[ops_custom] [$cur_date] "$1
}

if [[ "x${ASCEND_CUSTOM_OPP_PATH}" != "x" ]]; then
    targetdir=${ASCEND_CUSTOM_OPP_PATH}
    if [[ ! -d ${targetdir} ]]; then
        log "[ERROR] The directory specified by ASCEND_CUSTOM_OPP_PATH does not exist."
        exit 1
    fi
elif [[ "x${ASCEND_OPP_PATH}" != "x" ]]; then
    targetdir=${ASCEND_OPP_PATH}
    if [[ ! -d ${targetdir} ]]; then
        log "[ERROR] The directory specified by ASCEND_OPP_PATH does not exist."
        exit 1
    fi
else
    log "[ERROR] The environment variables ASCEND_CUSTOM_OPP_PATH and ASCEND_OPP_PATH are no set."
    exit 1
fi

dir_to_delete=$targetdir/$vendordir

if [[ ! -d $dir_to_delete ]]; then
    log "[INFO] no need to delete ops $dir_to_delete files"
    exit 1
else
    log "[INFO] Starting to delete $vendor_name ..."
    chmod u+x -R $dir_to_delete
    rm -rf $dir_to_delete
    if [ $? -ne 0 ]; then
        log "[INFO] Failed to delete $vendor_name."
        exit 1
    fi
fi

config_file=${targetdir}/vendors/config.ini
found_vendors=$(printf '%s\n' "$vendor_name" | sed 's/[[\.*^$()+?{|]/\\&/g; s/]/\\]/g')
sed -i "/load_priority=$found_vendors/d" "$config_file"

log "[INFO] Successfully deteled $vendor_name."
exit 0