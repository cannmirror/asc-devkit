#!/bin/bash
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

set -e

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/build_out
CANN_3RD_LIB_PATH=${BUILD_DIR}
USER_ID=$(id -u)
CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)))
JOB_NUM="-j${CPU_NUM}"
ASAN="false"
COV="false"

if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/ascend-toolkit/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
fi

function log() {
    local current_time=`date +"%Y-%m-%d %H:%M:%S"`
    echo "[$current_time] "$1
}

function set_env()
{
    source $ASCEND_CANN_PACKAGE_PATH/bin/setenv.bash || echo "0"
}

function clean()
{
    if [ -n "${BUILD_DIR}" ];then
        rm -rf ${BUILD_DIR}
    fi

    if [ -z "${TEST}" ];then
        if [ -n "${OUTPUT_DIR}" ];then
            rm -rf ${OUTPUT_DIR}
        fi
    fi

    mkdir -p ${BUILD_DIR} ${OUTPUT_DIR}
}

function cmake_config()
{
    local extra_option="$1"
    log "Info: cmake config ${CUSTOM_OPTION} ${extra_option} ."
    cmake ..  ${CUSTOM_OPTION} ${extra_option}
}

function build()
{
    local target="$1"
    cmake --build . --target ${target} ${JOB_NUM} --verbose
}

function build_package(){
    cmake_config
    build package
    cp ${BUILD_DIR}/_CPack_Packages/makeself_staging/*.run ${OUTPUT_DIR}
}

function build_test() {
    cmake_config
    build all
}

while [[ $# -gt 0 ]]; do
    case $1 in
    --ccache)
        CCACHE_PROGRAM="$2"
        shift 2
        ;;
    -p|--package-path)
        ascend_package_path="$2"
        shift 2
        ;;
    -t|--test)
        TEST="all"
        shift
        ;;
    --asan)
        ASAN="true"
        shift
        ;;
    --cov)
        COV="true"
        shift
        ;;
    --pkg)
        PKG="true"
        shift
        ;;
    --cann_3rd_lib_path=*)
        CANN_3RD_LIB_PATH="${1#*=}"
        shift
        ;;
    *)
        break
        ;;
    esac
done

if [ -n "${TEST}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_TEST=ON"
fi

if [ "${ASAN}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_ASAN=true"
fi

if [ "${COV}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_GCOV=true"
fi

if [ "${PKG}" == "true" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DPACKAGE_OPEN_PROJECT=ON"
fi

if [ -n "${CANN_3RD_LIB_PATH}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
fi

if [ -n "${ascend_package_path}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ascend_package_path}
elif [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
else
    log "Error: Please set the toolkit package installation directory through parameter -p|--package-path."
    exit 1
fi

CUSTOM_OPTION="${CUSTOM_OPTION} -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}"

set_env

clean

cd ${BUILD_DIR}

if [ -n "${TEST}" ];then
    build_test
else
    build_package
fi
