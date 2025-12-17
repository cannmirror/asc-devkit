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

set -e

SUPPORTED_SHORT_OPTS=("h" "j" "t" "p")
SUPPORTED_LONG_OPTS=(
    "help" "cov" "cache" "pkg" "asan" "make_clean" "cann_3rd_lib_path" "test" "cann_path" "adv_test" "basic_test_one" "basic_test_two" "basic_test_three" "build-type"
)

CURRENT_DIR=$(dirname $(readlink -f ${BASH_SOURCE[0]}))
BUILD_DIR=${CURRENT_DIR}/build
OUTPUT_DIR=${CURRENT_DIR}/build_out
CANN_3RD_LIB_PATH=${BUILD_DIR}
USER_ID=$(id -u)
CPU_NUM=$(($(cat /proc/cpuinfo | grep "^processor" | wc -l)))
THREAD_NUM=16
BUILD_TYPE="Release"

dotted_line="----------------------------------------------------------------"

function log() {
    local current_time=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$current_time] "$1
}

usage() {
  local specific_help="$1"

  if [[ -n "$specific_help" ]]; then
    case "$specific_help" in
      package)
        echo "Package Build Options:"
        echo $dotted_line
        echo "    --pkg                Compile run package"
        echo "    -p, --cann_path      Set the cann package installation directory, eg: /usr/local/Ascend/latest"
        echo "    -j                   Compile thread nums, default is 16, eg: -j 8"
        echo "    --cann_3rd_lib_path  Set the path for third-party library dependencies, eg: ./build"
        echo "    --asan               Enable ASAN (address Sanitizer)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --pkg -j 8"
        echo "    bash build.sh --pkg --asan -j 32"
        return
        ;;
      test)
        echo "Test Options:"
        echo $dotted_line
        echo "    -t, --test           Build and run all unit tests"
        echo "    -p, --cann_path      Set the cann package installation directory, eg: /usr/local/Ascend/latest"
        echo "    -j                   Compile thread nums, default is 16, eg: -j 8"
        echo "    --adv_test            Build and run the adv part of unit tests"
        echo "    --basic_test_one      Build and run the basic_one part of unit tests"
        echo "    --basic_test_two      Build and run the basic_two part of unit tests"
        echo "    --basic_test_three    Build and run the basic_three part of unit tests"
        echo "    --cann_3rd_lib_path  Set the path for third-party library dependencies, eg: ./build"
        echo "    --cov                Enable code coverage for unit tests"
        echo "    --asan               Enable ASAN (address Sanitizer)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh -t --cov"
        echo "    bash build.sh --test_part --asan -j 32"
        return
        ;;
      clean)
        echo "Clean Options:"
        echo $dotted_line
        echo "    --make_clean         Clean build artifacts"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh -t --make_clean"
        return
        ;;
    esac
  fi

  echo "build script for asc-devkit repository"
  echo "Usage: bash build.sh [OPTION]..."
  echo ""
  echo "    The following are all supported arguments:"
  echo $dotted_line
  echo "    -h, --help           Display help information"
  echo "    -j                   Compile thread nums, default is 16, eg: -j 8"
  echo "    -t, --test           Build and run all unit tests"
  echo "    -p, --cann_path      Set the cann package installation directory, eg: /usr/local/Ascend/latest"
  echo "    --adv_test            Build and run the adv part of unit tests"
  echo "    --basic_test_one      Build and run the basic_one part of unit tests"
  echo "    --basic_test_two      Build and run the basic_two part of unit tests"
  echo "    --basic_test_three    Build and run the basic_three part of unit tests"
  echo "    --pkg                Compile run package"
  echo "    --cann_3rd_lib_path  Set the path for third-party library dependencies, eg: ./build"
  echo "    --cov                Enable code coverage for unit tests"
  echo "    --asan               Enable ASAN (address Sanitizer)"
  echo "    --make_clean         Clean build artifacts"
  echo "    --build-type=<TYPE>"
  echo "                         Specify build type (TYPE options: Release/Debug), Default:Release"
}

check_option_validity() {
  local arg="$1"

  if [[ "$arg" =~ "=" ]]; then
    arg="${arg%%=*}"
  fi

  if [[ "$arg" =~ ^-[^-] ]]; then
    if [[ ! " ${SUPPORTED_SHORT_OPTS[@]} " =~ " ${arg:1} " ]]; then
      log "[ERROR] Invalid short option: ${arg}"
      return 1
    fi
  fi

  if [[ "$arg" =~ ^-- ]]; then
    if [[ ! " ${SUPPORTED_LONG_OPTS[@]} " =~ " ${arg:2} " ]]; then
      log "[ERROR] Invalid long option: ${arg}"
      return 1
    fi
  fi
  return 0
}

check_help_combinations() {
  local args=("$@")
  local has_test=false
  local test_part=''
  local has_cov=false
  local has_pkg=false

  for arg in "${arg[@]}"; do
    case "$arg" in
      -t|--test) has_test=true ;;
      --adv_test) test_part="adv_test" ;;
      --basic_test_one) test_part="basic_test_one" ;;
      --basic_test_two) test_part="basic_test_two" ;;
      --basic_test_three) test_part="basic_test_three" ;;
      --cov) has_cov=true ;;
      --pkg) has_pkg=true ;;
      -h|--help) ;;
    esac
  done

  if [[ "$has_test" == "true" && -n "$test_part" ]]; then
    log "[ERROR] --$test_part cannot be used with test(-t, --test)."
    return 1
  fi
  if [[ ("$has_test" == "true" || -n "$test_part") && "$has_pkg" == "true" ]]; then
    log "[ERROR] --pkg cannot be used with test(-t, --test, --$test_part)."
    return 1
  fi
  if [[ "$has_cov" == "true" && ("$has_test" == "true" || -n "$test_part") ]]; then
    log "[ERROR] --cov must be used with test(-t, --test, --$test_part)."
    return 1
  fi
  return 0
}

check_param_with_help() {
  for arg in "$@"; do
    if [[ "$arg" =~ ^- ]]; then
      if ! check_option_validity "$arg"; then
        log "[INFO] Use 'bash build.sh --help' for more information."
        exit 1
      fi
    fi
  done

  seen=()
  for arg in "$@"; do
    arg="${arg%%=*}"
    if [[ " ${seen[@]} " =~ " $arg " ]]; then
      log "[ERROR] $arg can only be input one."
      exit 1
    fi
    seen+=("$arg")
  done

  for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
      # 检查帮助信息中的组合参数
      check_help_combinations "$@"
      local comb_result=$?
      if [ $comb_result -eq 1 ]; then
        exit 1
      fi
      SHOW_HELP="general"

      for prev_arg in "$@"; do
        case "$prev_arg" in
          --pkg) SHOW_HELP="package" ;;
          -t|--test) SHOW_HELP="test" ;;
          --adv_test) SHOW_HELP="test" ;;
          --basic_test_one) SHOW_HELP="test" ;;
          --basic_test_two) SHOW_HELP="test" ;;
          --basic_test_three) SHOW_HELP="test" ;;
          --make_clean) SHOW_HELP="clean" ;;
        esac
      done

      usage "$SHOW_HELP"
      exit 0
    fi
  done
}

check_param_j() {
  if [[ ! $THREAD_NUM =~ ^-?[0-9]+$ ]]; then
   log "[ERROR] -j only support positive integers."
   exit 1
  fi

  if [[ "$THREAD_NUM" -gt "$CPU_NUM" ]]; then
    log "[WARNING] compile thread num:$THREAD_NUM over core num:$CPU_NUM, adjust to core num."
    THREAD_NUM=$CPU_NUM
  fi
}

check_param_clean() {
  if [[ "$#" -gt 1 || ( "$#" -eq 2 && $has_h == "true" ) ]]; then
    log "[ERROR] --make_clean must be used separately."
    exit 1
  fi
}

check_param_test_pkg() {
  if [[ "$TEST" == "all" && "$PKG" == "true" ]]; then
    log "[ERROR] --pkg cannot be used with test(-t, --test)."
    exit 1
  fi
}

check_param_test_part() {
  if [[ "$TEST" == "all" && -n "$TEST_PART" ]]; then
    log "[ERROR] --$TEST_PART cannot be used with test(-t, --test)."
    exit 1
  fi
}

check_param_cov() {
  if [[ "$COV" == "true" && "$TEST" != "all" ]]; then
    log "[ERROR] --cov must be used with test(-t, --test)."
    exit 1
  fi
}

set_options() {
  while [[ $# -gt 0 ]]; do
    case $1 in
    -h|--help)
      has_h="true"
      usage
      exit 0
      ;;
    --ccache)
      CCACHE_PROGRAM="$2"
      shift 2
      ;;
    -p=*|--cann_path=*)
      cann_path="${1#*=}"
      shift
      ;;
    -p|--cann_path)
      cann_path="$2"
      shift 2
      ;;
    -t|--test)
      TEST="all"
      check_param_test_pkg
      shift
      ;;
    --adv_test)
      TEST_PART="adv_test"
      check_param_test_part
      shift
      ;;
    --basic_test_one)
      TEST_PART="basic_test_one"
      check_param_test_part
      shift
      ;;
    --basic_test_two)
      TEST_PART="basic_test_two"
      check_param_test_part
      shift
      ;;
    --basic_test_three)
      TEST_PART="basic_test_three"
      check_param_test_part
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
      check_param_test_pkg
      shift
      ;;
    --cann_3rd_lib_path=*)
      CANN_3RD_LIB_PATH="${1#*=}"
      shift
      ;;
    --cann_3rd_lib_path)
      CANN_3RD_LIB_PATH="$2"
      shift 2
      ;;
    --make_clean)
      MAKE_CLEAN="true"
      check_param_clean
      clean
      exit 0
      ;;
    -j=*)
      THREAD_NUM="${1#*=}"
      check_param_j
      shift
      ;;
    -j)
      THREAD_NUM="$2"
      check_param_j
      shift 2
      ;;
    --build-type=*)
      BUILD_TYPE="${1#*=}"
      shift
      ;;
    --build-type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    *)
      log "[ERROR] Undefined option: $1"
      usage
      break
      ;;
    esac
  done
}

set_env() {
  if [ "${USER_ID}" != "0" ]; then
    DEFAULT_TOOLKIT_INSTALL_DIR="${HOME}/Ascend/latest"
    DEFAULT_INSTALL_DIR="${HOME}/Ascend/latest"
  else
    DEFAULT_TOOLKIT_INSTALL_DIR="/usr/local/Ascend/latest"
    DEFAULT_INSTALL_DIR="/usr/local/Ascend/latest"
  fi

  if [ -n "${cann_path}" ];then
    ASCEND_CANN_PACKAGE_PATH=${cann_path}
  elif [ -n "${ASCEND_HOME_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=${ASCEND_HOME_PATH}
  elif [ -n "${ASCEND_OPP_PATH}" ];then
    ASCEND_CANN_PACKAGE_PATH=$(dirname ${ASCEND_OPP_PATH})
  elif [ -d "${DEFAULT_TOOLKIT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_TOOLKIT_INSTALL_DIR}
  elif [ -d "${DEFAULT_INSTALL_DIR}" ];then
    ASCEND_CANN_PACKAGE_PATH=${DEFAULT_INSTALL_DIR}
  else
    log "Error: Please set the cann package installation directory through parameter -p|--cann_path."
    exit 1
  fi

}

function clean()
{
  if [ -n "${BUILD_DIR}" ];then
    rm -rf ${BUILD_DIR}
  fi

  if [ -n "${OUTPUT_DIR}" ];then
    rm -rf ${OUTPUT_DIR}
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
  cmake --build . --target ${target} -j ${THREAD_NUM}
}

function build_package(){
  CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_TEST=OFF"
  if [ -n "${BUILD_TYPE}" ]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  fi
  cmake_config
  build package
  cp ${BUILD_DIR}/_CPack_Packages/makeself_staging/*.run ${OUTPUT_DIR}
}

function build_test() {
  cmake_config
  build all
}

function build_test_part() {
  if [[ "$TEST_PART" == "adv_test" ]]; then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DTEST_MOD=adv"
    build_test
    return 0
  fi

  source ${CURRENT_DIR}/tests/unit/basic_api/common/ci/basic_tests_part.sh

  if [ "$TEST_PART" == "basic_test_one" ]; then
    BASIC_TEST_PART=("${test_one_targets[@]}")
  elif [ "$TEST_PART" == "basic_test_two" ]; then
    BASIC_TEST_PART=("${test_two_targets[@]}")
  elif [ "$TEST_PART" == "basic_test_three" ]; then
    BASIC_TEST_PART=("${test_three_targets[@]}")
  fi

  for tag in "${BASIC_TEST_PART[@]}"; do
    TARGETS="${TARGETS} --target ${tag}"
    TEST_MOD="${tag},${TEST_MOD}"
  done

  CUSTOM_OPTION="${CUSTOM_OPTION} -DTEST_MOD=${TEST_MOD}"
  if [ "${COV}" == "true" ]; then
    TARGETS="${TARGETS} --target collect_coverage_data"
  fi

  cmake_config
  cmake --build . ${TARGETS} -j ${THREAD_NUM}
  return 0
}

main() {
  check_param_with_help "$@"
  set_options "$@"

  set_env

  CUSTOM_OPTION="${CUSTOM_OPTION} -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=${ASCEND_CANN_PACKAGE_PATH}"

  if [ -n "${TEST}" ];then
    CUSTOM_OPTION="${CUSTOM_OPTION} -DENABLE_TEST=ON -DTEST_MOD=all"
  fi

  if [ -n "${TEST_PART}" ]; then
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

  if [[ ! -d "${BUILD_DIR}" ]]; then
    mkdir -p "${BUILD_DIR}"
  fi
  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
  fi

  cd ${BUILD_DIR}

  if [ -n "${TEST}" ]; then
    build_test
  elif [ -n "$TEST_PART" ]; then
    build_test_part
  elif [ -n "${PKG}" ]; then
    build_package
  else
    cmake_config
    build all
  fi
}

set -o pipefail
if [[ $# -eq 0 ]]; then
  usage
fi

main "$@"