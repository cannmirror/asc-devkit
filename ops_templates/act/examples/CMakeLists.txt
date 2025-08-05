# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

cmake_minimum_required(VERSION 3.15)
project(act_examples)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(FATAL_ERROR "Cannot find ASCEND_HOME_PATH, please run set_env.sh or export ASCEND_HOME_PATH=/usr/local/Ascend/latest")
else()
  set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
endif()

set(CMAKE_CCEC_COMPILER ccec)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CCEC_COMPILER_OPTIONS
    -O2 -std=c++17 -xcce --cce-aicore-arch=dav-c220
    --cce-aicore-block-local-init
    --cce-auto-sync
    -mllvm -cce-aicore-stack-size=0x8000
    -mllvm -cce-aicore-function-stack-size=0x8000
    -mllvm -cce-aicore-record-overflow=true
    -mllvm -cce-aicore-addr-transform
    -mllvm -cce-aicore-dcci-insert-for-scalar=false
    -DL2_CACHE_HINT

    -I${ASCEND_HOME_PATH}/compiler/tikcpp
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
    -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
    -I${ASCEND_HOME_PATH}/include
    -I${ASCEND_HOME_PATH}/include/experiment
    -I${ASCEND_HOME_PATH}/include/experiment/runtime
    -I${ASCEND_HOME_PATH}/include/experiment/msprof

    -I${ASCEND_HOME_PATH}/x86_64-linux/ascendc/include/highlevel_api
    -I${ASCEND_HOME_PATH}/x86_64-linux/ascendc/include/basic_api

    -I${ASCEND_HOME_PATH}/x86_64-linux/ascendc/include/highlevel_api/impl
    -I${ASCEND_HOME_PATH}/x86_64-linux/ascendc/include/basic_api/impl
    -I${ASCEND_HOME_PATH}/x86_64-linux/ascendc/include/basic_api/interface

    -I${CMAKE_SOURCE_DIR}
    -L${ASCEND_HOME_PATH}/lib64
    -Wno-macro-redefined -Wno-ignored-attributes
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
)

message("ASCEND_HOME_PATH: ${ASCEND_HOME_PATH}")
message("CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message("ARGN: ${ARGN}")
message("CCEC_COMPILER_OPTIONS: ${CCEC_COMPILER_OPTIONS}")

file(GLOB_RECURSE ACT_INCLUDE_FILES ${CMAKE_SOURCE_DIR}/include/*.h)

message("ACT_INCLUDE_FILES: ${ACT_INCLUDE_FILES}")

add_custom_target(act_examples)

function(act_examples_add_executable NAME)
    add_custom_command(
        OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        COMMAND ${CMAKE_CCEC_COMPILER}  ${CCEC_COMPILER_OPTIONS} ${ARGN} -o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME}
        DEPENDS ${ARGN} ${ACT_INCLUDE_FILES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Compiling executable kernel: ${NAME}"
    )

    add_custom_target(${NAME} ALL DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${NAME})
    add_dependencies(act_examples ${NAME})

endfunction()

set(EXAMPLE
    00_basic_matmul
    01_misplace_core_matmul
    02_batch_matmul
    03_quant_matmul
    04_l2_misplace_core_matmul
    05_l2_misplace_core_batchmatmul
    06_l2_misplace_core_quant_matmul
    07_naive_matmul
)

foreach(EXAMPLE_ITEM ${EXAMPLE})
    add_subdirectory(${EXAMPLE_ITEM})
endforeach()
