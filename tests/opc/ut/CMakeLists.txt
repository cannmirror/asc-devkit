cmake_minimum_required(VERSION 3.16.0)
project(opc_ut)
message(STATUS "llt/atc/opcompiler/opc/ut compile start")

set(OPC_BASE "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "OPC_BASE: ${OPC_BASE}")

file(COPY ${TOP_DIR}/asl/ops/cann/auto_schedule/python/tbe/common/context/op_info.py DESTINATION ${TOP_DIR}/llt/atc/opcompiler/opc/stub/tbe/common/context/)

set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../../metadef)
# 生成 ge_ir.pb.cc 和 ge_ir.pb.h 文件
add_custom_command(
  OUTPUT ${OPC_BASE}/../stub/cpp/ge_ir.pb.cc ${OPC_BASE}/../stub/cpp/ge_ir.pb.h
  COMMAND ${PROTOC_PROGRAM} -I=${METADEF_DIR}/proto/ --cpp_out=${OPC_BASE}/../stub/cpp ${METADEF_DIR}/proto/ge_ir.proto
  DEPENDS ${METADEF_DIR}/proto/ge_ir.proto
)

# 编译 ge_ir_parser.cc 并生成共享库
add_library(ge_ir_parser SHARED ${OPC_BASE}/../stub/cpp/ge_ir_parser.cc ${OPC_BASE}/../stub/cpp/ge_ir.pb.cc)
target_include_directories(ge_ir_parser PRIVATE ${OPC_BASE}/../stub/cpp)
target_compile_definitions(ge_ir_parser PRIVATE google=ascend_private _GLIBCXX_USE_CXX11_ABI=0)
target_compile_options(ge_ir_parser PRIVATE
  -fno-common
  -fstack-protector-all
  -fvisibility-inlines-hidden
  -Wextra
  -Wall
)
target_link_libraries(ge_ir_parser PRIVATE ascend_protobuf_static)

install(TARGETS ge_ir_parser OPTIONAL
    LIBRARY DESTINATION ${TOP_DIR}/atc/opcompiler/opc/python/opc_tool/
)

# 确保 libge_ir_parser.so 生成后再进行复制
add_custom_command(
  TARGET ge_ir_parser POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E echo "Source file: $<TARGET_FILE:ge_ir_parser>"
  COMMAND ${CMAKE_COMMAND} -E echo "Destination file: ${TOP_DIR}/atc/opcompiler/opc/python/opc_tool/libge_ir_parser.so"
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:ge_ir_parser> ${TOP_DIR}/atc/opcompiler/opc/python/opc_tool/libge_ir_parser.so
  COMMAND ${CMAKE_COMMAND} -E echo "Copied ge_ir_parser.so to ${TOP_DIR}/atc/opcompiler/opc/python/opc_tool/"
)

file(
  COPY
    ${TOP_DIR}/llt/atc/opcompiler/opc/stub/files/ge_ir_pb2.py
    ${TOP_DIR}/llt/atc/opcompiler/opc/stub/te_fusion/reduce_classify_fusion.py
    ${TOP_DIR}/llt/atc/opcompiler/opc/stub/te_fusion/cube_classify_fusion.py
    ${TOP_DIR}/llt/atc/opcompiler/opc/stub/te_fusion/common_classify_fusion.py
    ${TOP_DIR}/llt/atc/opcompiler/opc/stub/te_fusion/norm_classify_fusion.py
  DESTINATION ${TOP_DIR}/atc/opcompiler/opc/python/opc_tool/
)

run_python_llt_test(
        TARGET opc_unittest
        DEPENDS ge_ir_parser
        SRC_FILES_DIR ${TOP_DIR}/atc/opcompiler/opc/python
        TEST_FILES_DIR ${TOP_DIR}/llt/atc/opcompiler/opc/ut/testcase
        EXPORT_PYTHONPATH  ${TOP_DIR}/atc/opcompiler/opc
        ENV_FILE ${TOP_DIR}/llt/atc/opcompiler/opc/common/ci/set_python_env.sh
        ENV_PARAMS ${TOP_DIR}\ py_ut\ 310
        COVERAGERC_DIR
)
