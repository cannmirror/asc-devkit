project (asc_opc)

#build opc whl pkg
message(STATUS "CMAKE_BINARY_DIR=" ${CMAKE_BINARY_DIR})
message(STATUS "CMAKE_CURRENT_SOURCE_DIR=" ${CMAKE_CURRENT_SOURCE_DIR})

set(OPC_WHL_NAME "asc_opc_tool-0.1.0-py3-none-any.whl")
set(OPC_DIR "./opc_python")
set(OPC_BASE "${CMAKE_CURRENT_SOURCE_DIR}")
set(METADEF_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../../metadef)
set(TE_FUSION_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../../atc/opcompiler/te_fusion)

add_custom_target(${OPC_WHL_NAME}
  COMMAND echo "[OPC] Build target ${OPC_WHL_NAME}" &&
    mkdir -p ${OPC_DIR}/asc_opc_tool && rm -rf ${OPC_DIR}/asc_opc_tool/* &&
    ${PROTOC_PROGRAM} -I=${METADEF_DIR}/proto/ --python_out=${OPC_BASE}/python/asc_opc_tool ${METADEF_DIR}/proto/ge_ir.proto &&
    cp ${TE_FUSION_DIR}/python/*_classify_fusion.py ${OPC_BASE}/python/asc_opc_tool &&
    cp -rf ${OPC_BASE}/python/asc_opc_tool/* ${OPC_DIR}/asc_opc_tool/ &&
    cp -rf ${OPC_BASE}/setup.py ${OPC_DIR}/ &&
    cd ${OPC_DIR} && ${HI_PYTHON} setup.py bdist_wheel && cd - &&
    ls && pwd &&
    ${CMAKE_COMMAND} -E rename  ${OPC_DIR}/dist/asc_opc*.whl ${CMAKE_INSTALL_PREFIX}/lib/${OPC_WHL_NAME} &&
    echo "[OPC] Build target ${OPC_WHL_NAME} end"
)
