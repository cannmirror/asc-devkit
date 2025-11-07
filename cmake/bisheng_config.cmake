if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/ccec_compiler)
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/ccec_compiler)
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/compiler)
else()
    set(ASCENDC_DEVKIT_PATH ${ASCEND_CANN_PACKAGE_PATH}/ascendc_devkit)
endif()


set(CCEC_PATH           ${ASCENDC_DEVKIT_PATH}/ccec_compiler/bin)
set(CMAKE_C_COMPILER    "${CCEC_PATH}/bisheng")
set(CMAKE_CXX_COMPILER  "${CCEC_PATH}/bisheng")
set(CMAKE_LINKER        "${CCEC_PATH}/ld.lld")
set(CMAKE_AR            "${CCEC_PATH}/llvm-ar")
set(CMAKE_STRIP         "${CCEC_PATH}/llvm-strip")
set(CMAKE_OBJCOPY       "${CCEC_PATH}/llvm-objcopy")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_SKIP_RPATH TRUE)
