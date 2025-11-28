#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
); cd $CURRENT_DIR

# 导出环境变量
DTYPE="float16"
HEIGHT=8
WIDTH=2048

SHORT=v:,h:,w:,
LONG=dtype:,height:,width:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
while :
do
    case "$1" in
        # float16, float, int32
        (-v | --dtype)
            DTYPE="$2"
            shift 2;;
        (-h | --height)
            HEIGHT="$2"
            shift 2;;
        (-w | --width)
            WIDTH="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/latest
    fi
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export ASCEND_HOME_PATH=$_ASCEND_INSTALL_PATH

PYTHON_VERSION=`python3 -V 2>&1 | awk '{print $2}' | awk -F '.' '{print $1"."$2}'`
export HI_PYTHON=python${PYTHON_VERSION}
export PYTHONPATH=$ASCEND_INSTALL_PATH/python/site-packages:$PYTHONPATH
export PATH=$ASCEND_INSTALL_PATH/python/site-packages/bin:$PATH

# 检查当前昇腾芯片的类型
function check_soc_version() {
    SOC_VERSION_CONCAT=`(export ASCEND_SLOG_PRINT_TO_STDOUT=0 && python3 -c '''
import ctypes, os
def get_soc_version():
    max_len = 256
    rtsdll = ctypes.CDLL(f"libruntime.so")
    c_char_t = ctypes.create_string_buffer(b"\xff" * max_len, max_len)
    rtsdll.rtGetSocVersion.restype = ctypes.c_uint64
    rt_error = rtsdll.rtGetSocVersion(c_char_t, ctypes.c_uint32(max_len))
    if rt_error:
        print("rt_error:", rt_error)
        return ""
    soc_full_name = c_char_t.value.decode("utf-8")
    find_str = "Short_SoC_version="
    ASCEND_INSTALL_PATH = os.environ.get("ASCEND_INSTALL_PATH")
    with open(f"{ASCEND_INSTALL_PATH}/compiler/data/platform_config/{soc_full_name}.ini", "r") as f:
        for line in f:
            if find_str in line:
                start_index = line.find(find_str)
                result = line[start_index + len(find_str):].strip()
                return "{},{}".format(soc_full_name, result.lower())
    return ""
print(get_soc_version())
    ''')`
    if [[ ${SOC_VERSION_CONCAT}"x" = "x" ]]; then
        echo "ERROR: SOC_VERSION_CONCAT is invalid!"
        return 1
    fi
    SOC_FULL_VERSION=`echo $SOC_VERSION_CONCAT | cut -d ',' -f 1`
    SOC_SHORT_VERSION=`echo $SOC_VERSION_CONCAT | cut -d ',' -f 2`
}

function main() {
    # 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm -rf $ASCEND_INSTALL_PATH/opp/vendors/*
    rm -rf op_verify/run_out/op_models/*.om

    # 增加自定义算子工程样例
    JSON_NAME=add_template_custom
    rm -rf custom_op/op_host/*.cpp custom_op/op_host/*.h
    rm -rf custom_op/op_kernel/*.cpp custom_op/op_kernel/*.h
    cp -rf op_dev/op_host/*.cpp op_dev/op_host/*.h custom_op/op_host
    cp -rf op_dev/op_kernel/*.cpp op_dev/op_kernel/*.h custom_op/op_kernel

    sed -i "s#/usr/local/Ascend/latest#$ASCEND_INSTALL_PATH#g" `grep "/usr/local/Ascend/latest" -rl custom_op/CMakePresets.json`

    # 测试不同输入数据类型, 修改对应代码
    if [[ ${DTYPE} == "float16" ]]; then
        sed -i "s/.astype(.*)/.astype(np.float16)/g" `grep ".astype(.*)" -rl op_verify/scripts/generate_data.py`
        sed -i "s/aclDataType dataType =.*;/aclDataType dataType = ACL_FLOAT16;/g" `grep "aclDataType dataType =.*;" -rl op_verify/src/main.cpp`
        sed -i "s/dtype=.*)/dtype=np.float16)/g" `grep "dtype=.*)" -rl op_verify/scripts/verify_result.py`
    elif [[ ${DTYPE} == "float" ]]; then
        sed -i "s/.astype(.*)/.astype(np.float32)/g" `grep ".astype(.*)" -rl op_verify/scripts/generate_data.py`
        sed -i "s/aclDataType dataType =.*;/aclDataType dataType = ACL_FLOAT;/g" `grep "aclDataType dataType =.*;" -rl op_verify/src/main.cpp`
        sed -i "s/dtype=.*)/dtype=np.float32)/g" `grep "dtype=.*)" -rl op_verify/scripts/verify_result.py`
    else
        echo "ERROR: DTYPE is invalid!"
        return 1
    fi

    # 构建自定义算子包并安装
    bash custom_op/run.sh
    if [ $? -ne 0 ]; then
        echo "ERROR: build and install custom op run package failed!"
        return 1
    fi
    echo "INFO: build and install custom op run package success!"

    # 编译离线om模型
    source $ASCEND_INSTALL_PATH/vendors/customize/bin/set_env.bash
    export ASCEND_CUSTOM_OPP_PATH=$(realpath $(dirname $0))/custom_op/build_out/lib/
    atc --singleop=op_verify/scripts/${JSON_NAME}.json --output=op_verify/run_out/op_models/ --soc_version=${SOC_FULL_VERSION} --op_debug_level=0

    # 编译acl可执行文件并运行
    bash op_verify/run.sh $HEIGHT $WIDTH

    # 清除自定义算子工程样例
    rm -rf custom_op/op_host/*.cpp custom_op/op_host/*.h
    rm -rf custom_op/op_kernel/*.cpp custom_op/op_kernel/*.h
}

check_soc_version
main
