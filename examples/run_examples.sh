SOC_VERSION=${1:-"910B"}
ASCEND_INSTALL_PATH=/usr/local/Ascend/latest
EXAMPLES_DIR="$(PWD)"
export CMAKE_PREFIX_PATH="${ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake"
source $ASCEND_INSTALL_PATH/bin/setenv.bash

function run_case(){
    if [[ -f $1/README.md ]];then
        if grep -q "${SOC_VERSION}" $1/README.md;then
            sed -n '/mkdir -p build/,/```/p' $1/README.md | sed '$d' > $1/auto_run.sh
            echo "============START RUN $1 SOC_VERSION:${SOC_VERSION}============"
            cd $1;bash auto_run.sh;cd -
        else
            echo "============$1 don't support ${SOC_VERSION}============"
        fi
    fi
}

find $EXAMPLES_DIR -mindepth 1 -maxdepth 1 -type d -name "*_*" | while read dir;do
    cd $dir
    if find . -maxdepth 1  -type d -name "*_*" | grep -q './';then
        find . -maxdepth 1  -type d -name "*_*" | while read subdir;do
            run_case $subdir
        done
    else
        run_case $dir
    fi
    cd - > /dev/null
done