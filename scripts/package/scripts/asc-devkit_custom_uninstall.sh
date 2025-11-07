#!/bin/sh
# Perform custom uninstall script for asc-devkit package
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

curpath=$(dirname $(readlink -f "$0"))

common_parse_dir=""
logfile=""
stage=""
is_quiet="n"
hetero_arch="n"

while true; do
    case "$1" in
    --common-parse-dir=*)
        common_parse_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        pkg_version_dir=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --logfile=*)
        logfile=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --stage=*)
        stage=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --quiet=*)
        is_quiet=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --hetero-arch=*)
        hetero_arch=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    -*)
        shift
        ;;
    *)
        break
        ;;
    esac
done

# 写日志
log() {
    local cur_date="$(date +'%Y-%m-%d %H:%M:%S')"
    local log_type="$1"
    local log_msg="$2"
    local log_format="[AscDevkit] [$cur_date] [$log_type]: $log_msg"
    if [ "$log_type" = "INFO" ]; then
        echo "$log_format"
    elif [ "$log_type" = "WARNING" ]; then
        echo "$log_format"
    elif [ "$log_type" = "ERROR" ]; then
        echo "$log_format"
    elif [ "$log_type" = "DEBUG" ]; then
        echo "$log_format" 1> /dev/null
    fi
    echo "$log_format" >> "$logfile"
}

get_arch_name() {
    local pkg_dir="$1"
    local scene_file="$pkg_dir/scene.info"
    grep '^arch=' $scene_file | cut -d"=" -f2
}

remove_stub_softlink() {
    local ref_dir="$1"
    if [ ! -d "$ref_dir" ]; then
        return
    fi
    local stub_dir="$2"
    if [ ! -d "$stub_dir" ]; then
        return
    fi
    local pwdbak="$(pwd)"
    cd $stub_dir && chmod u+w . && ls -1 "$ref_dir" | xargs --no-run-if-empty rm -rf
    [ -L "x86_64" ] && rm -rf "x86_64"
    [ -L "aarch64" ] && rm -rf "aarch64"
    cd $pwdbak
}

custom_uninstall() {
    if [ -z "$common_parse_dir/asc-devkit" ]; then
        log "ERROR" "ERR_NO:0x0001;ERR_DES:asc-devkit directory is empty"
        exit 1
    elif [ "$hetero_arch" != "y" ]; then
        local arch_name="$(get_arch_name $common_parse_dir/asc-devkit)"
        local ref_dir="$common_parse_dir/asc-devkit/lib64/stub/linux/$arch_name"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/asc-devkit/lib64/stub"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/$arch_name-linux/devlib"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/$arch_name-linux/lib64/stub"
    else
        local arch_name="$(get_arch_name $common_parse_dir/asc-devkit)"
        local ref_dir="$common_parse_dir/asc-devkit/lib64/stub/linux/$arch_name"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/asc-devkit/lib64/stub"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/../devlib"
        remove_stub_softlink "$ref_dir" "$common_parse_dir/../lib64/stub"
    fi
    return 0
}

custom_uninstall
if [ $? -ne 0 ]; then
    exit 1
fi
exit 0
