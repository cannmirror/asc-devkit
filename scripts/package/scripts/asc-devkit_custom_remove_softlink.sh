#!/bin/sh
# Perform custom remove softlink script for asc-devkit package
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

while true; do
    case "$1" in
    --install-path=*)
        install_path=$(echo "$1" | cut -d"=" -f2-)
        shift
        ;;
    --version-dir=*)
        version_dir=$(echo "$1" | cut -d"=" -f2)
        shift
        ;;
    --latest-dir=*)
        latest_dir=$(echo "$1" | cut -d"=" -f2)
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

do_remove_stub_softlink() {
    local arch_name="$(get_arch_name $install_path/$version_dir/asc-devkit)"
    local arch_linux_path="$install_path/$latest_dir/$arch_name-linux"
    if [ ! -e "$arch_linux_path" ] || [ -L "$arch_linux_path" ]; then
        return
    fi
    local ref_dir="$install_path/$version_dir/asc-devkit/lib64/stub/linux/$arch_name"
    remove_stub_softlink "$ref_dir" "$arch_linux_path/devlib"
    remove_stub_softlink "$ref_dir" "$arch_linux_path/lib64/stub"
    if [ -d "$install_path/$latest_dir/tools" ]; then
        rm -f "$install_path/$latest_dir/tools"
    fi
}

do_remove_stub_softlink
