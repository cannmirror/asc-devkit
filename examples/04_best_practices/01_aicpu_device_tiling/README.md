# AI CPU算子样例
## 概述

本样例介绍使用AI CPU算子进行tiling下沉计算，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── 01_aicpu_device_tiling
│   ├── CMakeLists.txt                     // 编译工程文件
│   ├── aicore_kernel.asc                  // AI Core算子实现
│   ├── kernel_args.h                      // tiling结构体头文件
│   ├── main.asc                           // AI CPU算子与AI Core算子调用
│   └── aicpu_tiling.aicpu                 // AI CPU算子实现
```

## 算子描述
- main.asc中内AI CPU算子与AI Core算子均使用内核调用符<<<...>>>进行调用，AI CPU算子将tiling计算的结果传给AI Core算子。
- AI CPU算子与AI Core算子在不同stream上进行launch，样例中分别为stream与stream2，event用于记录stream上已下发的任务。使用aclrtRecordEvent在指定stream中记录event，使用aclrtStreamWaitEvent阻塞指定的stream，直到指定的event完成。

## 编译运行
- 配置环境变量  
  以命令行方式下载样例代码，master分支为例。
  ```bash
  cd ${git_clone_path}/examples/04_best_practices/01_aicpu_device_tiling
  ```
  请根据当前环境上CANN开发套件包的[安装方式](../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=/usr/local/Ascend/cann
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=$HOME/Ascend/cann
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=${install_path}/cann
    ```
  配置安装路径后，执行以下命令统一配置环境变量。
  ```bash
  # 配置CANN环境变量
  source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
  ```
- 样例执行
  ```bash
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  ./demo                        # 执行编译生成的可执行程序，执行样例
  ```
  观察是否出现如下打屏
  ```bash
  Hello World: int mode 2 len 4.
  Hello World: float mode 1 len 2.
  ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/28 | 样例目录调整，新增本readme |