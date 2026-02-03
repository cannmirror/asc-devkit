# AI CPU算子直调样例

## 概述

本样例介绍了基于Ascend C的AI CPU算子的核函数直调方法，算子实现简单的HelloWorld打印功能。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── hello_world_aicpu
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── hello_world.aicpu   // AI CPU算子实现
|   └── main.asc            // AI CPU算子调用
```

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/cann/set_env.sh
    ```
    
- 样例执行
  ```bash
  mkdir -p build && cd build;   # 创建并进入build目录
  cmake ..;make -j;             # 编译工程
  ./demo                        # 执行样例
  ```
  执行结果如下，说明执行成功。
  ```bash
  Hello World!!!
  ```