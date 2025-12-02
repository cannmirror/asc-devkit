# HelloWorld算子直调样例
## 概述
本样例通过使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，核函数内通过printf打印输出结果。

## 支持的AI处理器
- Ascend 910C
- Ascend 910B
## 目录结构介绍
```
├── 00_helloworld
│   ├── CMakeLists.txt      // 编译工程文件
│   └── hello_world.asc     // Ascend C算子实现 & 调用样例
```

## 编译运行
在本样例根目录下执行如下步骤，编译并执行算子。
- 配置环境变量  
  以命令行方式下载样例代码，master分支为例。
  ```bash
  cd ${git_clone_path}/examples/00_introduction/00_helloworld
  ```

  请根据当前环境上CANN开发套件包的[安装方式](../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=/usr/local/Ascend/latest
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=$HOME/Ascend/latest
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    export ASCEND_INSTALL_PATH=${install_path}/latest
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
  ./demo                        # 执行样例
  ```
  执行结果如下，说明执行成功。
  ```bash
  Hello World!!!
  ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/06 | 样例目录调整，新增本readme |