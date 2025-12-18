# Broatcast算子直调样例
## 概述
本样例基于Broadcast算子工程，介绍了单算子直调。样例支持将输入张量按目标形状进行广播，通过直接<<<>>>调用核函数，免去框架调度开销，实现高效、低延时的张量扩展运算。
## 支持的AI处理器
- Ascend 910C
- Ascend 910B
## 目录结构介绍
```
├── 05_broadcast  
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── broadcast_custom.asc    // AscendC算子实现 & 调用样例
```
## 算子描述
- 算子功能：  
  Broadcast算子实现了将输入数据按照输出shape进行广播的功能，比如A的shape为(4096,1)，广播的目标shape为(4096,3)，则会将原来的一列扩展为相同的3列, A的shape变为(4096,3)。

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Broadcast</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape_range</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">4096 * 1</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">4096 * 3</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">broadcast_custom</td></tr>
  </table>

## 编译运行
- 配置环境变量  
  以命令行方式下载样例代码，master分支为例。
  ```bash
  cd ${git_clone_path}/examples/00_introduction/05_broadcast/
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
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass
  ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/06 | 样例目录调整，新增本readme |
