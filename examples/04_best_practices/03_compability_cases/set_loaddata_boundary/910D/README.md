# SetLoadDataBoundary兼容性样例
## 概述
兼容实现L1 Buffer设置边界值。
## 支持的AI处理器
- Ascend 910D
## 目录结构介绍
```
├── set_loaddata_boundary
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   └── demo.asc    // AscendC算子实现 & 调用样例
```
## 算子描述
- 算子功能：  
    兼容实现L1 Buffer设置边界值的端到端用例
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">32 * 32</td><td align="center">half</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">32 * 32</td><td align="center">half</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">32 * 32</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">demo</td></tr>
  </table>
- 算子实现：  
    351x架构硬件删除了L1 Buffer的边界值设定相关寄存器，不再支持SetLoadDataBoundary接口。该接口用于设置Load3D时L1 Buffer的边界值。如果指令在处理源操作数时，源操作数在L1 Buffer上的地址超出设置的边界，则会从L1 Buffer的起始地址开始夺取。设置为0表示无边界，可使用整个L1 Buffer。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。
## 编译运行
  - 配置环境变量  
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/examples/04_best_practices/02_compability_cases/set_loaddata_boundary/910D
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
    mkdir -p build && cd build;                                              # 创建并进入build目录
    cmake ..;make -j;                                                        # 编译工程
    cd ..                                                                    # 回退到主目录
    python3 ./scripts/gen_data.py                                            # 生成测试输入数据
    ./build/demo                                                             # 执行编译生成的可执行程序，执行样例
    python3 ./scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
    ```
    执行结果如下，说明精度对比成功。
    ```bash
    test pass
    ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/12/01 | 新增本readme |