# Pybind算子直调样例

## 概述
本样例以Add算子为示例，介绍了使用Pybind方式调用核函数。

## 支持的AI处理器
- Ascend 910C
- Ascend 910B

## 目录结构介绍
```
├── pybind
│   ├── CMakeLists.txt        // 编译工程文件
│   ├── add_custom_test.py    // Python调用脚本
│   └── add_custom.asc        // Ascend C算子实现 & Pybind封装
```

## 算子描述
- 算子功能：

  Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：
  ```
  z = x + y
  ```
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddCustom</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float16</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  </table>
- 算子实现：

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。

- 调用实现：

  通过PyTorch框架进行模型的训练、推理时，会调用很多算子进行计算。使用Pybind可以实现PyTorch框架调用算子Kernel程序，从而实现Ascend C算子在Pytorch框架的集成部署。

  add_custom.asc使用了pybind11库来将C++代码封装成Python模块。该代码实现中定义了一个名为m的pybind11模块，其中包含一个名为run_add_custom的函数。该函数与my_add::run_add_custom函数相同，用于将C++函数转成Python函数。在函数实现中，通过c10_npu::getCurrentNPUStream() 的函数获取当前NPU上的流，通过内核调用符<<<>>>调用自定义的Kernel函数add_custom，在NPU上执行算子。

  在add_custom_test.py调用脚本中，通过导入自定义模块add_custom，调用自定义模块add_custom中的run_add_custom函数，在NPU上执行x和y的加法操作，并将结果保存在变量z中。

## 编译运行
- 安装PyTorch (这里以使用2.1.0版本为例)
  aarch64:
  ```bash
  pip3 install torch==2.1.0
  ```

  x86:
  ```bash
  pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu
  ```

- 安装torch-npu （以Pytorch2.1.0、python3.9、CANN版本8.0.RC1.alpha002为例）
  ```bash
  git clone https://gitee.com/ascend/pytorch.git -b v6.0.rc1.alpha002-pytorch2.1.0
  cd pytorch/
  bash ci/build.sh --python=3.9
  pip3 install dist/*.whl
  ```

  安装pybind11
  ```bash
  pip3 install pybind11
  ```
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
  mkdir -p build; cd build             # 创建并进入build目录
  cmake ..; make -j                    # 编译算子so
  python3 ../add_custom_test.py        # 执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  Ran 1 test in **s.
  OK
  ```