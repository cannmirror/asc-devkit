# torch_library注册自定义算子直调样例
## 概述
本样例展示了如何使用PyTorch的torch.library机制注册自定义算子，并通过<<<>>>内核调用符调用核函数，以简单的Add算子为例，实现两个向量的逐元素相加。

## 支持的产品
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── torch_library
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── add_custom_test.py      // PyTorch调用自定义算子的测试脚本
│   └── add_custom.asc          // Ascend C算子实现 & 自定义算子注册
```

## 算子描述
- 算子功能：  
  Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：
  ```
  z = x + y
  ```

- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
  </table>

- 算子实现：

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。

- 自定义算子注册：

  PyTorch提供`TORCH_LIBRARY`宏作为自定义算子注册的核心接口，用于创建并初始化自定义算子库，注册后在Python侧可以通过`torch.ops.namespace.op_name`方式进行调用，例如：
  ```c++
  TORCH_LIBRARY(ascendc_ops, m) {
      m.def(ascendc_add"(Tensor x, Tensor y) -> Tensor");
  }
  ```
  另外，若相同命名空间需要在多个文件中拆分注册，需要使用`TORCH_LIBRARY_FRAGMENT`扩展现有算子库，避免重复创建命名空间导致冲突。

  `TORCH_LIBRARY_IMPL`用于将算子逻辑绑定到特定的DispatchKey（PyTorch设备调度标识）。针对NPU设备，需要将算子实现注册到PrivateUse1这一专属的DispatchKey上，例如：
  ```c++
  TORCH_LIBRARY_IMPL(ascendc_ops, PrivateUse1, m)
  {
      m.impl("ascendc_add", TORCH_FN(ascendc_ops::ascendc_add));
  }
  ```

  本样例在add_custom.asc中定义了一个名为ascendc_ops的命名空间，并在其中注册了ascendc_add函数，Python侧可以通过`torch.ops.ascendc_ops.ascendc_add`调用自定义的API。在ascendc_add函数中通过`c10_npu::getCurrentNPUStream()`函数获取当前NPU上的流，并通过内核调用符<<<>>>调用自定义的Kernel函数add_custom，在NPU上执行算子。

- Python测试脚本

  在add_custom_test.py中，首先通过`torch.ops.load_library`加载生成的自定义算子库，并定义一个仅包含单算子的PyTorch模型SingleOpModel，其前向计算直接调用自定义算子。在测试执行时，脚本通过torchair配置编译策略并利用`torch.compile`使能模型在NPU进行全图编译优化，同时启动`torch_npu.profiler`性能分析器来获取算子执行过程中NPU和CPU的性能数据，最终通过对比NPU输出与CPU标准加法结果来验证自定义算子的数值正确性。

## 编译运行
在本样例根目录下执行如下步骤，编译并执行算子。
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
- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```
  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/ascend-toolkit/set_env.sh
    ```
  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/ascend-toolkit/set_env.sh
    ```
- 样例执行
  ```bash
  mkdir -p build && cd build;     # 创建并进入build目录
  cmake ..;make -j;               # 编译工程
  python3 ../add_custom_test.py   # 执行测试脚本
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  Ran 1 test in **s.
  OK
  ```