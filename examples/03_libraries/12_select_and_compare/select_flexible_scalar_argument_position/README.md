# SelectFlexibleScalarArgumentPosition算子直调样例
## 概述
本样例基于Select实现对于给定的两个源操作数src0和scalar标量，根据selMask（用于选择的Mask掩码）的比特位值选取元素，得到目的操作数dst。选择的规则为：当selMask的比特位是1时，从src0中选取，比特位是0时选取scalar标量。
本样例通过Ascend C编程语言实现了SelectFlexibleScalarArgumentPosition算子，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现。

## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍
```
├── select_flexible_scalar_argument_position
│   ├── scripts
│   │   ├── gen_data.py                                 // 输入数据和真值数据生成脚本
│   │   └── verify_result.py                            // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt                                  // 编译工程文件
│   ├── data_utils.h                                    // 数据读入写出函数
│   └── select_flexible_scalar_argument_position.asc    // Ascend C算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  SelectFlexibleScalarArgumentPosition算子实现了对于给定的两个源操作数src0和scalar标量，根据selMask的比特位值选取元素，并返回结果的功能。

- 算子规格：  
  <table>  
  <tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">SelectFlexibleScalarArgumentPosition</th></tr>  
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>  
  <tr><td align="center">x</td><td align="center">256</td><td align="center">float32</td><td align="center">ND</td><td align="center">\</td></tr>  
  <tr><td align="center">y</td><td align="center">32</td><td align="center">uint8</td><td align="center">ND</td><td align="center">\</td></tr>   
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">256</td><td align="center">float32</td><td align="center">ND</td><td align="center">\</td></tr>  
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">main_sel_demo</td></tr>  
  </table>

- 算子实现：  
  本样例中实现的是固定shape为256的SelectFlexibleScalarArgumentPosition算子。
  - kernel实现   
    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成对于给定的两个源操作数src0和scalar标量，根据selMask的比特位值选取元素的操作，得到最终结果，再搬出到外部存储上。   

    SelectFlexibleScalarArgumentPosition算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm，yGm搬运至Local Memory，分别存储在xLocal，yLocal，Compute任务负责对xLocal，yLocal执行相关操作，计算结果存储在outLocal中，CopyOut任务负责将输出数据从outLocal搬运至Global Memory上的输出Tensor outGm中。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

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
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                        # 执行编译生成的可执行程序，执行样例
  python3 ../scripts/verify_result.py output/output.bin output/golden.bin   # 验证输出结果是否正确，确认算法逻辑正确
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```