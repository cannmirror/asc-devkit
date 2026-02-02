
# DataCopy数据切片算子样例

## 概述
本样例通过Ascend C编程语言实现了DataCopy数据切片算子，支持数据的切片搬运，提取多维Tensor数据的子集进行搬运。使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现。
## 支持的产品
- Ascend 950PR/Ascend 950DT
- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品


## 目录结构
```
├── slice_data_movement
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── slice.asc              // Ascend C算子实现 & 调用样例
```

## 算子描述
- 算子功能：  
  实现了slice_data_movement算子，支持数据的切片搬运，提取多维Tensor数据的子集进行搬运。

- 算子规格：  
 
  <table>  
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="5" align="center">Slice</td></tr>
  </tr>
  <tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">切片参数</td></tr>  
  <tr><td align="center">x</td><td align="center">[3, 87]</td><td align="center">float32</td><td align="center">ND</td><td align="center">[[0, 16:40], [0, 47:71]], [[2, 16:40][2, 47:71]]</tr>  
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">[2, 48]</td><td align="center">float32</td><td align="center">ND</td><td align="center">\</td></tr>  
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">kernel_slice</td></tr>  
  </table>

- 算子实现：  
  - Kernel实现   
    计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先按切片参数搬运进片上存储，得到最终结果，再搬出到外部存储上。   

    slice_data_movement算子的实现流程分为3个基本任务：CopyIn，CopyOut。CopyIn任务负责将Global Memory上的输入xGm，按切片参数搬运至Local Memory。CopyOut任务负责将输出数据从outLocal搬运至Global Memory上的输出Tensor outGm中。

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