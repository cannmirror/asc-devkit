# Reduce算子直调样例
## 概述
本样例介绍了调用Sum高阶API实现reduce算子，算子对输入张量沿最后一个维度进行求和，采用<<<>>>直调的方式，规避框架调度开销，实现高效的归约计算。
## 支持的AI处理器
- Ascend 910C
- Ascend 910B
## 目录结构介绍
```
├── 06_reduce    
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   └── reduce_custom.asc       // AscendC算子实现 & 调用样例
```
## 算子描述
- 算子功能：  
  reduce算子，获取输入数据最后一个维度的元素总和。如果输入是向量，则对向量中各元素进行累加；如果输入是矩阵，则沿最后一个维度对每行中元素求和。
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">reduce</td></tr>

  <tr><td rowspan="3" align="center">算子输入</td></tr>
  <tr><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
  <tr><td align="center">x</td><td align="center">7*2023</td><td align="center">float</td><td align="center">ND</td></tr>
  <tr><td rowspan="2" align="center">算子输出</td></tr>
  <tr><td align="center">y</td><td align="center">8*1</td><td align="center">float</td><td align="center">ND</td></tr>

  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">sum_custom</td></tr>
  </table>
- 算子实现：  
  本样例中实现的是固定shape为输入x[7, 2023]，输出y[8]的reduce算子，其中y中的有效值数量为7，对输入x的每行元素求和后，输出y的有效数据为前7位，最后一位为padding填充的数据。

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用Sum高阶API接口完成sum计算，得到最终结果，再搬出到外部存储上。

  reduce算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor srcGm存储在srcLocal中，Compute任务负责对srcLocal执行sum计算，计算结果存储在dstLocal中，CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGm。

  根据输入数据的内轴长度、内轴实际长度、外轴长度确定所需tiling参数，例如输出内轴补齐后长度等。调用GetSumMaxMinTmpSize接口获取Sum接口完成计算所需的临时空间大小。
## 编译运行
- 配置环境变量  
  以命令行方式下载样例代码，master分支为例。
  ```bash
  cd ${git_clone_path}/examples/00_introduction/06_reduce/
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
  python3 ../scripts/gen_data.py   # 生成测试输入数据
  ./demo                       # 执行编译生成的可执行程序，执行样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/11/06 | 样例目录调整，新增本readme |
