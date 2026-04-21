# cast样例

## 概述
本样例基于RegBase编程范式实现Cast运算，用于RegTensor（Reg矢量计算基本单元）的数据类型转换，主要调用Cast接口。

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── cast
│   ├── scripts
│   │   ├── gen_data.py                // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt                 // 编译工程文件
│   ├── data_utils.h                   // 数据读入写出函数
│   ├── cast.asc                       // AscendC样例实现 & 调用样例
│   └── README.md                      // 样例介绍
```

## 样例描述
- 样例功能：  
  本样例对输入向量做数据类型转换。当输入输出的数据类型的位宽不同时，Cast指令会间隔地读取或读取，所以本样例在数据搬入或搬出时会使用相应的压缩或解压缩模式，LoadAlign/StoreAlign接口的使用仅供参考，具体说明如下：

  **场景1：数据类型位宽小转大**
  - 输入：half数据类型，位宽为16bit
  - 输出：int32_t数据类型，位宽为32bit
  - 说明：从half到int32_t的Cast指令每次处理64个数据，具体流程和示意图如下：
    - 搬入：调用LoadAlign接口，使用解压缩模式，搬入数据至2\*N位置，同时2\*N+1位置置0
    - 计算：调用Cast接口，输入输出位宽比为1:2，所以从xReg的2\*N位置读取数据，类型转换后依次写入至yReg
    - 搬出：调用StoreAlign接口，常规搬出  
    <img src="figure/reg_cast_1.png">
  - 样例规格：
    <table>
    <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="3" align="center">AIV样例</td></tr>
    <tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">half</td></tr>
    <tr><td rowspan="1" align="center">样例输出</td><td align="center">y</td><td align="center">[1, 256]</td><td align="center">int32_t</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cast</td></tr>
    </table>
  - 样例实现：  
    CastVF函数内调用Cast接口进行计算，结果写回UB
    - 调用实现  
      使用内核调用符<<<>>>调用核函数。

  **场景2：数据类型位宽大转小**
  - 输入：float数据类型，位宽为32bit
  - 输出：int16_t数据类型，位宽为16bit
  - 说明：从half到int32_t的Cast指令每次处理64个数据，具体流程和示意图如下：
    - 搬入：调用LoadAlign接口，常规搬入
    - 计算：调用Cast接口，输入输出位宽比为2:1，所以从xReg依次读取数据，类型转换后依次写入至yReg的2\*N位置，同时2\*N+1位置置0
    - 搬出：调用StoreAlign接口，采用压缩模式，仅搬出2\*N位置的数据  
    <img src="figure/reg_cast_2.png">
  - 样例规格：
    <table>
    <tr><td rowspan="1" align="center">样例类型(OpType)</td><td colspan="3" align="center">AIV样例</td></tr>
    <tr><td rowspan="2" align="center">样例输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
    <tr><td align="center">x</td><td align="center">[1, 256]</td><td align="center">float</td></tr>
    <tr><td rowspan="1" align="center">样例输出</td><td align="center">y</td><td align="center">[1, 256]</td><td align="center">int16_t</td></tr>
    <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">cast</td></tr>
    </table>
  - 样例实现：  
    CastVFF322S16和CastVFF162S32函数内调用Cast接口进行计算，结果写回UB
    - 调用实现  
      使用内核调用符<<<>>>调用核函数。

## 编译运行
在本样例根目录下执行如下步骤，编译并执行样例。
- 配置环境变量  
  请根据当前环境上CANN开发套件包的[安装方式](../../../../../../docs/quick_start.md#prepare&install)，选择对应配置环境变量的命令。
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
  SCENARIO=1                                                      # 执行场景1
  mkdir -p build && cd build;                                     # 创建并进入build目录
  cmake -DSCENARIO_NUM=$SCENARIO -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;  # 编译工程（默认npu模式）
  python3 ../scripts/gen_data.py $SCENARIO                        # 生成测试输入数据
  ./demo                                                          # 执行编译生成的可执行程序，执行样例
  ```
 
  使用 CPU调试 或 NPU仿真 模式时，添加 `-DCMAKE_ASC_RUN_MODE=cpu` 或 `-DCMAKE_ASC_RUN_MODE=sim` 参数即可。

  示例如下：
  ```bash
  cmake -DSCENARIO_NUM=1 -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # cpu调试模式
  cmake -DSCENARIO_NUM=1 -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU仿真模式
  ```

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明

| 选项 | 可选值 | 说明 |
|------|--------|------|
| `SCENARIO_NUM` | 1、2 | 样例执行场景：场景1：数据类型位宽小转大、场景2：数据类型位宽大转小 |
| `CMAKE_ASC_RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真 |
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU 架构：dav-3510 对应 Ascend 950PR/Ascend 950DT |

- 执行结果  
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```
