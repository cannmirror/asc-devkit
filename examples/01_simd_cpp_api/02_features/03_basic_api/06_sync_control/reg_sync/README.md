# reg_sync样例

## 概述
本样例演示了基于RegBase编程范式下对UB（Unified Buffer）读或写操作的同步指令使用。样例中使用到了寄存器保序特性（优化读写之间的同步指令）和LoalMemBar接口来保序。本样例支持两种计算场景：读写依赖和写写依赖，通过环境变量选择场景。
    <table>
 	  	 	<tr>
 	  	 		<td>scenarioNum</td>
 	  	 		<td>同步场景</td>
 	  	 	</tr>
 	  	 	<tr>
 	  	 		<td>1</td>
 	  	 		<td>读写依赖（寄存器保序）</td>
 	  	 	</tr>
 	  	 	<tr>
 	  	 		<td>2</td>
 	  	 		<td>写写依赖（LocalMemBar）</td>
 	  	 	</tr>
 	  	 </table>

## 支持的产品
- Ascend 950PR/Ascend 950DT

## 目录结构介绍
```
├── reg_sync
│   ├── scripts
│   │   ├── gen_data.py                // 输入数据和真值数据生成脚本
│   ├── CMakeLists.txt                 // 编译工程文件
│   ├── data_utils.h                   // 数据读入写出函数
│   ├── reg_sync.asc                   // AscendC算子实现 & 调用样例
│   └── README.md                      // 样例介绍
```

## 算子描述
- 算子功能：  
  算子输入一个长度为1024的向量A和一个长度为1024的向量B，数据类型为float，进行向量相加或相减操作。
- 算子规格：
  <table>
  <tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="3" align="center">AIV算子</td></tr>
  </tr>
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td></tr>
  <tr><td align="center">x</td><td align="center">1024</td><td align="center">float</td></tr>
  <tr><td align="center">y</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  </tr>
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">1024</td><td align="center">float</td></tr>
  </tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reg_sync</td></tr>
  </table>
- 算子实现：  
  主要包括以下步骤：
  1. 在vf函数中每次迭代读取向量A到reg0，读取向量B的数据到reg1，进行相加后结果写到reg0，随后将reg0的数据保存到dstAddr，写指令与Add指令都用到了寄存器reg0，读指令与Add指令都用到了寄存器reg1，触发了寄存器保序机制，可以优化掉读写之间的同步指令。
  2. 在vf函数中每次迭代读取向量A到reg0，读取向量B的数据到reg1，进行相减后结果写到reg3，每次迭代都将reg3的数据保存到dstAddr，由于每次保存的数据地址都相同，所以需要加LocalMemBar指令来保序。该样例场景2只为演示LocalMemBar指令的使用。
  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行
在本样例根目录下执行如下步骤，编译并执行算子。
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
  SCENARIO=1
  mkdir -p build && cd build;                                                   # 创建并进入build目录
  cmake -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j; # 编译工程（默认npu模式）
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO                         # 生成测试输入数据
  ./demo                                                                        # 执行编译生成的可执行程序，执行样例
  ```

  使用 CPU调试 或 NPU仿真 模式时，添加 `-DCMAKE_ASC_RUN_MODE=cpu` 或 `-DCMAKE_ASC_RUN_MODE=sim` 参数即可。

  示例如下：
  ```bash
  SCENARIO=1
  cmake -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j; # cpu调试模式
  cmake -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DSCENARIO_NUM=$SCENARIO ..;make -j; # NPU仿真模式
  ```

  > **注意：** 切换编译模式或场景前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

- 编译选项说明

| 选项　　　　　 | 可选值　　　　　　　　　　　| 说明　　　　　　　　　　　　　　　　　　　　　　　|
| ----------------| -----------------------------| ---------------------------------------------------|
| `CMAKE_ASC_RUN_MODE` | `npu`（默认）、`cpu`、`sim` | 运行模式：NPU 运行、CPU调试、NPU仿真　　　　　　　|
| `CMAKE_ASC_ARCHITECTURES` | `dav-3510` | NPU 架构：dav-3510 对应 Ascend 950PR/Ascend 950DT |
| `SCENARIO_NUM` | `1`、`2`　　　　　　　　　　| 场景编号：1=读写依赖，2=写写依赖　　　　　　　　　|

- 执行结果  
  执行结果如下，说明精度对比成功。
  ```bash
  test pass!
  ```
