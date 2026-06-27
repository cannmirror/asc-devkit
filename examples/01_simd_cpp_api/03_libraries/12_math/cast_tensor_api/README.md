# cast_tensor_api 样例

## 概述

本样例基于 **Tensor API** 的Transform<Inst::Cast, Trait>接口实现数据类型转换。

**主要改动：**

| 维度 |  Tensor API |
|---|---|
| 数据形状 | 16x16 Layout 矩阵  |
| 搬运入 | MakeCopy(CopyGM2UB{}) -> Copy(atom, xUb, xGm) |
| Cast 计算 | Transform<Inst::Cast, Trait>(yUbTensor, xUbTensor) |
| 搬运出 | MakeCopy(CopyUB2GM{}) -> Copy(atom, yGm, yUb) |
| 内存分配 | 计算offset + MakeMemPtr<Location::UB, T>(offset) |

**支持六种场景：**

| SCENARIO_NUM | 转换 | RoundMode | SatMode | IndexPos |
|---|---|---|---|---|
| 1 | half -> int32_t | RD (floor) | NoSat | - |
| 2 | float -> int16_t | RN (round) | Sat | - |
| 3 | int8_t -> int32_t | RD (floor) | NoSat | PartP0 |
| 4 | int32_t -> uint8_t | RD (floor) | Sat | PartP0 |
| 5 | bfloat16_t -> float | - | - | Even |
| 6 | float -> bfloat16_t | RN (round) | NoSat | Even |

## 编译运行

- 样例执行

  在本样例目录下执行如下命令。

  ```bash
  SCENARIO_NUM=1                                                                 # 执行场景1
  mkdir -p build && cd build;                                                    # 创建并进入build目录
  cmake -DSCENARIO_NUM=$SCENARIO_NUM -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j;  # 编译工程（默认npu模式）
  python3 ../scripts/gen_data.py -scenarioNum=$SCENARIO_NUM                      # 生成测试输入数据
  ./demo                                                                         # 执行编译生成的可执行程序，执行样例
  ```

  使用 CPU调试 或 NPU仿真 模式时，添加 `-DCMAKE_ASC_RUN_MODE=cpu` 或 `-DCMAKE_ASC_RUN_MODE=sim` 参数即可。

  示例如下：

  ```bash
  cmake -DSCENARIO_NUM=1 -DCMAKE_ASC_RUN_MODE=cpu -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # cpu调试模式
  cmake -DSCENARIO_NUM=4 -DCMAKE_ASC_RUN_MODE=sim -DCMAKE_ASC_ARCHITECTURES=dav-3510 ..;make -j; # NPU仿真模式
  ```

  > **注意：** 切换编译模式前需清理 cmake 缓存，可在 build 目录下执行 `rm CMakeCache.txt` 后重新 cmake。

执行结果：

```text
test pass!
```
