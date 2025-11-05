# 样例介绍
| 样例名                                                         | 描述                                                         | 算子调用方式|
| ------------------------------------------------------------ | ------------------------------------------------------------- | ------------------ |
| [00_basic_matmul](./00_basic_matmul) | 基于Layout实现的基础K轴遍历Matmul算子。                                                     |<<<>>>直调 |
| [01_misplace_core_matmul](./01_misplace_core_matmul) | 基于Layout实现的使用错位分核策略Matmul算子。                                 |<<<>>>直调 |
| [02_batch_matmul](./02_batch_matmul) | 基于Layout实现的使用错位分核策略的BatchMatmul算子。                                          |<<<>>>直调 |
| [03_quant_matmul](./03_quant_matmul) | 基于Layout实现的基础K轴遍历Matmul算子，并将int8_t的结果量化为float16类型。               |<<<>>>直调|
| [04_l2_misplace_core_matmul](./04_l2_misplace_core_matmul) | 基于Layout实现的带L2切分的使用错位分核策略的Matmul算子。                 |<<<>>>直调 |
| [05_l2_misplace_core_batchmatmul](./05_l2_misplace_core_batchmatmul) | 基于Layout实现的带L2切分的使用错位分核策略的BatchMatmul算子。 |<<<>>>直调 |
| [06_l2_misplace_core_quant_matmul](./06_l2_misplace_core_quant_matmul) | 基于Layout实现的带L2切分的使用错位分核策略的Matmul算子，并将int8_t的结果量化为float16类型。              |<<<>>>直调 |
| [07_naive_matmul](./07_naive_matmul) | 简单for循环的基础K轴遍历Matmul算子。                                                        |<<<>>>直调 |
| [08_sparse_matmul](./08_sparse_matmul) |针对4:2稀疏矩阵专用的基础K轴遍历Matmul算子。                                                |<<<>>>直调 |
