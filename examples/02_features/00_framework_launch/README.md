# FrameworkLaunch样例介绍

## 概述

基于Ascend C自定义算子工程，介绍了Ascend C算子实现、编译部署、单算子调用等一系列开发流程。

## 算子开发样例

| 目录名称                                                     | 功能描述                                                                                                              |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| [aclnn_invocation](./aclnn_invocation)                       | 本样例基于示例自定义算子工程，介绍了aclnn`OpType`单算子API的方式执行固定shape算子。                                   |
| [aclop_invocation](./aclop_invocation)                       | 本样例基于示例自定义算子工程，介绍了`aclopExecuteV2`单算子模型的方式执行固定shape算子。                               |
| [custom_op](./custom_op)                                     | 本样例以简单的自定义算子为示例，展示了其编译、打包成自定义算子包，并部署到CANN环境中的流程。                          |
| [leaky_relu_onnx_invocation](./leaky_relu_onnx_invocation)   | 本样例介绍通过onnx网络调用的方式调用LeakyReluCustom算子                                                               |
| [static_aclnn_invocation](./static_aclnn_invocation)         | 本样例AddCustom算子为例，展示如何编译、打包并链接自定义算子静态库，通过aclnn的方式执行算子。                          |
| [tensorflow_builtin](./tensorflow_builtin)                   | 本样例展示了如何使用Ascend C自定义算子AddCustom映射到TensorFlow内置算子Add，并通过TensorFlow调用Ascend C算子          |
| [tensorflow_custom](./tensorflow_custom)                     | 本样例展示了如何使用Ascend C自定义算子AddCustom映射到TensorFlow自定义算子AddCustom，并通过TensorFlow调用Ascend C算子  |
| [tiling_sink_programming](./tiling_sink_programming)         | 本样例基于示例自定义算子工程，介绍了PyTorch图模式下调用自定义算子，并通过使能Tiling下沉到device侧执行，优化调度性能。 |
| [tiling_template_programming](./tiling_template_programming) | 本样例基于示例自定义算子工程，使用Tiling模板编程进行单算子API方式的算子执行，以有效减少多TilingKey的复杂度。          |