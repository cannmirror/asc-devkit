# 随路relu<a name="ZH-CN_TOPIC_0000002538231206"></a>

**特性说明：**

矩阵计算的搬出过程中支持随路ReLU能力，当前支持3种类型的随路ReLU能力。**DataCopyCO12DstParams结构体参数**定义中reluPre可设置为1开启不同的随路ReLU能力。

- 参数配置为0时，即不开启随路ReLU能力；

- 参数配置为1时，即Normal ReLU时无需配置额外寄存器，对输出数据进行相应的激活处理；
