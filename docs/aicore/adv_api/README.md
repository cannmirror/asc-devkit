<!--声明：本文使用[Creative Commons License version 4.0](https://creativecommons.org/licenses/by/4.0/legalcode)许可协议，转载、引用或修改等操作请遵循此许可协议。-->

## 支持的API

本开源仓覆盖的高阶API列表如下，下表列出简单功能说明，具体使用说明请参考[高阶API](https://hiascend.com/document/redirect/CannCommunityAscendCHighLevelApi)。同时我们还配套了其具体实现的算法框图，可配合源码阅读。

<table>
    <td> 类别 </td>
    <td> API </td>
    <td> 描述 </td>
    <tr>
        <th rowspan="31"> 数学库 </th>
        <td> Acos </td>
        <td> 按元素做反余弦函数计算。 </td>
    </tr>
    <tr>
        <td> Acosh </td>
        <td> 按元素做双曲反余弦函数计算。 </td>
    </tr>
    <tr>
        <td> Asin </td>
        <td> 按元素做反正弦函数计算。 </td>
    </tr>
    <tr>
        <td> Asinh </td>
        <td> 按元素做反双曲正弦函数计算。 </td>
    </tr>
    <tr>
        <td> Atan </td>
        <td> 按元素做三角函数反正切运算。 </td>
    </tr>
    <tr>
        <td> Atanh </td>
        <td> 按元素做反双曲正切余弦函数计算。 </td>
    </tr>
    <tr>
        <td> Axpy </td>
        <td> 源操作数中每个元素与标量求积后和目的操作数中的对应元素相加。 </td>
    </tr>
    <tr>
        <td> Ceil </td>
        <td> 获取大于或等于x的最小的整数值，即向正无穷取整操作。 </td>
    </tr>
    <tr>
        <td> ClampMax </td>
        <td> 将srcTensor中大于scalar的数替换为scalar，小于等于scalar的数保持不变，作为dstTensor输出。 </td>
    </tr>
    <tr>
        <td> ClampMin </td>
        <td> 将srcTensor中小于scalar的数替换为scalar，大于等于scalar的数保持不变，作为dstTensor输出。 </td>
    </tr>
    <tr>
        <td> Cos </td>
        <td> 按元素做三角函数余弦运算。 </td>
    </tr>
    <tr>
        <td> Cosh </td>
        <td> 按元素做双曲余弦函数计算。 </td>
    </tr>
    <tr>
        <td> CumSum </td>
        <td> 对数据按行依次累加或按列依次累加。 </td>
    </tr>
    <tr>
        <td> Digamma </td>
        <td> 按元素计算x的gamma函数的对数导数。 </td>
    </tr>
    <tr>
        <td> Erf </td>
        <td> 按元素做误差函数计算，也称为高斯误差函数。 </td>
    </tr>
    <tr>
        <td> Erfc </td>
        <td> 返回输入x的互补误差函数结果，积分区间为x到无穷大。 </td>
    </tr>
    <tr>
        <td> Exp </td>
        <td> 按元素取自然指数。 </td>
    </tr>
    <tr>
        <td> Floor </td>
        <td> 获取小于或等于x的最小的整数值，即向负无穷取整操作。 </td>
    </tr>
    <tr>
        <td> Fmod </td>
        <td> 按元素计算两个浮点数相除后的余数。 </td>
    </tr>
    <tr>
        <td> Frac </td>
        <td> 按元素做取小数计算。 </td>
    </tr>
    <tr>
        <td> Lgamma </td>
        <td> 按元素计算x的gamma函数的绝对值并求自然对数。 </td>
    </tr>
    <tr>
        <td> Log </td>
        <td> 按元素以e、2、10为底做对数运算。 </td>
    </tr>
    <tr>
        <td> Power </td>
        <td> 实现按元素做幂运算功能。 </td>
    </tr>
    <tr>
        <td> Round </td>
        <td> 将输入的元素四舍五入到最接近的整数。 </td>
    </tr>
    <tr>
        <td> Sign </td>
        <td> 按元素执行Sign操作，Sign是指返回输入数据的符号。 </td>
    </tr>
    <tr>
        <td> Sin </td>
        <td> 按元素做正弦函数计算。 </td>
    </tr>
    <tr>
        <td> Sinh </td>
        <td> 按元素做双曲正弦函数计算。 </td>
    </tr>
    <tr>
        <td> Tan </td>
        <td> 按元素做正切函数计算。 </td>
    </tr>
    <tr>
        <td> Tanh </td>
        <td> 按元素做逻辑回归Tanh。 </td>
    </tr>
    <tr>
        <td> Trunc </td>
        <td> 按元素做浮点数截断操作，即向零取整操作。 </td>
    </tr>
    <tr>
        <td> Xor </td>
        <td> 按元素执行Xor（异或）运算。 </td>
    </tr>
    <tr>
        <th rowspan="3"> 量化反量化 </th>
        <td> AscendAntiQuant </td>
        <td> 按元素做伪量化计算，比如将int8_t数据类型伪量化为half数据类型。 </td>
    </tr>
    <tr>
        <td> AscendDequant </td>
        <td> 按元素做反量化计算，比如将int32_t数据类型反量化为half/float等数据类型。 </td>
    </tr>
    <tr>
        <td> AscendQuant </td>
        <td> 按元素做量化计算，比如将half/float数据类型量化为int8_t数据类型。 </td>
    </tr>
    <tr>
        <th rowspan="10"> 数据归一化 </th>
        <td> BatchNorm </td>
        <td> 对于每个batch中的样本，对其输入的每个特征在batch的维度上进行归一化。 </td>
    </tr>
    <tr>
        <td> DeepNorm </td>
        <td> 在深层神经网络训练过程中，可以替代LayerNorm的一种归一化方法。 </td>
    </tr>
    <tr>
        <td> LayerNorm </td>
        <td> 将输入数据收敛到[0, 1]之间，可以规范网络层输入输出数据分布的一种归一化方法。 </td>
    </tr>
    <tr>
        <td> LayerNormGrad </td>
        <td> 用于计算LayerNorm的反向传播梯度。 </td>
    </tr>
    <tr>
        <td> LayerNormGradBeta </td>
        <td> 用于获取反向beta/gmma的数值，和LayerNormGrad共同输出pdx, gmma和beta。 </td>
    </tr>
    <tr>
        <td> RmsNorm </td>
        <td> 实现对shape大小为[B，S，H]的输入数据的RmsNorm归一化。 </td>
    </tr>
    <tr>
        <td> GroupNorm </td>
        <td> 对输入数据在 channel 维度进行分组并对每个组做归一化的方法。 </td>
    </tr>
    <tr>
        <td> Normalize </td>
        <td> 已知均值和方差，计算shape为[A, R]的输入数据的标准差倒数rstd和归一化结果y的方法。 </td>
    </tr>
    <tr>
        <td> WelfordUpdate </td>
        <td> Welford算法的前处理，一种在线计算均值和方差的方法。 </td>
    </tr>
    <tr>
        <td> WelfordFinalize </td>
        <td> Welford算法的后处理，一种在线计算均值和方差的方法。 </td>
    </tr>
    <tr>
        <th rowspan="18"> 激活函数 </th>
        <td> AdjustSoftMaxRes </td>
        <td> 用于对SoftMax相关计算结果做后处理，调整SoftMax的计算结果为指定的值。 </td>
    </tr>
    <tr>
        <td> FasterGelu </td>
        <td> FastGelu化简版本的一种激活函数。 </td>
    </tr>
    <tr>
        <td> FasterGeluV2 </td>
        <td> FastGeluV2版本，可以降低GELU的算力需求。 </td>
    </tr>
    <tr>
        <td> GeGLU </td>
        <td> 采用GELU作为激活函数的GLU变体。 </td>
    </tr>
    <tr>
        <td> Gelu </td>
        <td> GELU是一个重要的激活函数，其灵感来源于Relu和Dropout，在激活中引入了随机正则的思想。 </td>
    </tr>
    <tr>
        <td> LogSoftMax </td>
        <td> 对输入tensor做LogSoftmax计算。 </td>
    </tr>
    <tr>
        <td> ReGlu </td>
        <td> 一种GLU变体，使用Relu作为激活函数。 </td>
    </tr>
    <tr>
        <td> Sigmoid </td>
        <td> 按元素做逻辑回归Sigmoid。 </td>
    </tr>
    <tr>
        <td> Silu </td>
        <td> 按元素做Silu运算。 </td>
    </tr>
    <tr>
        <td> SimpleSoftMax </td>
        <td> 使用计算好的sum和max数据对输入tensor做Softmax计算。 </td>
    </tr>
    <tr>
        <td> SoftMax </td>
        <td> 对输入tensor按行做Softmax计算。 </td>
    </tr>
    <tr>
        <td> SoftmaxFlash </td>
        <td> Softmax增强版本，除了可以对输入tensor做SoftmaxFlash计算，还可以根据上一次Softmax计算的sum和max来更新本次的Softmax计算结果。 </td>
    </tr>
    <tr>
        <td> SoftmaxFlashV2 </td>
        <td> SoftmaxFlash增强版本，对应FlashAttention-2算法。 </td>
    </tr>
    <tr>
        <td> SoftmaxFlashV3 </td>
        <td> SoftmaxFlash增强版本，对应Softmax PASA算法。 </td>
    </tr>
    <tr>
        <td> SoftmaxGrad </td>
        <td> 对输入tensor做grad反向计算的一种方法。 </td>
    </tr>
    <tr>
        <td> SoftmaxGradFront </td>
        <td> 对输入tensor做grad反向计算的一种方法，其中dstTensor的last长度固定为32Byte。 </td>
    </tr>
    <tr>
        <td> SwiGLU </td>
        <td> 采用Swish作为激活函数的GLU变体。 </td>
    </tr>
    <tr>
        <td> Swish </td>
        <td> 神经网络中的Swish激活函数。 </td>
    </tr>
    <tr>
        <th rowspan="10"> 归约操作 </th>
        <td> Mean </td>
        <td> 根据最后一轴的方向对各元素求平均值。 </td>
    </tr>
    <tr>
        <td> ReduceXorSum </td>
        <td> 按照元素执行Xor（按位异或）运算，并将计算结果ReduceSum求和。 </td>
    </tr>
    <tr>
        <td> Sum </td>
        <td> 获取最后一个维度的元素总和。 </td>
    </tr>
    <tr>
        <td> ReduceSum </td>
        <td> 对一个多维向量按照指定的维度进行数据累加。 </td>
    </tr>
        <tr>
        <td> ReduceMean </td>
        <td> 对一个多维向量按照指定的维度求平均值。 </td>
    </tr>
        <tr>
        <td> ReduceProd </td>
        <td> 对一个多维向量按照指定的维度求积。 </td>
    </tr>
        <tr>
        <td> ReduceMax </td>
        <td> 对一个多维向量按照指定的维度求最大值。 </td>
    </tr>
        <tr>
        <td> ReduceMin </td>
        <td> 对一个多维向量按照指定的维度求最小值。 </td>
    </tr>
        <tr>
        <td> ReduceAny </td>
        <td> 对一个多维向量按照指定的维度求逻辑或。 </td>
    </tr>
        <tr>
        <td> ReduceAll </td>
        <td> 对一个多维向量按照指定的维度求逻辑与。 </td>
    </tr>
    <tr>
        <th rowspan="2"> 排序 </th>
        <td> TopK </td>
        <td> 获取最后一个维度的前k个最大值或最小值及其对应的索引。 </td>
    </tr>
    <tr>
        <td> Sort </td>
        <td> 对输入tensor做Sort计算，按照数值大小进行降序排序。 </td>
    </tr>
    <tr>
        <th rowspan="3"> 数据填充 </th>
        <td> Broadcast </td>
        <td> 将输入按照输出shape进行广播。 </td>
    </tr>
    <tr>
        <td> Pad </td>
        <td> 对height * width的二维Tensor在width方向上pad到32B对齐。 </td>
    </tr>
    <tr>
        <td> UnPad </td>
        <td> 对height * width的二维Tensor在width方向上进行unpad。 </td>
    </tr>
    <tr>
        <th rowspan="1"> 数据过滤 </th>
        <td> Dropout </td>
        <td> 提供根据MaskTensor对源操作数进行过滤的功能，得到目的操作数。 </td>
    </tr>
    <tr>
        <th rowspan="1"> 比较选择 </th>
        <td> SelectWithBytesMask </td>
        <td> 给定两个源操作数src0和src1，根据maskTensor相应位置的值(非bit位)选取元素，得到目的操作数dst。 </td>
    </tr>
    <tr>
        <th rowspan="2"> 变形 </th>
        <td> ConfusionTranspose </td>
        <td> 对输入数据进行数据排布及Reshape操作。 </td>
    </tr>
    <tr>
        <td> TransData </td>
        <td> 对输入数据排布格式转换为输出所需的数据排布格式 </td>
    </tr>
    <tr>
        <th rowspan="1"> 索引操作 </th>
        <td> ArithProgression </td>
        <td> 给定起始值，等差值和长度，返回一个等差数列。 </td>
    </tr>
    <tr>
        <th rowspan="1"> Matmul </th>
        <td> Matmul </td>
        <td> Matmul矩阵乘法的运算。 </td>
    </tr>
    <tr>
        <th rowspan="1"> 工具类 </th>
        <td> InitGlobalMemory </td>
        <td> 将Global Memory上的数据初始化为指定值。 </td>
    </tr>
</table>
