# Activation Function Operator Examples

## Overview

This example set demonstrates typical usage of activation function high-level APIs, with each example containing a complete end-to-end implementation.

## Example List

<div align="left">

|  Directory Name                                                   |  Description                                              |
| :------------------------------------------------------------ | :---------------------------------------------------- |
| [geglu](./geglu) | This example demonstrates the implementation based on the GeGLU high-level API, supporting GLU variant scenarios where GELU activation is used as the activation function |
| [gelu](./gelu) | This example demonstrates the operator implementation based on the Gelu high-level API. The example performs GELU activation computation on the input Tensor element-wise |
| [logsoftmax](./logsoftmax) | This example demonstrates the operator implementation based on the LogSoftMax high-level API. The example performs LogSoftmax computation on the input tensor |
| [sigmoid](./sigmoid) | This example demonstrates the operator implementation based on the Sigmoid high-level API. The example performs logistic regression Sigmoid element-wise |
| [softmax](./softmax) | This example introduces calling the SoftMax high-level API to implement a single softmax operator. For an input tensor[m0, m1, ...mt, n] (where t is greater than or equal to 0), the product of non-last axis lengths is treated as m, so the input tensor shape is viewed as [m, n] |
| [softmaxflashv2](./softmaxflashv2) | This example introduces calling the SoftmaxFlashV2 high-level API to implement a single softmaxflashv2 operator, which is an enhanced version of SoftmaxFlash corresponding to the FlashAttention-2 algorithm |
| [softmaxflashv3](./softmaxflashv3) | This example introduces calling the SoftmaxFlashV3 high-level API to implement a single softmaxflashv3 operator, which is an enhanced version of SoftmaxFlash corresponding to the Softmax PASA algorithm |
| [softmaxgrad](./softmaxgrad) | This example introduces calling the SoftmaxGrad high-level API to implement a single softmaxgrad operator. For an input tensor[m0, m1, ...mt, n] (where t is greater than or equal to 0), the product of non-last axis lengths is treated as m, so the input tensor shape is viewed as [m, n] |
| [swish](./swish) | This example demonstrates the operator implementation based on the Swish/Silu high-level APIs, with compilation macros to switch between two modes. Swish performs activation computation element-wise, and Silu is a special case of Swish with beta=1 |
| [swiglu](./swiglu) | This example demonstrates the operator implementation based on the SwiGLU high-level API. The example uses Swish as the activation function for the GLU variant |

</div>