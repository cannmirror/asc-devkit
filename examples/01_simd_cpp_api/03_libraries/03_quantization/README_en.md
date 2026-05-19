# Quantization Operator Examples

## Overview

This example set introduces typical usage of different features of quantization operators and provides corresponding end-to-end implementations.

## Example List

| Directory Name | Description |
| --------------- | ----------- |
| [antiquant](./antiquant) | This example demonstrates implementing the antiquant operator using the AntiQuant high-level API. The example performs pseudo-quantization computation element-wise |
| [dequant](./dequant) | This example introduces implementing the Dequant single operator using the Dequant high-level API based on the Kernel direct call operator project, performing dequantization computation element-wise, for example, dequantizing int32_t data type to half/float and other data types |
| [quant](./quant) | This example introduces implementing the QuantCustom single operator using the Quant high-level API based on the Kernel direct call operator project, performing quantization computation element-wise, for example, quantizing half/float data type to int8_t data type |