# Features Sample Introduction

## Overview

Based on Ascend C feature samples, this document introduces features such as Aclnn (GE graph entry) projects, LocalMemoryAllocator, and Barrier separate memory allocation.

## Sample List

| Directory Name | Description |
| -------------- | ----------- |
| [00_compilation](./00_compilation) | This sample introduces the implementation of custom operator compilation projects and static Aclnn invocation |
| [01_invocation](./01_invocation) | This sample introduces the implementation of Aclnn and Aclop operator invocation |
| [02_framework](./02_framework) | This sample introduces custom operator implementation methods for PyTorch, TensorFlow, and ONNX frameworks |
| [03_basic_api](./03_basic_api) | This sample introduces the usage of Ascend C Basic API, including data movement, matrix computation, memory vector computation, resource management, synchronization control, system access, atomic operations, and scalar computation |
| [04_aicpu](./04_aicpu) | This sample introduces the implementation of using AI CPU operators for Tiling sink computation |
| [05_tensor_api](./05_tensor_api) | This sample demonstrates the implementation of using TENSOR_API to build Matmul operators in multi-core scenarios, with Relu and inline quantization enabled |
| [06_aclrtc](./06_aclrtc) | This sample demonstrates using Aclrtc (runtime compilation) interfaces to compile Ascend C kernel functions in string form on the Host side and complete the loading and execution of compilation results |