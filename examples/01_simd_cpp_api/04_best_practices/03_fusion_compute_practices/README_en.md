# Fusion Compute Practices Sample Introduction
## Overview
Operator samples implemented using SIMT and SIMD hybrid programming, introducing SIMT-based flexible branch logic implementation, as well as high-performance Matmul fusion operator implementation and performance optimization methods using UB to improve discrete memory access efficiency.
 
## Sample List
|  Directory Name                                                   |  Description                                              |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| [grouped_matmul](./grouped_matmul) | This sample introduces high-performance implementation of QuantGroupMatmul operator on NPU, supporting grouped quantization matrix multiplication and Gelu activation computation. |
| [simt_and_simd_floor_mod](./simt_and_simd_floor_mod) | Operator sample implemented using SIMT and SIMD hybrid programming, introducing SIMT-based flexible branch logic implementation. |
| [simt_gather_with_ub](./simt_gather_with_ub) | This sample uses the Gather operator as an example to demonstrate performance optimization using UB to improve discrete memory access efficiency in SIMD and SIMT hybrid programming mode |