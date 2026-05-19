# Data Movement API Examples Introduction

## Overview

This directory contains examples for multiple APIs related to data movement. Each example is based on the Ascend C `<<<>>>` direct call method, supporting the implementation of both the main function and kernel function in the same cpp file.

## Example List

| Directory Name | Function Description |
| ----------------------------------------------------------| -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [broadcast_ub2l0c](./broadcast_ub2l0c) | This example implements data broadcast movement based on BroadCastVecToMM, broadcasting data located on UB (Unified Buffer) and moving it to CO1 (L0C Buffer) |
| [copy_ub2ub](./copy_ub2ub) | This example implements data movement based on Copy, applicable for data movement between VECIN and VECOUT, supporting mask continuous mode and counter mode |
| [data_copy_gm2ub_slice](./data_copy_gm2ub_slice) | This example implements data slice movement based on DataCopy, extracting subsets of multi-dimensional Tensor data for movement between GM (Global Memory) and UB (Unified Buffer) pathways |
| [data_copy_gm2ub_nddma](./data_copy_gm2ub_nddma) | This example introduces how to use the multi-dimensional data movement interface to implement data movement from GM (Global Memory) to UB (Unified Buffer) pathway. By freely configuring the input dimension information and corresponding Stride, it can be used for various data transformation operations such as Padding, Transpose, BroadCast, Slice, etc. |
| [data_copy_l0c2gm](./data_copy_l0c2gm) | This example implements data inline quantization activation movement based on DataCopy in convolution scenarios |
| [data_copy_pad_gm2ub_ub2gm](./data_copy_pad_gm2ub_ub2gm) | This example implements movement of non-32-byte aligned data based on DataCopyPad, with data padding |
| [data_copy_ub2l1](./data_copy_ub2l1) | This example implements data movement from UB (Unified Buffer) to L1 (L1 Buffer) based on DataCopy in Mmad matrix multiplication scenarios |
| [ld_st_reg_mask](./ld_st_reg_mask) | This example implements loading and storing from UB (Unified Buffer) to MaskReg (mask register) based on the Reg programming interface, as well as operations using mask for masked store |
| [ld_st_reg_align](./ld_st_reg_align) | This example implements continuous and non-continuous aligned data movement operations from UB (Unified Buffer) to RegTensor (Reg vector computation basic unit) based on the Reg programming interface |
| [ld_st_reg_unalign](./ld_st_reg_unalign) | This example implements unaligned data movement operations from UB (Unified Buffer) to RegTensor (Reg vector computation basic unit) based on the Reg programming interface |
| [gather_ld_reg](./gather_ld_reg) | This example demonstrates using the Gather interface to implement discrete data load, including two scenarios: high-dimensional Gather (source is LocalTensor) and Reg::GatherB (collection by DataBlock) |
| [scatter_st_reg](./scatter_st_reg) | This example demonstrates using the Reg::Scatter interface to implement discrete data store (dispersing elements to UB) |
| [auxscalar_reg](./auxscalar_reg) | This example demonstrates using the AuxScalar method to read multiple scalar data from UB for computation |