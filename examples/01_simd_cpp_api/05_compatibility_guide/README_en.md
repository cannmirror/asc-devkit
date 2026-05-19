# Compatibility Guide Sample Introduction

## Overview

This section provides several samples for migrating features that are incompatible with Atlas A2 training series products, Atlas A2 inference series products, and Ascend 950PR/Ascend 950DT. Users can perform migration based on these samples.

The samples in this section use the `<<<>>>` kernel call operator to complete the basic process of running operator kernel functions on the NPU side, providing corresponding end-to-end implementations.

## Sample List

| Directory Name | Description |
| -------------------------------------------------- | ---------------------------------------------------- |
| [data_copy_l1togm](./data_copy_l1togm) | This sample demonstrates the end-to-end process of copying data from L1 to GM. |
| [fill](./fill) | This sample demonstrates how to use the Fill interface to initialize L0A Buffer and L0B Buffer. |
| [matmul_s4_910B](./matmul_s4_910B) | This sample directly uses the Matmul high-level API for matrix computation. |
| [matmul_s4_950](./matmul_s4_950) | The new architecture removes the int4b_t data type from the Cube computation unit. Users can perform Cast conversion from int4b_t to int8_t on the Vector Core in MIX mode on the operator side, then transfer to L1 via UB for Mmad computation. |
| [pattern_transformation](./pattern_transformation) | Based on the basic mmad sample, this sample demonstrates the fractal transformation logic for the L1 Buffer to L0A Buffer path. |
| [scatter](./scatter) | This sample demonstrates the data scatter function, which scatters an input tensor to a result tensor based on the input tensor and destination address offset tensor. |
| [set_loaddata_boundary](./set_loaddata_boundary) | This sample implements setting boundary values for L1 Buffer. |