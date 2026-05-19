# Reg Vector Compute Practices Example Introduction

## Overview

VF-based performance optimization examples using the <<<>>> direct invocation implementation method, introducing VF loop optimization, VF instruction dual-issue optimization, VF continuous non-aligned scenario optimization, and VF fusion optimization methods.

## Example List

|  Directory Name  |  Description  |
| -------------------------------------------------- | ---------------------------------------------------- |
| [optimize_vf_continious_align](./optimize_vf_continious_align) | This example demonstrates operator implementation with transfer optimization using continuous non-aligned transfer interfaces LoadUnAlign/StoreUnAlign in SIMD scenarios. |
| [optimize_vf_dual_instr](./optimize_vf_dual_instr) | This example demonstrates VF instruction dual-issue optimization based on the Reg programming interface in SIMD scenarios. By properly splitting VF loops and appropriately moving intermediate results to UB, data dependencies are reduced. |
| [optimize_vf_fusion](./optimize_vf_fusion) | This example demonstrates VF fusion optimization for operator code implementation based on the Reg programming interface in SIMD scenarios. |
| [optimize_vf_loop](./optimize_vf_loop) | Optimize VF loops through loop member variable access optimization, loop instruction distribution optimization, loop address management optimization, and other methods. |
| [gelu_high_performance](./gelu_high_performance) | This example uses Gelu computation to introduce RegBase vector performance tuning methods, demonstrating performance gains after enabling VF fusion. |