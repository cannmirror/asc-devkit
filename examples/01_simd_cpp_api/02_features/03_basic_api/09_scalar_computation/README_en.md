# Scalar Computation API Sample Introduction

## Overview

This directory contains API samples related to scalar computation. The samples are based on the Ascend C `<<<>>>` direct invocation method, supporting the implementation of both the main function and kernel function in the same file.

## Sample List

| Directory Name | Description |
| ------- | -------- |
| [gm_by_pass_dcache](./gm_by_pass_dcache)             | This sample demonstrates reading data from and writing data to GM without going through DCache, based on the ReadGmByPassDcache and WriteGmByPassDcache interfaces. |