# Profiling Sample Introduction

## Overview

This sample demonstrates how to collect performance data using Profiling tools.

## Sample List

| Directory Name | Description |
| -------------- | ----------- |
| [msProf](./msProf) | This sample implements MatmulLeakyRelu using Ascend C programming language and uses the msProf tool for debugging, providing the corresponding end-to-end implementation |
| [torch_library_report_tensor](./torch_library_report_tensor) | This sample demonstrates how to integrate Profiling and collect Shape information for the Add operator when calling kernel functions using the `<<<>>>` kernel invocation operator |