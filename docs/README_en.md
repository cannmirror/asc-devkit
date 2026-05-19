# Project Documentation

## Directory Description
Key directory structure is as follows:
```
├── api                            # API documentation directory
│   ├── context                    # Documentation for each API
│   └── README.md                  # Ascend C API list
├── figures                        # Image directory
├── asc_adv_api_contributing.md    # Ascend C high-level API contribution guide
├── asc_basic_api_contributing.md  # Ascend C basic API contribution guide
├── asc_c_api_contributing.md      # Ascend C C API contribution guide
├── quick_start.md                 # Quick start documentation
└── README.md
```

## Documentation Description
To help developers quickly familiarize with this project, corresponding documentation can be obtained as needed. Documentation content includes:

| Document | Target Audience | Content Introduction |
|---|---|---|
| [API List](./api/README.md) | Users developing customized APIs or operators based on Ascend C open source repository. | Introduces all APIs included in the project. |
| [High-level API Contribution Guide](./en/asc_adv_api_contributing.md) | Users developing customized APIs based on Ascend C open source repository. | Introduces how to extend or develop Ascend C high-level API. High-level API abstracts and encapsulates common algorithms based on single-core, implementing commonly used computational algorithms to improve operator development efficiency. |
| [Basic API Contribution Guide](./en/asc_basic_api_contributing.md) | Users developing customized APIs based on Ascend C open source repository. | Introduces how to extend or develop Ascend C basic API. Basic API implements abstraction of hardware capabilities, opening chip capabilities, ensuring completeness and compatibility. |
| [C API Contribution Guide](./en/asc_c_api_contributing.md) | Users developing customized APIs based on Ascend C open source repository. | Introduces how to extend or develop Ascend C C API. C API provides pure C style interfaces, conforming to C language operator development habits, opening complete chip programming capabilities. |
| [Ascend C Programming Guide](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC) | Developers writing operator programs using Ascend C based on Ascend AI hardware, developing custom operators. | Ascend C is a programming language launched by CANN for operator development scenarios, with native support for C and C++ standard specifications, balancing development efficiency and runtime performance. Write operator programs using Ascend C, running on Ascend AI processors to implement custom innovative algorithms. |
| [Ascend C Best Practices](https://hiascend.com/document/redirect/CannCommunityAscendCBestPractice) | Developers needing to further optimize operator performance based on already developed Ascend C operators. | Characteristics of heterogeneous computing, operator debugging methods, and operator performance optimization strategies. Introduces debugging and optimization concepts in Ascend C programming, combining various performance optimization methods with specific cases to help developers achieve high-performance operator development. |


## Appendix
Besides the systematic development documentation introduced above, you can also selectively learn about corresponding specialized content based on actual scenarios and development stages.
- Technical Articles  
  - Basics Introduction
    - [Ascend C Programming Introduction](https://www.hiascend.com/zh/developer/techArticles/20230830-1)
    - [Ascend C Quick Start](https://www.hiascend.com/zh/developer/techArticles/20230830-2)
    - [Ascend C Twin Debugging](https://www.hiascend.com/zh/developer/techArticles/20231215-2)
    - [Ascend C Operator Invocation Methods](https://www.hiascend.com/zh/developer/techArticles/20240523-1)
  - Concept Principles
    - [Ascend C Non-aligned Data Processing Solutions](https://www.hiascend.com/zh/developer/techArticles/20250627-1)
    - [Deep Understanding of Multi-core Parallel/Pipeline Computing/Double Buffer Technology](https://www.hiascend.com/zh/developer/techArticles/20230807-1)
  - Problem Cases
    - [Ascend C Operator Development Common Problem Cases](https://www.hiascend.com/zh/developer/techArticles/20240106-1) 
    - [Locating Precision Issues in Operators Containing Matmul High-level API](https://www.hiascend.com/zh/developer/techArticles/20250107-1)
  - Performance Optimization
    - [Ascend C Operator Performance Optimization Practical Tips 01 - Pipeline Optimization](https://www.hiascend.com/zh/developer/techArticles/20240819-1)
    - [Ascend C Operator Performance Optimization Practical Tips 02 - Memory Optimization](https://www.hiascend.com/zh/developer/techArticles/20240823-1)
    - [Ascend C Operator Performance Optimization Practical Tips 03 - Data Movement Optimization](https://www.hiascend.com/zh/developer/techArticles/20240906-1)
    - [Ascend C Operator Performance Optimization Practical Tips 04 - Tiling Optimization](https://www.hiascend.com/zh/developer/techArticles/20240920-1)
    - [Ascend C Operator Performance Optimization Practical Tips 05 - API Usage Optimization](https://www.hiascend.com/zh/developer/techArticles/20241107-1)
  - Best Practices
    - [Matmul Operator Performance Optimization Best Practice Based on Ascend C](https://www.hiascend.com/zh/developer/techArticles/20240816-1)
    - [FlashAttention Operator Performance Optimization Best Practice Based on Ascend C](https://www.hiascend.com/zh/developer/techArticles/20240607-1)


- Training Videos
  - [Ascend C Series Tutorial (Introductory)](https://www.hiascend.com/developer/courses/detail/1691696509765107713)
  - [Ascend C Series Tutorial (Advanced)](https://www.hiascend.com/zh/developer/courses/detail/1696414606799486977)
  - [Ascend C Series Tutorial (Expert)](https://www.hiascend.com/zh/developer/courses/detail/1696690858236694530)
