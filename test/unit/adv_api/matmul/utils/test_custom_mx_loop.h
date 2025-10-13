#include <cstdint>
#include "test_custom_loop.h"

namespace TestCustomModules {
template <typename IMPL, class A_TYPE, const auto &MM_CFG, typename = void>
class CustomMxLoop : public CustomLoop<IMPL, A_TYPE, MM_CFG>
{
public:
    __aicore__ inline int32_t GetOuterScaleKaIdx()
    {
        return 0;
    }
};

template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG, typename = void>
class CustomMxKLoop : public CustomKLoop<IMPL, TRANS_T, A_TYPE, MM_CFG> {
public:
    __aicore__ inline int32_t GetScaleFactorKa() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetScaleFactorKb() const
    {
        return 1;
    }

    __aicore__ inline int32_t GetNextOuterScaleKaIdx() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetNextOuterScaleKbIdx() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetOuterScaleKaIdx()
    {
        return 1;
    }

    __aicore__ inline int32_t GetOuterScaleKbIdx()
    {
        return 1;
    }

    __aicore__ inline int32_t GetTileShapeScaleA() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetTileShapeScaleB() const
    {
        return 0;
    }

    __aicore__ inline bool IsAKL1FullLoad() const
    {
        return false;
    }

    __aicore__ inline bool IsScaleAKL1FullLoad() const
    {
        return false;
    }

    __aicore__ inline bool IsBKL1FullLoad() const
    {
        return false;
    }

    __aicore__ inline bool IsScaleBKL1FullLoad() const
    {
        return false;
    }

    __aicore__ inline int32_t GetStepInnerIdx() const
    {
        return 0;
    }
};
}
