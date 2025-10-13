#include <cstdint>

namespace TestCustomModules {
template <typename IMPL, class A_TYPE, const auto &MM_CFG, typename = void>
class CustomLoop
{
public:
    __aicore__ inline int32_t GetTotalIter()
    {
        return 2;
    }

    __aicore__ inline bool OuterNext()
    {
        return false;
    }

    __aicore__ inline void OuterStart() {}

    __aicore__ inline bool OuterEnd()
    {
        return true;
    }

    __aicore__ inline int32_t GetOuterIdx()
    {
        return 0;
    }

    __aicore__ inline int32_t GetOuterIter()
    {
        return 1;
    }

    __aicore__ inline int32_t GetTileShape()
    {
        return 64;
    }

    __aicore__ inline int32_t GetTileShapeOf(uint32_t idx)
    {
        return 64;
    }

    __aicore__ inline int32_t GetTileBlockShape()
    {
        return 4;
    }

    __aicore__ inline bool InnerNext()
    {
        return false;
    }

    __aicore__ inline void InnerStart() {}

    __aicore__ inline bool InnerEnd()
    {
        return true;
    }

    __aicore__ inline bool IsLastInnerIter()
    {
        return true;
    }

    __aicore__ inline int32_t GetInnerIdx()
    {
        return 0;
    }

    __aicore__ inline int32_t GetInnerIter()
    {
        return 2;
    }

    __aicore__ inline int32_t GetBaseShape()
    {
        return 32;
    }

    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return 2;
    }

    __aicore__ inline bool IsLastOuterIter() const
    {
        return true;
    }

    __aicore__ inline bool IsAML1FullLoad() const
    {
        return false;
    }

    __aicore__ inline bool IsBNL1FullLoad() const
    {
        return false;
    }
};

template <typename IMPL, typename TRANS_T, class A_TYPE, const auto& MM_CFG, typename = void>
class CustomKLoop {
public:
    __aicore__ inline uint32_t GetTotalIter()
    {
        return 2;
    }

    __aicore__ inline void OuterStart() {}

    __aicore__ inline bool OuterNext()
    {
        return false;
    }

    __aicore__ inline bool OuterEnd()
    {
        return true;
    }

    __aicore__ inline uint32_t GetOuterIdx() const
    {
        return 0;
    }

    __aicore__ inline uint32_t GetOuterIter() const
    {
        return 1;
    }

    __aicore__ inline int32_t InnerStart()
    {
        return 0;
    }

    __aicore__ inline bool InnerNext()
    {
        return false;
    }

    __aicore__ inline bool InnerEnd()
    {
        return true;
    }

    __aicore__ inline bool FirstOuterIter() const
    {
        return true;
    }

    __aicore__ inline bool LastOuterIter() const
    {
        return true;
    }

    __aicore__ inline bool FirstInnerIter() const
    {
        return true;
    }

    __aicore__ inline uint32_t GetInnerIdx() const
    {
        return 0;
    }

    __aicore__ inline uint32_t GetInnerStartIdx() const
    {
        return 0;
    }

    __aicore__ inline uint32_t GetOuterKaIdx() const
    {
        return 0;
    }

    __aicore__ inline uint32_t GetOuterKbIdx() const
    {
        return 0;
    }

    __aicore__ inline uint32_t GetNextOuterKaIdx() const
    {
        return 1;
    }

    __aicore__ inline uint32_t GetNextOuterKbIdx() const
    {
        return 1;
    }

    __aicore__ inline uint32_t GetInnerIter() const
    {
        return 0;
    }

    __aicore__ inline int32_t GetTileShapeA() const
    {
        return 64;
    }

    __aicore__ inline int32_t GetTileShapeAof(int32_t kIdx) const
    {
        return 64;
    }

    __aicore__ inline int32_t GetTileShapeB() const
    {
        return 64;
    }

    __aicore__ inline int32_t GetTileShapeBOf(int32_t kIdx) const
    {
        return 64;
    }

    __aicore__ inline int32_t GetTileBlockShapeA() const
    {
        return 4;
    }

    __aicore__ inline int32_t GetTileBlockShapeB() const
    {
        return 4;
    }

    __aicore__ inline int32_t GetBaseShape() const
    {
        return 32;
    }

    __aicore__ inline int32_t GetBaseBlockShape() const
    {
        return 2;
    }

    __aicore__ inline bool IsAKL1FullLoad() const
    {
        return true;
    }

    __aicore__ inline bool IsBKL1FullLoad() const
    {
        return true;
    }
};

}
