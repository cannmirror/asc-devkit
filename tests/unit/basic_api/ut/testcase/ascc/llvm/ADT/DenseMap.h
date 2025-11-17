/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef LLVM_ADT_DENSE_MAP_H
#define LLVM_ADT_DENSE_MAP_H
#include <vector>

namespace llvm {
template <typename KeyT, typename ValueT>
class DenseMap {
public:
    using BucketT = std::pair<KeyT, ValueT>;
    std::vector<BucketT> Buckets;

    size_t size() const {
        return Buckets.size();
    }

    void clear() {
        Buckets.clear();
    }

    ValueT& operator[](const KeyT& Key) {
        for (auto& Bucket : Buckets) {
            if (Bucket.first == Key)
                return Bucket.second;
        }
        Buckets.push_back({Key, ValueT()});
        return Buckets.back().second;
    }

    class iterator {
        BucketT* Ptr;
    public:
        iterator(BucketT* P) : Ptr(P) {}
        BucketT& operator*() const { return *Ptr; }
        BucketT* operator->() { return Ptr; }
        bool operator!=(const iterator& RHS) const { return Ptr != RHS.Ptr; }
    };

    iterator find(const KeyT& Key) {
        for (auto& Bucket : Buckets) {
            if (Bucket.first == Key)
                return iterator(&Bucket);
        }
        return end();
    }

    iterator end() const {
        return iterator(nullptr);
    }
};
}
#endif