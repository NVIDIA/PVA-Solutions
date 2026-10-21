/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "TileSizeEnumeration.hpp"

#include <iostream>
#include <numeric>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace pva_solutions::priv {

// factor (n) into prime numbers, 1 is included
// e.g. factor(100) will generate std::vector<int>{1,2,2,5,5}
auto factor(int input) -> std::vector<int>
{
    std::vector<int> output{1};

    while (input % 2 == 0)
    {
        output.push_back(2);
        input /= 2;
    }

    for (int i = 3; i * i <= input; i += 2)
    {
        while (true)
        {
            auto dv = std::div(input, i);
            if (dv.rem != 0)
            {
                break;
            }
            output.push_back(i);
            input = dv.quot;
        }
    }

    if (input > 1)
    {
        output.push_back(input);
    }

    return output;
}

// Generate full combination, choose k from n, where k is implicitly set by
// seq.size(). (seq) should be initialized with 0,1,2,...,k-1.
// It returns true if there exists a combination for the next call,
// which is similar to that of std::next_permutation().
auto next_combination(std::vector<int> &seq, uint32_t n) -> bool
{
    int k = static_cast<int>(seq.size());
    for (int i = k - 1; i >= 0; i--)
    {
        if (seq[i] < n - k + i)
        {
            seq[i]++;
            for (int j = i + 1; j < k; j++)
            {
                seq[j] = seq[j - 1] + 1;
            }
            return true;
        }
    }
    return false;
}

auto enumerateTileSz(int32_t width, int32_t height, int32_t blkSz, int32_t maxTileSz)
    -> std::optional<std::pair<int32_t, int32_t>>
{
    std::vector<int> factor0 = factor(width);
    std::vector<int> factor1 = factor(height);
    auto len0                = factor0.size();
    auto len1                = factor1.size();
    auto len                 = len0 + len1;

    // simple metric to compare which tile is better
    // 1) the larger tile size
    // 2) the larger tileX is the tile size is the same
    struct CmpMetric
    {
        bool operator()(const std::pair<int32_t, int32_t> &lhs, const std::pair<int32_t, int32_t> &rhs) const
        {
            int lhsTileSz = lhs.first * lhs.second;
            int rhsTileSz = rhs.first * rhs.second;
            if (lhsTileSz < rhsTileSz || (lhsTileSz == rhsTileSz && lhs.first < rhs.first))
            {
                return true;
            }
            else
            {
                return false;
            }
        }
    };

    std::set<std::pair<int32_t, int32_t>, CmpMetric> candidate;
    for (int k = len - 1; k >= 1; k--)
    {
        std::vector<int> seq(k);
        std::iota(std::begin(seq), std::end(seq), 0);
        do
        {
            bool bHasWidth{false};
            bool bHasHeight{false};
            int32_t tileX{1};
            int32_t tileY{1};
            for (int i = 0; i < k; i++)
            {
                // (seq) is the index vector, need to get the actual factored number
                int index = seq[i];
                // value 0~(len0-1) --> width, otherwise --> height
                if (index < len0)
                {
                    bHasWidth = true;
                    tileX *= factor0[index];
                }
                else
                {
                    bHasHeight = true;
                    tileY *= factor1[index - len0];
                }
            }

            if (bHasWidth && bHasHeight && (blkSz * blkSz * tileX * tileY <= maxTileSz))
            {
                candidate.insert(std::make_pair(tileX, tileY));
            }
        } while (next_combination(seq, len));
    }

    if (candidate.empty())
    {
        return {};
    }
    else
    {
        auto it = candidate.rbegin();
        return std::make_pair(it->first, it->second);
    }
}

} // namespace pva_solutions::priv
