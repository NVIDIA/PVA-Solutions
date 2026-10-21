/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "longlong_div.h"

#include "limits.h"
const int N_WORD_BITS = 64;

inline uint64_t div_pos(uint64_t x, uint64_t d, chess_output uint64_t &r)
//  unsigned division, 'x' divided by 'd' with remainder 'r'.
{
    if ((int64_t)d < 0)
    { // MSB of 'd' is one
        uint64_t q = 0;
        r          = x;
        if (r >= d)
        {
            r = r - d;
            q = 1;
        }
        return q;
    }
    // MSB of 'd' is zero
    r             = 0;
    uint64_t wrap = 0;
    for (int i = 0; i < N_WORD_BITS; i++)
    {
        r    = (r << 1) | chess_dont_warn_range(x >> (N_WORD_BITS - 1));
        wrap = ((int64_t)(d - r - 1) >> (N_WORD_BITS - 1)); // wrap = (d > r)?0: 0xffffffffffffffff (via sign extension)
        r -= d & wrap;                                      // if (wrap) r -= d;
        x = (x << 1) | (wrap & 1);                          // if (wrap) x = (x<<1) | 1;
    }
    return x;
}

uint64_t ll_udiv(uint64_t x, uint64_t d, chess_output uint64_t &r)
{
    return div_pos(x, d, r);
}

int64_t ll_sdiv(int64_t x, int64_t d, chess_output int64_t &r)
//  signed division, 'x' divided by 'd' with remainder 'r'.
{
    uint64_t xu = x, du = d, ru;
    if (x < 0)
        xu = -xu;
    if (d < 0)
        du = -du;
    uint64_t qu = div_pos(xu, du, ru);
    r           = ru;
    if (x < 0)
        r = -r;
    return ((x ^ d) < 0) ? -qu : qu;
}
